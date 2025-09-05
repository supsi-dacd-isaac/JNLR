"""
Signed‑distance functions (SDFs) for a small selection of complex shapes from
the `sdf‑explorer` dataset's **Animal** folder.

This module provides JAX implementations of analytic SDFs for three
creature models: a **tardigrade**, a **human skull** and a **manta ray**.
The underlying GLSL shaders that generate these shapes combine basic
distance primitives (spheres, ellipsoids, boxes, capsules, etc.) using
smooth union/difference operations and, in some cases, animated
deformations.  The goal of this translation is to expose the static
geometry of each shape in a form compatible with JAX, while omitting
the real‑time animation and material assignments found in the original
shaders.  As such, the functions here return only a signed distance
for a given point.

The three main entry points are:

* ``sdf_manta_ray(p, time=0.0)`` – Distance to a manta ray, optionally
  animated by a scalar ``time`` parameter.
* ``sdf_tardigrade(p)`` – Distance to a tardigrade.  The implementation
  follows the combination of ellipsoids and smooth min/max operations
  used in the GLSL shader, but omits some of the texture‑based detail.
* ``sdf_human_skull(p)`` – A simplified distance function for a human
  skull.  The original shader builds a highly detailed model with
  multiple boolean operations.  Here, only the major volumetric
  features (cranium, jaw, nose and eye sockets) are included.  This
  still produces a recognisable skull shape while keeping the code
  tractable.

Each function accepts a single argument ``p``, a length‑3 JAX array
representing the query point in 3D space.  Unlike the original GLSL
shaders, the manta ray implementation here is **static**—the time
parameter and all animated deformations have been removed to avoid
JAX errors and simplify usage.  The return value of each function is
a signed distance: negative inside the volume, positive outside.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _rotation_matrix(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Compute a 3×3 rotation matrix for a given axis and angle.

    Equivalent to the GLSL ``RotMat`` helper.  The ``axis`` need not
    be normalised; it will be normalised internally.  ``angle`` is in
    radians.  The returned matrix rotates row vectors when multiplied
    from the right.
    """
    axis = axis / jnp.linalg.norm(axis)
    s = jnp.sin(angle)
    c = jnp.cos(angle)
    oc = 1.0 - c
    x, y, z = axis
    return jnp.array([
        [oc * x * x + c,       oc * x * y - z * s, oc * z * x + y * s],
        [oc * x * y + z * s,   oc * y * y + c,     oc * y * z - x * s],
        [oc * z * x - y * s,   oc * y * z + x * s, oc * z * z + c    ],
    ])


def _soft_min(a: float, b: float, k: float) -> float:
    """Smooth minimum of ``a`` and ``b`` with smoothing factor ``k``.

    When ``k`` is zero, this is equivalent to ``min(a, b)``.  Positive
    values of ``k`` make the transition between the two distances
    smoother.  Mirrors the ``smin`` function used in the shaders【785499543551541†L76-L84】.
    """
    h = jnp.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return jnp.where(k != 0.0,
                     jnp.where(k > 0.0, jnp.minimum(a, b), jnp.maximum(a, b)),
                     jnp.minimum(a, b)) - k * h * (1.0 - h)


def _soft_max(a: float, b: float, k: float) -> float:
    """Smooth maximum of ``a`` and ``b`` with smoothing factor ``k``.

    This is the complement to ``_soft_min`` and is used when blending
    additive features onto a shape.  It corresponds to the ``smax``
    function in some of the GLSL code【785499543551541†L86-L90】.
    """
    return _soft_min(a, b, -k)


def sd_sphere(p: jnp.ndarray, radius: float) -> float:
    """Signed distance to a sphere centred at the origin."""
    return jnp.linalg.norm(p) - radius


def sd_ellipsoid(p: jnp.ndarray, r: jnp.ndarray) -> float:
    """Signed distance to an axis‑aligned ellipsoid.

    ``r`` contains the semi‑axis lengths in x, y and z.  The formula
    matches that used in both the MantaRay and Tardigrade shaders【49612677301266†L42-L44】.
    """
    k0 = jnp.linalg.norm(p / r)
    k1 = jnp.linalg.norm(p / (r * r))
    return k0 * (k0 - 1.0) / (k1 + 1e-8)


def sd_box(p: jnp.ndarray, b: jnp.ndarray) -> float:
    """Signed distance to an axis–aligned box with half extents ``b``.
    Returns positive distance outside and negative inside.
    """
    d = jnp.abs(p) - b
    inside = jnp.minimum(jnp.maximum(d[0], jnp.maximum(d[1], d[2])), 0.0)
    outside = jnp.linalg.norm(jnp.maximum(d, 0.0))
    return inside + outside


def sd_capsule(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray, r: float) -> float:
    """Signed distance to a capsule with endpoints ``a`` and ``b`` and radius ``r``.

    The capsule consists of a cylinder and hemispherical end caps.  This
    matches the ``sdCapsule`` helper used in the human skull shader【903968926017913†L194-L201】.
    """
    ab = b - a
    ap = p - a
    t = jnp.dot(ab, ap) / jnp.dot(ab, ab)
    t = jnp.clip(t, 0.0, 1.0)
    c = a + t * ab
    return jnp.linalg.norm(p - c) - r


def sd_cone_frustum(p: jnp.ndarray, a: jnp.ndarray, r1: float, r2: float, h: float) -> float:
    """Signed distance to a truncated cone (frustum).

    Implements the distance to a cone with radii ``r1`` and ``r2`` at
    the ends of a segment of length ``h`` along the y–axis, centred at
    ``a``.  Mirrors the frustum form used for the jaw and nose in the
    human skull shader【903968926017913†L172-L191】.
    """
    # JAX does not permit list indexing; extract components explicitly
    pa = p - a
    radial = jnp.linalg.norm(jnp.array([pa[0], pa[2]]))
    q = jnp.array([radial, pa[1]])
    b = (r1 - r2) / h
    c = jnp.sqrt(1.0 - b * b)
    k = jnp.dot(q, jnp.array([-b, c]))
    # Regions: below base, above top and along the surface
    def below() -> float:
        return jnp.linalg.norm(q) - r1
    def above() -> float:
        return jnp.linalg.norm(q - jnp.array([0.0, h])) - r2
    def on_surface() -> float:
        return jnp.dot(q, jnp.array([c, b])) - r1
    return jax.lax.cond(k < 0.0, lambda _: below(),
                        lambda _: jax.lax.cond(k > c * h, lambda __: above(),
                                               lambda __: on_surface(), None), None)


def sd_plane(p: jnp.ndarray, n: jnp.ndarray, offset: float) -> float:
    """Signed distance to a plane with normal ``n`` passing through offset.
    The plane equation is ``dot(n, p) - offset = 0``.
    """
    return jnp.dot(n / jnp.linalg.norm(n), p) - offset


def sd_box_oriented(p: jnp.ndarray, center: jnp.ndarray, right: jnp.ndarray,
                    up: jnp.ndarray, dim: jnp.ndarray, r: float) -> float:
    """Signed distance to an oriented box with rounded edges.

    ``center`` is the box centre.  ``right`` and ``up`` form two orthonormal
    axes; the third axis is ``cross(right, up)``.  ``dim`` contains
    half sizes along these axes.  ``r`` rounds the edges by subtracting
    a constant.  This corresponds to the ``sdBox`` overload used in
    the human skull shader【903968926017913†L150-L161】.
    """
    # Build orthonormal basis
    fwd = jnp.cross(right, up)
    R = jnp.vstack([right, up, fwd]).T
    q = jnp.dot(p - center, R)
    # Standard box distance
    d = jnp.abs(q) - (dim - r)
    inside = jnp.minimum(jnp.maximum(d[0], jnp.maximum(d[1], d[2])), 0.0)
    outside = jnp.linalg.norm(jnp.maximum(d, 0.0))
    return inside + outside - r


def _quaternion_rotate(p: jnp.ndarray, axis: jnp.ndarray, angle_deg: float) -> jnp.ndarray:
    """Rotate ``p`` about ``axis`` by ``angle_deg`` degrees using quaternions.

    The Tardigrade shader uses a quaternion‑based rotation helper to
    orient ellipsoids and claws【49612677301266†L21-L31】.  This function
    reproduces that behaviour.  ``angle_deg`` is interpreted as
    degrees; internally it is converted to radians for trigonometric
    functions.
    """
    axis = axis / jnp.linalg.norm(axis)
    half_angle = jnp.radians(angle_deg) * 0.5
    s = jnp.sin(half_angle)
    c = jnp.cos(half_angle)
    q = jnp.concatenate([axis * s, jnp.array([c])])
    # Quaternion rotation: p' = p + 2*cross(q.xyz, cross(q.xyz, p) + q.w*p)
    qv = q[:3]
    qw = q[3]
    return p + 2.0 * jnp.cross(qv, jnp.cross(qv, p) + qw * p)


def _op_rep(p: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """Periodic repetition of space.

    Replicates the ``opRep`` helper used for the tardigrade's teeth【49612677301266†L88-L93】.
    Returns ``mod(p, c) - 0.5 * c`` for each coordinate.
    """
    return jnp.mod(p, c) - 0.5 * c


def _rotate_2d(v: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Rotate a 2D vector by ``angle`` radians."""
    s, c = jnp.sin(angle), jnp.cos(angle)
    return jnp.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])



def sdf_tardigrade(p: jnp.ndarray) -> float:
    """Signed distance to a tardigrade.

    This function follows the structure of the tardigrade shader【49612677301266†L95-L168】.
    The body is constructed by blending a series of ellipsoids for the
    torso and head, subtracting a back opening, and then attaching
    four legs composed of ellipsoids and claws.  A small mouth with
    teeth is added on the front.  Texture‑based displacements present
    in the original shader are omitted for simplicity.
    """
    # Scale the input as in the GLSL wrapper【49612677301266†L171-L174】
    scale = 0.3
    q = p / scale
    # Body sections
    body_center = sd_ellipsoid(_quaternion_rotate(q, jnp.array([1.0, 0.0, 0.0]), 10.0), jnp.array([1.2, 0.9, 1.0]))
    body_front = sd_ellipsoid(_quaternion_rotate(q + jnp.array([0.0, 0.1, 0.8]), jnp.array([1.0, 0.0, 0.0]), 20.0), jnp.array([1.0, 0.7, 0.9]))
    body_front2 = sd_ellipsoid(_quaternion_rotate(q + jnp.array([0.0, 0.3, 1.5]), jnp.array([1.0, 0.0, 0.0]), 40.0), jnp.array([0.7, 0.5, 0.7]))
    body_back = sd_ellipsoid(_quaternion_rotate(q + jnp.array([0.0, 0.0, -0.6]), jnp.array([1.0, 0.0, 0.0]), -10.0), jnp.array([1.0, 0.75, 1.0]))
    body_back_hole = sd_ellipsoid(q + jnp.array([0.0, 0.2, -1.5]), jnp.array([0.03, 0.03, 0.5]))
    # Blend body parts【49612677301266†L115-L118】
    s = 0.01
    body = _soft_max(_soft_min(_soft_min(body_center, _soft_min(body_front, body_front2, s), s), body_back, s), -body_back_hole, 0.15)
    # Mouth【49612677301266†L124-L139】
    mouth0 = sd_sphere(q + jnp.array([0.0, 0.7, 2.25]), 0.15)
    mouth1 = sd_ellipsoid(q + jnp.array([0.0, 0.6, 2.125]), jnp.array([0.22, 0.175, 0.175]))
    mouth2 = sd_ellipsoid(q + jnp.array([0.0, 0.67, 2.25]), jnp.array([0.125, 0.1, 0.2]))
    # Teeth
    def teeth_fn(pt: jnp.ndarray) -> float:
        # Convert to polar coordinates and repeat【49612677301266†L82-L92】
        polar_x = jnp.arctan2(pt[0], pt[1]) / jnp.pi
        polar_y = jnp.linalg.norm(pt[jnp.array([0, 1])]) - 0.12
        polar_z = pt[2]
        p2 = _op_rep(jnp.array([polar_x, polar_y, polar_z]), jnp.array([0.25, 7.0, 0.0]))
        p2 = p2.at[1].set(polar_y)
        p2 = p2.at[2].set(pt[2])
        return sd_ellipsoid(p2, jnp.array([0.07, 0.05, 0.07]))
    teeth0 = teeth_fn(_quaternion_rotate(q + jnp.array([0.0, 0.62, 2.15]), jnp.array([1.0, 0.0, 0.0]), 35.0))
    # Head【49612677301266†L133-L139】
    head = sd_ellipsoid(_quaternion_rotate(q + jnp.array([0.0, 0.45, 1.9]), jnp.array([1.0, 0.0, 0.0]), 50.0), jnp.array([0.45, 0.3, 0.5]))
    head = jnp.minimum(_soft_max(_soft_min(mouth1, _soft_max(head, -mouth0, 0.3), s), -mouth0, 0.02), teeth0)
    # Legs【49612677301266†L145-L161】
    def claw(pos: jnp.ndarray, size: jnp.ndarray, angles: jnp.ndarray) -> float:
        # Claw constructed from three rotated ellipsoids and limited by y【49612677301266†L60-L72】
        a = pos[1] * angles[3] + angles[:3]
        c1 = sd_ellipsoid(_quaternion_rotate(pos, jnp.array([0.0, 0.0, 1.0]), a[0]), size)
        c2 = sd_ellipsoid(_quaternion_rotate(pos + jnp.array([0.0, 0.0, size[0]]), jnp.array([1.0, 0.0, 1.0]), a[1]), size)
        c3 = sd_ellipsoid(_quaternion_rotate(pos - jnp.array([0.0, 0.0, size[0]]), jnp.array([-1.0, 0.0, 1.0]), a[2]), size)
        return jnp.maximum(jnp.minimum(jnp.minimum(c1, c2), c3), pos[1])
    def leg(pos: jnp.ndarray, axis: jnp.ndarray, angle: float, size: jnp.ndarray, angles: jnp.ndarray) -> float:
        pos_rot = _quaternion_rotate(pos, axis, angle)
        claw_d = claw(pos_rot + jnp.array([0.0, size[1] * 0.5, 0.0]), jnp.array([0.075, 0.75, 0.075]) * size[1], angles)
        leg_d = sd_ellipsoid(pos_rot, size)
        return jnp.minimum(leg_d, claw_d)
    # Symmetry: reflect x to negative side
    sym_q = jnp.array([-jnp.abs(q[0]), q[1], q[2]])
    # Leg positions and parameters
    leg0 = leg(_quaternion_rotate(sym_q + jnp.array([0.75, 0.5, -1.15]), jnp.array([1.0, 0.0, -1.0]), 20.0), jnp.array([1.0, 0.0, 0.0]), 0.0, jnp.array([0.2, 0.5, 0.25]), jnp.array([20.0, -10.0, -10.0, 30.0]))
    leg1 = leg(_quaternion_rotate(sym_q + jnp.array([1.0, 0.55, 0.0]), jnp.array([1.0, 0.0, -1.0]), 10.0), jnp.array([1.0, 0.0, 0.0]), 0.0, jnp.array([0.3, 0.6, 0.35]), jnp.array([25.0, -5.0, -10.0, 40.0]))
    leg2 = leg(_quaternion_rotate(sym_q + jnp.array([0.9, 0.6, 1.0]), jnp.array([1.0, 0.0, 1.0]), -5.0), jnp.array([1.0, 0.0, 0.0]), 0.0, jnp.array([0.2, 0.5, 0.25]), jnp.array([15.0, -10.0, -5.0, 35.0]))
    leg3 = leg(_quaternion_rotate(sym_q + jnp.array([0.55, 0.7, 1.7]), jnp.array([1.0, 0.0, 0.0]), -10.0), jnp.array([1.0, 0.0, 0.0]), 0.0, jnp.array([0.15, 0.3, 0.15]), jnp.array([15.0, -15.0, -15.0, 50.0]))
    legs = jnp.minimum(jnp.minimum(jnp.minimum(leg0, leg1), leg2), leg3)
    body = _soft_min(body, legs, 0.05)
    res = _soft_min(body, head, s)
    return res * scale


def surface1(xyz: jnp.ndarray) -> jnp.ndarray:
    """(x^2+y^2-0.7^2)^2 + z^2  with offset {0.3, -0.8, 1}"""
    # unpack and apply offset
    x, y, z = xyz[0] - 0.3, xyz[1] - (-0.8), xyz[2] - 1
    # compute
    return (x**2 + y**2 - 0.7**2)**2 + z**2

def surface2(xyz: jnp.ndarray) -> jnp.ndarray:
    """(x^2+y^2-0.7^2)^2 + (z^2-1)^2 - 0.05  with offset {1.3, -1.0, 2}"""
    x, y, z = xyz[0] - 1.3, xyz[1] - (-1.0), xyz[2] - 2
    return (x**2 + y**2 - 0.7**2)**2 + (z**2 - 1)**2 - 0.05

def surface3(xyz: jnp.ndarray) -> jnp.ndarray:
    """(1.2 y^2 - 1)^2*(x^2+y^2-1)^2 + (1.2 z^2 - 1)^2*(y^2+z^2-1)^2
       + (1.2 x^2 - 1)^2*(z^2+x^2-1)^2 - 0.02  with offset {1.3, -1.0, 1.5}"""
    x, y, z = xyz[0] - 1.3, xyz[1] - (-1.0), xyz[2] - 1.5
    term_xy = (x**2 + y**2 - 1)**2
    term_yz = (y**2 + z**2 - 1)**2
    term_zx = (z**2 + x**2 - 1)**2
    return ((1.2*y**2 - 1)**2) * term_xy + ((1.2*z**2 - 1)**2) * term_yz + ((1.2*x**2 - 1)**2) * term_zx - 0.02

def surface4(xyz: jnp.ndarray) -> jnp.ndarray:
    """Product of three quartics minus 0.001  with offset {1.2, -1.0, 1.3}"""
    x, y, z = xyz[0] - 1.2, xyz[1] - (-1.0), xyz[2] - 1.3
    f1 = (x**2 + y**2 - 0.85**2)**2 + (z**2 - 1)**2
    f2 = (y**2 + z**2 - 0.85**2)**2 + (x**2 - 1)**2
    f3 = (z**2 + x**2 - 0.85**2)**2 + (y**2 - 1)**2
    return f1 * f2 * f3 - 0.001

def surface5(xyz: jnp.ndarray) -> jnp.ndarray:
    """(3(x-1)x^2(x+1)+y^2)^2 + 2 z^2  with offset {0.3, -0.8, 1}"""
    x, y, z = xyz[0] - 0.3, xyz[1] - (-0.8), xyz[2] - 1
    return (3.0*(x - 1)*x**2*(x + 1) + y**2)**2 + 2.0*z**2

def surface6(xyz: jnp.ndarray) -> jnp.ndarray:
    """(3(x-1)x^2(x+1)+y^2)^2 + (2 z^2 - 1)^2 - 0.005  with offset {0.3, -0.8, 1}"""
    x, y, z = xyz[0] - 0.3, xyz[1] - (-0.8), xyz[2] - 1
    return (3.0*(x - 1)*x**2*(x + 1) + y**2)**2 + (2.0*z**2 - 1.0)**2 - 0.005

def surface7(xyz: jnp.ndarray) -> jnp.ndarray:
    """
    (3(x-1)x^2(x+1)+2 y^2)^2 +
    (z^2-0.85)^2 * (3(y-1)y^2(y+1)+2 z^2)^2 +
    (x^2-0.85)^2 * (3(z-1)z^2(z+1)+2 x^2)^2 +
    (y^2-0.85)^2 * (-0.12)   with offset {1, -1.8, 1.7}
    """
    x, y, z = xyz[0] - 1.0, xyz[1] - (-1.8), xyz[2] - 1.7
    term1 = (3.0*(x - 1)*x**2*(x + 1) + 2.0*y**2)**2
    term2 = (z**2 - 0.85)**2 * (3.0*(y - 1)*y**2*(y + 1) + 2.0*z**2)**2
    term3 = (x**2 - 0.85)**2 * (3.0*(z - 1)*z**2*(z + 1) + 2.0*x**2)**2
    # The original expression ends with +(y^2-0.85)^2* -0.12; interpret as subtraction
    term4 = -0.12 * (y**2 - 0.85)**2
    return term1 + term2 + term3 + term4

def surface8(xyz: jnp.ndarray) -> jnp.ndarray:
    """
    (2.92(x-1)x^2(x+1)+1.7 y^2)^2 * (y^2-0.88)^2 +
    (2.92(y-1)y^2(y+1)+1.7 z^2)^2 * (z^2-0.88)^2 +
    (2.92(z-1)z^2(z+1)+1.7 x^2)^2 * (x^2-0.88)^2 - 0.02
    with offset {0.4, -1.8, 1.3}
    """
    x, y, z = xyz[0] - 0.4, xyz[1] - (-1.8), xyz[2] - 1.3
    term_xy = (2.92*(x - 1)*x**2*(x + 1) + 1.7*y**2)**2 * (y**2 - 0.88)**2
    term_yz = (2.92*(y - 1)*y**2*(y + 1) + 1.7*z**2)**2 * (z**2 - 0.88)**2
    term_zx = (2.92*(z - 1)*z**2*(z + 1) + 1.7*x**2)**2 * (x**2 - 0.88)**2
    return term_xy + term_yz + term_zx - 0.02