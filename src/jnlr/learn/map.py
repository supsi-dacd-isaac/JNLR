from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go

from jnlr.utils.function_utils import infer_io_shapes


Params = Any


class ResidualMLP(nn.Module):
    """Residual MLP used for the learned projection map."""

    hidden_sizes: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = z
        for hidden_size in self.hidden_sizes:
            h = nn.Dense(hidden_size)(h)
            h = nn.silu(h)
        h = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        return z + h


class PrimalDualMLP(nn.Module):
    """MLP head that predicts primal residuals and Lagrange multipliers."""

    hidden_sizes: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = z
        for hidden_size in self.hidden_sizes:
            h = nn.Dense(hidden_size)(h)
            h = nn.silu(h)
        return nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)


def _constraint_vector(
    g: Callable[[jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.ravel(jnp.atleast_1d(g(z)))


def compute_V(
    g: Callable[[jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
    lambda_reg: float = 1e-3,
) -> jnp.ndarray:
    r"""
    Gauss-Newton projection field

    $$V_g(z) = -J_g(z)^T (J_g(z)J_g(z)^T + \lambda I)^{-1} g(z).$$

    This is a Newton-like step toward the manifold ``{g = 0}``: scale invariant
    and convergent in a handful of iterations. ``g`` may return either a scalar
    or a vector-valued constraint.
    """
    z = jnp.asarray(z)

    def g_vec(x):
        return jnp.atleast_1d(g(x))

    g_z = g_vec(z)
    jac_g = jax.jacfwd(g_vec)(z)
    gram = jac_g @ jac_g.T
    regularizer = jnp.asarray(lambda_reg, dtype=z.dtype) * jnp.eye(
        gram.shape[0], dtype=z.dtype
    )
    return -(jac_g.T @ jnp.linalg.solve(gram + regularizer, g_z))


def gradient_field(
    g: Callable[[jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
) -> jnp.ndarray:
    r"""
    Cheap descent field $-J_g(z)^T g(z) = -\nabla(\tfrac12 ||g(z)||^2)$.

    Computed with a single vector-Jacobian product, so it needs neither an
    explicit Jacobian nor a linear solve, unlike :func:`compute_V`. It shares
    the same fixed point ($g = 0$) and normal direction as the Gauss-Newton
    field, but is plain gradient descent (no Newton rescaling), so it converges
    more slowly and is more sensitive to the step size.
    """
    z = jnp.asarray(z)
    g_z, vjp = jax.vjp(lambda x: jnp.atleast_1d(g(x)), z)
    return -vjp(g_z)[0]


def project_to_manifold(
    g: Callable[[jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
    n_steps: int = 8,
    eta: float = 1.0,
    field: str = "gauss_newton",
    lambda_reg: float = 1e-6,
) -> jnp.ndarray:
    r"""
    Project a single point onto the manifold ``{g = 0}`` by iterating a field.

    With ``field="gauss_newton"`` it takes Newton-like steps (:func:`compute_V`),
    which are scale invariant and converge in a handful of iterations. With
    ``field="gradient"`` it uses the lighter :func:`gradient_field`. The loop is
    unrolled (``n_steps`` is static), so the whole projection is differentiable
    and can be composed with a network as a model head.
    """
    if field not in {"gauss_newton", "gradient"}:
        raise ValueError('field must be either "gauss_newton" or "gradient"')

    y = jnp.asarray(z)
    for _ in range(int(n_steps)):
        if field == "gauss_newton":
            step = compute_V(g, y, lambda_reg=lambda_reg)
        else:
            step = gradient_field(g, y)
        y = y + eta * step
    return y


def kkt_residual(
    g: Callable[[jnp.ndarray], jnp.ndarray],
    z0: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    r"""
    KKT residual for the Euclidean projection of ``z0`` onto ``{g = 0}``.

    ``y`` is the concatenated primal-dual vector ``[z, lambda]``. The residual is

    $$F(z, \lambda; z_0) = [z - z_0 + J_g(z)^T\lambda,\; g(z)].$$
    """
    z0 = jnp.asarray(z0)
    y = jnp.asarray(y)
    z = y[: z0.shape[0]]
    lam = y[z0.shape[0] :]

    def g_vec(x):
        return _constraint_vector(g, x)

    g_z = g_vec(z)
    jac_g = jax.jacfwd(g_vec)(z)
    stationarity = z - z0 + jac_g.T @ lam
    return jnp.concatenate([stationarity, g_z])


def kkt_fixed_point_step(
    g: Callable[[jnp.ndarray], jnp.ndarray],
    z0: jnp.ndarray,
    y: jnp.ndarray,
    damping: float = 1.0,
    lambda_reg: float = 1e-6,
    matrix: str = "exact",
) -> jnp.ndarray:
    r"""
    One damped Newton fixed-point step for the projection KKT system.

    With ``matrix="exact"``, this computes ``T(y; z0) = y - damping *
    K(y; z0)^{-1} F(y; z0)``, where ``K`` is the Jacobian of
    :func:`kkt_residual` with respect to ``y`` and includes the
    ``sum_i lambda_i Hessian(g_i)`` block.

    With ``matrix="sqp"``, it uses the cheaper Gauss-Newton/SQP approximation
    ``[[I, J_g(z)^T], [J_g(z), 0]]``. A small diagonal ``lambda_reg`` is added to
    either matrix for numerical robustness.
    """
    if matrix not in {"exact", "sqp"}:
        raise ValueError('matrix must be either "exact" or "sqp"')

    z0 = jnp.asarray(z0)
    y = jnp.asarray(y)
    residual = kkt_residual(g, z0, y)

    if matrix == "exact":
        jacobian = jax.jacfwd(lambda yy: kkt_residual(g, z0, yy))(y)
    else:
        z = y[: z0.shape[0]]

        def g_vec(x):
            return _constraint_vector(g, x)

        jac_g = jax.jacfwd(g_vec)(z)
        eye = jnp.eye(z0.shape[0], dtype=y.dtype)
        zero = jnp.zeros((jac_g.shape[0], jac_g.shape[0]), dtype=y.dtype)
        top = jnp.concatenate([eye, jac_g.T], axis=1)
        bottom = jnp.concatenate([jac_g, zero], axis=1)
        jacobian = jnp.concatenate([top, bottom], axis=0)

    regularizer = jnp.asarray(lambda_reg, dtype=y.dtype) * jnp.eye(
        jacobian.shape[0], dtype=y.dtype
    )
    jacobian = jacobian + regularizer
    direction = -jnp.linalg.solve(jacobian, residual)
    return y + damping * direction


class Projector:
    r"""
    Learn an amortised map that projects points onto the manifold ``{g = 0}``.

    Training combines a differentiable projection head (a fixed number of
    Gauss-Newton or gradient steps) with a stop-gradient consistency term,

    $$E_{z_0}\big[
        ||\operatorname{proj}_g(f_\theta(z_0)) - z_0||^2
        + w\,||f_\theta(z_0) - \operatorname{sg}(\operatorname{proj}_g(f_\theta(z_0)))||^2
    \big].$$

    The first term uses the head to supply a minimal-distance signal; the second
    pulls the bare network onto its own projection. At the optimum
    $f_\theta(z_0)$ equals its projection (the nearest manifold point), so
    :meth:`predict` is a single forward pass at inference (no ``g`` evaluation).
    Pass ``project=True`` to :meth:`predict` to additionally run the numerical
    projection head, warm-started from the network output, for exact constraint
    satisfaction.
    """

    def __init__(
        self,
        g: Callable[[jnp.ndarray], jnp.ndarray],
        hidden_sizes: Sequence[int] = (32, 32),
        key: int | jnp.ndarray = 0,
    ):
        self.g = g
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

        input_shape, _ = infer_io_shapes(g)
        self.input_dim = int(input_shape[0])

        key = self._as_prng_key(key)
        self.key, init_key = jax.random.split(key)
        self.model = ResidualMLP(
            hidden_sizes=self.hidden_sizes,
            output_dim=self.input_dim,
        )
        self.params = self._init_params(init_key)
        self.loss_history = jnp.array([], dtype=jnp.float32)
        self._projection_kwargs: dict[str, Any] = dict(
            n_steps=8, eta=1.0, field="gauss_newton", lambda_reg=1e-6
        )

    @staticmethod
    def _as_prng_key(key: int | jnp.ndarray) -> jnp.ndarray:
        if isinstance(key, (int, np.integer)):
            return jax.random.PRNGKey(int(key))
        return jnp.asarray(key)

    def _init_params(self, key: jnp.ndarray) -> Params:
        dummy = jnp.zeros((self.input_dim,), dtype=jnp.float32)
        return self.model.init(key, dummy)["params"]

    def _apply_params(self, params: Params, z: jnp.ndarray) -> jnp.ndarray:
        return self.model.apply({"params": params}, z)

    def _loss(
        self,
        params: Params,
        z0: jnp.ndarray,
        lambda_reg: float,
        n_proj_steps: int,
        proj_eta: float,
        proj_field: str,
        consistency_weight: float,
    ) -> jnp.ndarray:
        def loss_one(z):
            f_z = self._apply_params(params, z)
            p = project_to_manifold(
                self.g,
                f_z,
                n_steps=n_proj_steps,
                eta=proj_eta,
                field=proj_field,
                lambda_reg=lambda_reg,
            )
            # Distance term: the projection head provides the minimal-distance
            # signal (differentiated through, so the network learns where on the
            # manifold to land).
            distance_loss = jnp.sum((p - z) ** 2)
            # Consistency term: pull the raw output onto its own projection so
            # f_theta alone lands on the manifold (target stopped, no Jacobian
            # backprop here).
            consistency_loss = jnp.sum((f_z - jax.lax.stop_gradient(p)) ** 2)
            return distance_loss + consistency_weight * consistency_loss

        return jnp.mean(jax.vmap(loss_one)(z0))

    def train(
        self,
        z0: jnp.ndarray,
        n_steps: int = 200,
        batch_size: int = 256,
        learning_rate: float = 2e-3,
        lambda_reg: float = 1e-6,
        n_proj_steps: int = 8,
        proj_eta: float = 1.0,
        proj_field: str = "gauss_newton",
        consistency_weight: float = 1.0,
    ) -> jnp.ndarray:
        """Train the projection map on samples ``z0`` and return the loss history.

        A differentiable projection head (``n_proj_steps`` steps of ``proj_field``,
        one of ``"gauss_newton"`` or ``"gradient"``, scaled by ``proj_eta``) supplies
        the minimal-distance signal ``||proj_g(f_theta(z0)) - z0||^2``, while a
        stop-gradient consistency term ``consistency_weight * ||f_theta(z0) -
        proj_g(f_theta(z0))||^2`` trains the bare network to land on the manifold by
        itself, so :meth:`predict` needs only one forward pass at inference.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if proj_field not in {"gauss_newton", "gradient"}:
            raise ValueError('proj_field must be either "gauss_newton" or "gradient"')
        if n_proj_steps <= 0:
            raise ValueError("n_proj_steps must be positive")
        if consistency_weight < 0:
            raise ValueError("consistency_weight must be non-negative")

        z_train = jnp.asarray(z0, dtype=jnp.float32)
        if z_train.ndim == 1:
            z_train = z_train[None, :]
        if z_train.ndim != 2 or z_train.shape[1] != self.input_dim:
            raise ValueError(
                f"z0 must have shape (n_samples, {self.input_dim}) or ({self.input_dim},)"
            )

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.params)
        n_samples = z_train.shape[0]

        @jax.jit
        def step(params, opt_state, key, samples):
            idx = jax.random.randint(key, (batch_size,), 0, n_samples)
            batch = samples[idx]
            loss, grads = jax.value_and_grad(
                lambda p: self._loss(
                    p,
                    batch,
                    lambda_reg=lambda_reg,
                    n_proj_steps=n_proj_steps,
                    proj_eta=proj_eta,
                    proj_field=proj_field,
                    consistency_weight=consistency_weight,
                )
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        params = self.params
        key = self.key
        losses = []
        for _ in range(n_steps):
            key, subkey = jax.random.split(key)
            params, opt_state, loss = step(params, opt_state, subkey, z_train)
            losses.append(loss)

        self.params = params
        self.key = key
        self.loss_history = jnp.stack(losses)
        self._projection_kwargs = dict(
            n_steps=n_proj_steps,
            eta=proj_eta,
            field=proj_field,
            lambda_reg=lambda_reg,
        )
        return self.loss_history

    def predict(self, z: jnp.ndarray, project: bool = False) -> jnp.ndarray:
        """Apply the learned map to one point or a batch of points.

        By default this is a single forward pass of the amortised network, which
        is trained to land on ``{g = 0}`` by itself (no ``g`` evaluation at
        inference). Pass ``project=True`` to additionally run the numerical
        projection head, warm-started from the network output, when exact
        constraint satisfaction is required.
        """
        z = jnp.asarray(z, dtype=jnp.float32)
        if z.ndim == 1 and z.shape[0] != self.input_dim:
            raise ValueError(f"z must have shape ({self.input_dim},)")
        if z.ndim > 1 and z.shape[-1] != self.input_dim:
            raise ValueError(f"z must have last dimension {self.input_dim}")

        y = self._apply_params(self.params, z)
        if project:
            kwargs = self._projection_kwargs
            if y.ndim == 1:
                return project_to_manifold(self.g, y, **kwargs)
            return jax.vmap(lambda yi: project_to_manifold(self.g, yi, **kwargs))(y)
        return y

    def V(self, z: jnp.ndarray, lambda_reg: float = 1e-3) -> jnp.ndarray:
        """Compute ``V_g`` at one point or a batch of points."""
        z = jnp.asarray(z, dtype=jnp.float32)
        if z.ndim == 1:
            return compute_V(self.g, z, lambda_reg=lambda_reg)
        return jax.vmap(lambda zi: compute_V(self.g, zi, lambda_reg=lambda_reg))(z)

    def plot_vector_field(
        self,
        points: jnp.ndarray,
        curve: jnp.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        show: bool = True,
        vector_scale: float = 1.0,
        arrow_scale: float = 0.3,
        width: int = 700,
        height: int = 600,
        title: str = "Learned Projection",
    ) -> go.Figure:
        """Return a 2D Plotly vector-field figure for the learned projection."""
        if self.input_dim != 2:
            raise ValueError("plot_vector_field is only available for 2D maps")

        points_np = np.asarray(points, dtype=float)
        if points_np.ndim == 1:
            points_np = points_np[None, :]
        if points_np.ndim != 2 or points_np.shape[1] != 2:
            raise ValueError("points must have shape (n_points, 2) or (2,)")

        projected_np = np.asarray(self.predict(points_np), dtype=float)
        vectors_np = projected_np - points_np

        fig = go.Figure()
        if curve is not None:
            curve_np = self._curve_points(curve, points_np)
            fig.add_trace(
                go.Scatter(
                    x=curve_np[:, 0],
                    y=curve_np[:, 1],
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="constraint",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="royalblue", width=1.5),
                name="vectors",
            )
        )
        for start, vector in zip(points_np, vectors_np, strict=True):
            dx, dy = vector_scale * vector
            if not np.isfinite(dx) or not np.isfinite(dy):
                continue
            if dx == 0 and dy == 0:
                continue
            x0, y0 = start
            fig.add_annotation(
                x=x0 + dx,
                y=y0 + dy,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=arrow_scale,
                arrowwidth=1.2,
                arrowcolor="royalblue",
            )

        fig.add_trace(
            go.Scatter(
                x=points_np[:, 0],
                y=points_np[:, 1],
                mode="markers",
                marker=dict(size=5, color="rgba(120,120,120,0.8)"),
                name="source",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=projected_np[:, 0],
                y=projected_np[:, 1],
                mode="markers",
                marker=dict(size=5, color="crimson"),
                name="mapped",
            )
        )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            xaxis_title="x",
            yaxis_title="y",
            yaxis_scaleanchor="x",
            template="plotly_white",
            margin=dict(l=50, r=30, b=50, t=60),
        )

        if show:
            fig.show()
        return fig

    @staticmethod
    def _curve_points(
        curve: jnp.ndarray | Callable[[np.ndarray], np.ndarray],
        points: np.ndarray,
    ) -> np.ndarray:
        if callable(curve):
            x_min = float(np.nanmin(points[:, 0]))
            x_max = float(np.nanmax(points[:, 0]))
            pad = 0.05 * max(x_max - x_min, 1.0)
            xs = np.linspace(x_min - pad, x_max + pad, 200)
            ys = np.asarray(curve(xs), dtype=float)
            return np.column_stack([xs, ys])

        curve_np = np.asarray(curve, dtype=float)
        if curve_np.ndim != 2 or curve_np.shape[1] != 2:
            raise ValueError("curve must have shape (n_points, 2) or be callable")
        return curve_np


class KKTProjector(Projector):
    r"""
    Learn an amortised primal-dual projection map with a KKT fixed-point loss.

    The network predicts a gated residual update for the primal variable and a
    gated multiplier,

    $$z_\theta(z_0) = z_0 + \rho(||g(z_0)||) r_\theta(z_0),\quad
      \lambda_\theta(z_0) = \rho(||g(z_0)||) \ell_\theta(z_0).$$

    The loss trains ``[z_theta, lambda_theta]`` to match one stop-gradient
    Newton improvement step for the projection KKT equations. ``predict`` keeps
    the same user-facing shape as :class:`Projector` and returns only the primal
    projected point; use :meth:`predict_primal_dual` to inspect multipliers.
    """

    def __init__(
        self,
        g: Callable[[jnp.ndarray], jnp.ndarray],
        hidden_sizes: Sequence[int] = (32, 32),
        key: int | jnp.ndarray = 0,
        gate_tau: float = 1e-3,
    ):
        self.g = g
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if gate_tau <= 0:
            raise ValueError("gate_tau must be positive")

        input_shape, output_shape = infer_io_shapes(g)
        self.input_dim = int(input_shape[0])
        self.constraint_dim = int(np.prod(output_shape))
        self.gate_tau = float(gate_tau)

        key = self._as_prng_key(key)
        self.key, init_key = jax.random.split(key)
        self.model = PrimalDualMLP(
            hidden_sizes=self.hidden_sizes,
            output_dim=self.input_dim + self.constraint_dim,
        )
        self.params = self._init_params(init_key)
        self.loss_history = jnp.array([], dtype=jnp.float32)
        self._projection_kwargs: dict[str, Any] = dict(
            n_steps=8, eta=1.0, field="gauss_newton", lambda_reg=1e-6
        )
        self._kkt_kwargs: dict[str, Any] = dict(damping=1.0, lambda_reg=1e-6)

    def _init_params(self, key: jnp.ndarray) -> Params:
        dummy = jnp.zeros((self.input_dim,), dtype=jnp.float32)
        return self.model.init(key, dummy)["params"]

    def _constraint_vector(self, z: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(_constraint_vector(self.g, z), (self.constraint_dim,))

    def _gate(self, z: jnp.ndarray) -> jnp.ndarray:
        violation = jnp.linalg.norm(self._constraint_vector(z))
        tau = jnp.asarray(self.gate_tau, dtype=z.dtype)
        return violation / (violation + tau)

    def _apply_primal_dual(self, params: Params, z: jnp.ndarray) -> jnp.ndarray:
        raw = self.model.apply({"params": params}, z)
        r = raw[: self.input_dim]
        lam = raw[self.input_dim :]
        gate = self._gate(z)
        return jnp.concatenate([z + gate * r, gate * lam])

    def _apply_params(self, params: Params, z: jnp.ndarray) -> jnp.ndarray:
        return self._apply_primal_dual(params, z)[: self.input_dim]

    def _loss(
        self,
        params: Params,
        z0: jnp.ndarray,
        damping: float,
        lambda_reg: float,
        multiplier_weight: float,
        matrix: str,
    ) -> jnp.ndarray:
        def loss_one(z):
            y = self._apply_primal_dual(params, z)
            target = kkt_fixed_point_step(
                self.g,
                z,
                y,
                damping=damping,
                lambda_reg=lambda_reg,
                matrix=matrix,
            )
            diff = y - jax.lax.stop_gradient(target)
            primal_loss = jnp.sum(diff[: self.input_dim] ** 2)
            multiplier_loss = jnp.sum(diff[self.input_dim :] ** 2)
            return primal_loss + multiplier_weight * multiplier_loss

        return jnp.mean(jax.vmap(loss_one)(z0))

    def train(
        self,
        z0: jnp.ndarray,
        n_steps: int = 200,
        batch_size: int = 256,
        learning_rate: float = 2e-3,
        damping: float = 1.0,
        lambda_reg: float = 1e-6,
        multiplier_weight: float = 1.0,
        matrix: str = "exact",
    ) -> jnp.ndarray:
        """Train the KKT fixed-point map on samples ``z0``."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if damping <= 0 or damping > 1:
            raise ValueError("damping must satisfy 0 < damping <= 1")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative")
        if multiplier_weight < 0:
            raise ValueError("multiplier_weight must be non-negative")
        if matrix not in {"exact", "sqp"}:
            raise ValueError('matrix must be either "exact" or "sqp"')

        z_train = jnp.asarray(z0, dtype=jnp.float32)
        if z_train.ndim == 1:
            z_train = z_train[None, :]
        if z_train.ndim != 2 or z_train.shape[1] != self.input_dim:
            raise ValueError(
                f"z0 must have shape (n_samples, {self.input_dim}) or ({self.input_dim},)"
            )

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.params)
        n_samples = z_train.shape[0]

        @jax.jit
        def step(params, opt_state, key, samples):
            idx = jax.random.randint(key, (batch_size,), 0, n_samples)
            batch = samples[idx]
            loss, grads = jax.value_and_grad(
                lambda p: self._loss(
                    p,
                    batch,
                    damping=damping,
                    lambda_reg=lambda_reg,
                    multiplier_weight=multiplier_weight,
                    matrix=matrix,
                )
            )(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        params = self.params
        key = self.key
        losses = []
        for _ in range(n_steps):
            key, subkey = jax.random.split(key)
            params, opt_state, loss = step(params, opt_state, subkey, z_train)
            losses.append(loss)

        self.params = params
        self.key = key
        self.loss_history = jnp.stack(losses)
        self._kkt_kwargs = dict(damping=damping, lambda_reg=lambda_reg, matrix=matrix)
        return self.loss_history

    def predict_primal_dual(self, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(z_theta, lambda_theta)`` for one point or a batch."""
        z = jnp.asarray(z, dtype=jnp.float32)
        if z.ndim == 1 and z.shape[0] != self.input_dim:
            raise ValueError(f"z must have shape ({self.input_dim},)")
        if z.ndim > 1 and z.shape[-1] != self.input_dim:
            raise ValueError(f"z must have last dimension {self.input_dim}")

        if z.ndim == 1:
            y = self._apply_primal_dual(self.params, z)
            return y[: self.input_dim], y[self.input_dim :]
        leading_shape = z.shape[:-1]
        flat_z = jnp.reshape(z, (-1, self.input_dim))
        y = jax.vmap(lambda zi: self._apply_primal_dual(self.params, zi))(flat_z)
        primal = jnp.reshape(
            y[:, : self.input_dim], leading_shape + (self.input_dim,)
        )
        multipliers = jnp.reshape(
            y[:, self.input_dim :], leading_shape + (self.constraint_dim,)
        )
        return primal, multipliers

    def predict(self, z: jnp.ndarray, project: bool = False) -> jnp.ndarray:
        """Apply the learned KKT map and return only the primal variable."""
        primal, _ = self.predict_primal_dual(z)
        if project:
            kwargs = self._projection_kwargs
            if primal.ndim == 1:
                return project_to_manifold(self.g, primal, **kwargs)
            return jax.vmap(lambda zi: project_to_manifold(self.g, zi, **kwargs))(
                primal
            )
        return primal


def _demo_parabola(show: bool = True) -> go.Figure:
    def parabola_g(z):
        x, y = z
        return jnp.array([x**2 - y], dtype=z.dtype)

    xs = jnp.linspace(-1.5, 1.5, 25)
    ys = jnp.linspace(-0.5, 2.5, 25)
    points = jnp.stack(jnp.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

    projector = Projector(parabola_g, key=0)
    projector.train(points, n_steps=200)

    field_xs = jnp.linspace(-1.4, 1.4, 13)
    field_ys = jnp.linspace(-0.3, 2.3, 13)
    field_points = jnp.stack(
        jnp.meshgrid(field_xs, field_ys, indexing="xy"), axis=-1
    ).reshape(-1, 2)
    return projector.plot_vector_field(field_points, curve=lambda x: x**2, show=show)


if __name__ == "__main__":
    _demo_parabola(show=True)
