# Learned Projection Techniques

This note describes the three projection-learning techniques exercised in
`tests/test_learn_map.py` and implemented in `src/jnlr/learn/map.py`.

Let

$$
M = \{z \in \mathbb{R}^d : g(z) = 0\}, \qquad
g : \mathbb{R}^d \to \mathbb{R}^m.
$$

For an off-manifold point \(z_0 \in \mathbb{R}^d\), the Euclidean projection
problem is

$$
\Pi_M(z_0)
= \arg\min_z \frac{1}{2}\|z - z_0\|_2^2
\quad \text{s.t.} \quad g(z) = 0.
$$

The tests use the parabola

$$
g(x, y) = x^2 - y,
\qquad
M = \{(x,y) : y = x^2\}.
$$

The plots saved by the comparison test are:

- `tests/learn_map_projection_head.html`
- `tests/learn_map_kkt_exact_newton.html`
- `tests/learn_map_kkt_sqp_gauss_newton.html`

## 1. Projection-Head Amortized Map

Class: `Projector`

This is the original amortized projection strategy. The network is a residual
map

$$
f_\theta(z_0) = z_0 + r_\theta(z_0).
$$

It is initialized as the identity because the final dense layer is initialized
with zero weights and zero bias. During training, the raw network output
\(f_\theta(z_0)\) is passed through a differentiable numerical projection head.

The numerical projection head applies a fixed number of steps

$$
y_{k+1} = y_k + \eta V_g(y_k).
$$

For the default Gauss-Newton field,

$$
V_g(y)
= -J_g(y)^\top
\left(J_g(y)J_g(y)^\top + \lambda I\right)^{-1}
g(y).
$$

The implemented training loss is

$$
\mathcal{L}_{\text{proj-head}}(\theta)
=
\mathbb{E}_{z_0}
\left[
\left\|
\operatorname{proj}_g(f_\theta(z_0)) - z_0
\right\|_2^2
+
w
\left\|
f_\theta(z_0)
-
\operatorname{sg}
\left(\operatorname{proj}_g(f_\theta(z_0))\right)
\right\|_2^2
\right],
$$

where:

- \(\operatorname{proj}_g\) is the unrolled projection head.
- \(\operatorname{sg}(\cdot)\) means stop-gradient.
- \(w\) is `consistency_weight`.

The first term asks the projection head of the network output to be close to the
source point \(z_0\). The second term pulls the raw network output toward its own
numerically projected value.

At inference time:

$$
\hat{z} = f_\theta(z_0)
$$

is returned by default. This is a single neural-network forward pass. If
`project=True`, the same numerical projection head is run again, warm-started
from \(f_\theta(z_0)\), to improve exact constraint satisfaction.

In the comparison test this technique is plotted with title:

$$
\text{Projection Head}.
$$

## 2. KKT Fixed-Point Map With Exact Newton Matrix

Class: `KKTProjector` with `matrix="exact"`

The Euclidean projection problem has Lagrangian

$$
\mathcal{L}(z, \lambda; z_0)
=
\frac{1}{2}\|z - z_0\|_2^2
+
\lambda^\top g(z).
$$

The KKT equations are

$$
z - z_0 + J_g(z)^\top \lambda = 0,
\qquad
g(z) = 0.
$$

Define the primal-dual variable

$$
y =
\begin{bmatrix}
z \\
\lambda
\end{bmatrix}
\in \mathbb{R}^{d+m}.
$$

The KKT residual is

$$
F(y; z_0)
=
F(z,\lambda; z_0)
=
\begin{bmatrix}
z - z_0 + J_g(z)^\top \lambda \\
g(z)
\end{bmatrix}.
$$

The KKT projector predicts both the primal point and the Lagrange multiplier.
The architecture uses a gated residual form:

$$
z_\theta(z_0)
=
z_0 + \rho(\|g(z_0)\|_2) r_\theta(z_0),
$$

$$
\lambda_\theta(z_0)
=
\rho(\|g(z_0)\|_2) \ell_\theta(z_0),
$$

with

$$
\rho(t) = \frac{t}{t + \tau},
\qquad
\tau > 0.
$$

Therefore, if \(g(z_0)=0\), then \(\rho(\|g(z_0)\|_2)=0\), and

$$
z_\theta(z_0) = z_0,
\qquad
\lambda_\theta(z_0) = 0.
$$

So points already on the manifold are fixed by construction.

The predicted primal-dual output is

$$
\Phi_\theta(z_0)
=
\begin{bmatrix}
z_\theta(z_0) \\
\lambda_\theta(z_0)
\end{bmatrix}.
$$

The exact Newton fixed-point operator is

$$
T_{\text{exact}}(y; z_0)
=
y - \alpha K_{\text{exact}}(y; z_0)^{-1}F(y; z_0),
$$

where \(0 < \alpha \le 1\) is `damping` and

$$
K_{\text{exact}}(y; z_0)
=
\nabla_y F(y; z_0).
$$

Written in block form,

$$
K_{\text{exact}}(z,\lambda; z_0)
=
\begin{bmatrix}
I + \sum_{i=1}^m \lambda_i \nabla^2 g_i(z) & J_g(z)^\top \\
J_g(z) & 0
\end{bmatrix}.
$$

This is the Hessian version: the upper-left block contains
\(\sum_i \lambda_i \nabla^2 g_i(z)\). In the implementation, this matrix is
formed by differentiating `kkt_residual` with respect to the concatenated
primal-dual variable \(y\).

For numerical robustness, the implementation solves with

$$
K_{\text{exact}}(y; z_0) + \lambda_{\text{reg}} I.
$$

The stop-gradient fixed-point loss is

$$
\mathcal{L}_{\text{KKT-exact}}(\theta)
=
\mathbb{E}_{z_0}
\left[
\left\|
\Phi_\theta(z_0)
-
\operatorname{sg}
\left(
T_{\text{exact}}(\Phi_\theta(z_0); z_0)
\right)
\right\|_W^2
\right].
$$

In code, \(W\) is represented by separate primal and multiplier weights:

$$
\|u\|_W^2
=
\|u_z\|_2^2
+
w_\lambda \|u_\lambda\|_2^2,
$$

where \(w_\lambda\) is `multiplier_weight`.

The gradient does not flow through the Newton target. Numerically, the residual
inside the norm is the Newton correction, but the target is detached:

$$
\Phi_\theta(z_0)
-
\operatorname{sg}
\left(
\Phi_\theta(z_0) + \alpha \Delta y
\right),
\qquad
\Delta y = -K_{\text{exact}}^{-1}F.
$$

At inference time, `KKTProjector.predict(z0)` returns only
\(z_\theta(z_0)\). The multipliers are available through
`predict_primal_dual`.

In the comparison test this technique is plotted with title:

$$
\text{KKT Exact Newton}.
$$

## 3. KKT Fixed-Point Map With SQP/Gauss-Newton Matrix

Class: `KKTProjector` with `matrix="sqp"`

This technique uses the same primal-dual predictor

$$
\Phi_\theta(z_0)
=
\begin{bmatrix}
z_\theta(z_0) \\
\lambda_\theta(z_0)
\end{bmatrix},
$$

the same KKT residual

$$
F(z,\lambda; z_0)
=
\begin{bmatrix}
z - z_0 + J_g(z)^\top \lambda \\
g(z)
\end{bmatrix},
$$

and the same stop-gradient fixed-point training idea. The difference is the
linear system used to define the target.

Instead of the exact Newton matrix, it uses the SQP/Gauss-Newton approximation

$$
K_{\text{sqp}}(z)
=
\begin{bmatrix}
I & J_g(z)^\top \\
J_g(z) & 0
\end{bmatrix}.
$$

This drops the second-derivative term

$$
\sum_{i=1}^m \lambda_i \nabla^2 g_i(z).
$$

The fixed-point operator is

$$
T_{\text{sqp}}(y; z_0)
=
y - \alpha K_{\text{sqp}}(z)^{-1}F(y; z_0).
$$

The training loss is

$$
\mathcal{L}_{\text{KKT-sqp}}(\theta)
=
\mathbb{E}_{z_0}
\left[
\left\|
\Phi_\theta(z_0)
-
\operatorname{sg}
\left(
T_{\text{sqp}}(\Phi_\theta(z_0); z_0)
\right)
\right\|_W^2
\right].
$$

As with the exact KKT version, the implementation solves the regularized system

$$
\left(K_{\text{sqp}}(z) + \lambda_{\text{reg}}I\right)\Delta y
=
-F(y; z_0).
$$

This version is cheaper because it only needs \(J_g(z)\), not Hessians of
\(g\). It is also the closer analogue of a single SQP step for equality
constraints with identity Hessian approximation.

In the comparison test this technique is plotted with title:

$$
\text{KKT SQP Gauss-Newton}.
$$

## What The Test Compares

The main comparison test trains three objects on the same parabola grid:

1. `Projector(...)`, the projection-head amortized map.
2. `KKTProjector(...).train(..., matrix="exact")`.
3. `KKTProjector(...).train(..., matrix="sqp")`.

For each method, the learned vector field is plotted on a coarser grid. The
test also computes:

$$
\text{constraint residual}
=
\mathbb{E}_{z_0}
\left[
|g(\hat{z}(z_0))|
\right],
$$

and

$$
\text{projection error}
=
\mathbb{E}_{z_0}
\left[
\|\hat{z}(z_0) - \Pi_M(z_0)\|_2
\right],
$$

where \(\Pi_M(z_0)\) is computed analytically for the parabola by solving the
cubic first-order condition for the nearest point on \(y=x^2\).

## Important Distinctions

The projection-head method trains through a numerical projection operator and
uses an explicit distance-to-\(z_0\) term. It is directly biased toward the
nearest projected point selected by that projection head.

The KKT methods train the network to be a fixed point of one primal-dual solver
step. A zero KKT fixed-point loss implies

$$
F(\Phi_\theta(z_0); z_0) = 0
$$

when the selected KKT matrix is nonsingular. This means the network output
satisfies first-order optimality conditions. For nonconvex manifolds, a KKT
point is not automatically the globally nearest projection unless additional
regularity, uniqueness, and second-order conditions hold.

The exact KKT method uses second derivatives through
\(\nabla_y F\). The SQP/Gauss-Newton method drops those second derivatives and
uses only the Jacobian \(J_g(z)\).
