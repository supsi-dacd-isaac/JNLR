import numpy as np
import plotly.graph_objects as go
from jnlr.reconcile import make_solver
import jax.numpy as jnp
from jax import vmap
from jnlr.utils.function_utils import f_impl


def plot_3d_projection(X, f_explicit=None, square_cutoff=1.5, round_cutoff=1.5, W=None, n_iterations=10, solver_builder=None, plot_history=False):
    """
    Plots original points X and their projections X_proj onto a surface defined by f_paraboloid.
    """
    fig = go.Figure()
    X_np = np.asarray(X)

    if f_explicit is not None:
        solver_builder = make_solver if solver_builder is None else solver_builder
        f_implicit = f_impl(f_explicit)
        W = jnp.eye(X.shape[1]) if W is None else W
        solver = solver_builder(f_implicit, W, n_iterations=n_iterations, return_history=plot_history)
        X_proj = solver(X)


        # Convert to NumPy for plotting
        Xp_np =np.asarray(X_proj[:, -1, :]) if plot_history else  np.asarray(X_proj)

        # Determine surface domain from data (pad a bit)
        all_xy = np.vstack([X_np[:, :2], Xp_np[:, :2]])
        mins = all_xy.min(axis=0)
        maxs = all_xy.max(axis=0)
        lo = mins
        hi = maxs

        # Build grid over x0,x1 and evaluate y = f_paraboloid([x0,x1])
        gx = jnp.linspace(lo[0], hi[0], 80)
        gy = jnp.linspace(lo[1], hi[1], 80)
        X0, X1 = jnp.meshgrid(gx, gy, indexing="xy")

        # cut it within the circle with radius 1.5
        if round_cutoff is not None:
            mask = X0 ** 2 + X1 ** 2 <= square_cutoff ** 2
            X0 = jnp.where(mask, X0, jnp.nan)
            X1 = jnp.where(mask, X1, jnp.nan)

        if square_cutoff is not None:
            X0 = jnp.clip(X0, -square_cutoff, square_cutoff)
            X1 = jnp.clip(X1, -square_cutoff, square_cutoff)

        grid_pairs = jnp.stack([X0.ravel(), X1.ravel()], axis=1)  # (G, 2)
        Z = vmap(f_explicit)(grid_pairs)
        Z = jnp.reshape(Z, X0.shape)

        # Prepare dashed line segments from X to X_proj using NaNs to break segments
        n = X_np.shape[0]
        x_lines = np.empty((n, 3), dtype=float)
        y_lines = np.empty((n, 3), dtype=float)
        z_lines = np.empty((n, 3), dtype=float)
        # original -> projected -> NaN (break)
        x_lines[:, 0], y_lines[:, 0], z_lines[:, 0] = X_np[:, 0], X_np[:, 1], X_np[:, 2]
        x_lines[:, 1], y_lines[:, 1], z_lines[:, 1] = Xp_np[:, 0], Xp_np[:, 1], Xp_np[:, 2]
        x_lines[:, 2] = y_lines[:, 2] = z_lines[:, 2] = np.nan


        # Paraboloid surface
        fig.add_trace(go.Surface(
            x=np.asarray(X0), y=np.asarray(X1), z=np.asarray(Z),
            colorscale="Viridis", showscale=False, opacity=0.55, name="paraboloid"
        ))

        # Projected points
        fig.add_trace(go.Scatter3d(
            x=Xp_np[:, 0], y=Xp_np[:, 1], z=Xp_np[:, 2],
            mode="markers",
            marker=dict(size=2, color="crimson"),
            name="projected"
        ))

        # add links
        if plot_history:
            for i in range(X_np.shape[0]):
                history = np.asarray(X_proj[i])
                start = np.asarray(X_np[i], dtype=history.dtype).reshape(1, -1)
                path = np.concatenate([start, history], axis=0)
                fig.add_trace(go.Scatter3d(
                    x=path[:, 0], y=path[:, 1], z=path[:, 2],
                    mode="lines+markers",
                    line=dict(color="black", width=1),
                    marker=dict(size=1, color="black"),
                    name="history",
                    showlegend=(i == 0)  # only show legend once
                ))


        else:
            fig.add_trace(go.Scatter3d(
                x=x_lines.ravel(), y=y_lines.ravel(), z=z_lines.ravel(),
                mode="lines",
                line=dict(color="black", width=1),
                name="links"
            ))

    # Original points
    fig.add_trace(go.Scatter3d(
        x=X_np[:, 0], y=X_np[:, 1], z=X_np[:, 2],
        mode="markers",
        marker=dict(size=2, color="rgba(120,120,120,0.9)"),
        name="original"
    ))


    fig.update_layout(title='Surface and projections',
                      scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis', aspectmode='cube'),
                      legend=dict(itemsizing="constant"),
                      height=800)


    return fig