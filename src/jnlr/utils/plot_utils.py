import numpy as np
import plotly.graph_objects as go
from jnlr.reconcile import make_solver_alm_optax
import jax.numpy as jnp
from jax import vmap
from jnlr.utils.function_utils import f_impl
import plotly.express as px

def plot_mesh_plotly(vertices: np.ndarray, triangles: np.ndarray, *,
                     color: str = None,
                     show_edges: bool = True,
                     edge_color: str = "black",
                     edge_width: float = 1.0,
                     opacity: float = .8, title="Mesh",
                     lines=None, points=None, colorscale="Purples", line_color=None):
    """
    Plot a triangular mesh with Plotly.

    Parameters
    ----------
    vertices : (n, 3) array
        vertex positions.
    triangles : (m, 3) array
        Triangle indices.
    color : str or array
        Surface color (name or per-vertex scalar array for colormap).
    show_edges : bool
        If True, also show mesh edges.
    edge_color : str
        Color of mesh edges.
    edge_width : float
        Width of mesh edges.
    opacity : float
        Surface opacity.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    intensity = vertices[:, -1]
    vmin, vmax = np.nanmin(intensity), np.nanmax(intensity)
    rng = vmax - vmin
    if rng == 0 or not np.isfinite(rng):
        # force a small spread so the colorscale has something to map
        intensity_norm = np.zeros_like(intensity)
    else:
        intensity_norm = (intensity - vmin) / (rng + 1e-12)
    # Main surface
    mesh = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=color if isinstance(color, str) else None,
        intensity=intensity_norm,
        colorscale=colorscale if not isinstance(color, str) else None,
        showscale=False,
        opacity=opacity,
        flatshading=True,
    )

    data = [mesh]

    # Optional edges
    if show_edges:
        Xe, Ye, Ze = [], [], []
        for f in triangles:
            for e in [(0, 1), (1, 2), (2, 0)]:
                Xe += [vertices[f[e[0]], 0], vertices[f[e[1]], 0], None]
                Ye += [vertices[f[e[0]], 1], vertices[f[e[1]], 1], None]
                Ze += [vertices[f[e[0]], 2], vertices[f[e[1]], 2], None]
        edges = go.Scatter3d(
            x=Xe, y=Ye, z=Ze,
            mode="lines",
            line=dict(color=edge_color, width=edge_width),
            hoverinfo="none",
            opacity=0.1,
        )
        data.append(edges)

    fig = go.Figure(data=data)

    # add all the pahts in lines if lines is not None
    if lines is not None:
        N = len(lines)
        colors = px.colors.sample_colorscale("Viridis", [i / N for i in range(N)])

        if not isinstance(lines, list) and lines.ndim == 2:
            lines = [lines]
        for i, line in enumerate(lines):
            showlegend = (i == 0)  or line_color is None  # only show legend once if line_color is None
            color_i = colors[i % len(colors)] if line_color is None else line_color
            line = np.asarray(line)
            fig.add_trace(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2],
                mode="lines",
                line=dict(color=color_i, width=2),
                hoverinfo="none",
                showlegend=showlegend,
                name="Path" if showlegend else None
            ))

    if points is not None:
        points = np.asarray(points)
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=4, color="rgba(120,120,120,0.9)"),
            hoverinfo="none",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube"
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
        title=title
    )


    return fig


def plot_3d_projection(X, f_explicit=None, f_implicit=None, square_cutoff=1.5, round_cutoff=1.5, W=None, n_iterations=10,
                       solver_builder=None, plot_history=False, colorscale="Purples", lo=None, hi=None,
                       remove_axes=False, proj_alpha=1.0,
                       orig_color="rgba(120,120,120,0.9)", orig_size=2,
                       proj_color="crimson", proj_size=2, n_grid=80, shrink_projection=False, **kwargs_fig):
    """
    Plots original points X and their projections X_proj onto a surface defined by f_paraboloid.
    """
    fig = go.Figure(**kwargs_fig)
    X_np = np.asarray(X)

    if f_explicit is not None or f_impl is not None:
        solver_builder = make_solver_alm_optax if solver_builder is None else solver_builder
        f_implicit = f_implicit if f_implicit is not None else f_impl(f_explicit)

        W = jnp.eye(X.shape[1]) if W is None else W
        solver = solver_builder(f_implicit, W, n_iterations=n_iterations, return_history=plot_history)
        X_proj = solver(X)


        # Convert to NumPy for plotting
        Xp_np =np.asarray(X_proj[:, -1, :]) if plot_history else  np.asarray(X_proj)

        if shrink_projection:
            # Blend projected points: 95% projected + 5% original
            Xp_np = 0.95 * Xp_np + 0.05 * X_np

        # Determine surface domain from data (pad a bit)
        all_xy = np.vstack([X_np[:, :2], Xp_np[:, :2]])
        mins = np.nanmin(all_xy, axis=0)
        maxs = np.nanmax(all_xy, axis=0)
        lo = lo if lo is not None else mins
        hi = hi if hi is not None else maxs

        # Build grid over x0,x1 and evaluate y = f_paraboloid([x0,x1])
        if f_explicit is not None:
            gx = jnp.linspace(lo[0], hi[0], n_grid)
            gy = jnp.linspace(lo[1], hi[1], n_grid)
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
            X0 = np.asarray(X0)
            X1 = np.asarray(X1)
            Z = jnp.reshape(Z, X0.shape)
        else:
            x = jnp.linspace(lo[0], hi[0], n_grid)
            y = jnp.linspace(lo[0], hi[0], n_grid)
            z = jnp.linspace(lo[0], hi[0], n_grid)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

            points = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
            f_vals = vmap(f_implicit)(points)
            mask = jnp.abs(f_vals) < 0.05
            surface_points = points[mask]
            X0 = np.asarray(surface_points[:, 0]).ravel()
            X1 = np.asarray(surface_points[:, 1]).ravel()
            Z = np.asarray(surface_points[:, 2]).ravel()

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
        if f_explicit is not None:
            fig.add_trace(go.Surface(
                x=X0, y=X1, z=Z,
                colorscale=colorscale, showscale=False, opacity=0.55, name="surface",
                lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
            ))
        else:
            # add Mesh3d
            fig.add_trace(go.Mesh3d(
               x=X0,
               y=X1,
               z=Z,
               alphahull=0,
               opacity=0.55,
               intensity=Z,
               colorscale=colorscale,
               showscale=False,
               name="surface",
               lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0)
            ))

        # Projected points
        fig.add_trace(go.Scatter3d(
            x=Xp_np[:, 0], y=Xp_np[:, 1], z=Xp_np[:, 2],
            mode="markers",
            marker=dict(size=proj_size, color=proj_color),
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
                line=dict(color=f"rgba(0,0,0,{proj_alpha})", width=1),
                name="links"
            ))

        if remove_axes:
            axis_config = dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                showbackground=False,  # This removes the colored 'walls' of the cube
                visible=False
            )

            fig.update_layout(
                scene=dict(
                    xaxis=axis_config,
                    yaxis=axis_config,
                    zaxis=axis_config
                )
            )

    # Original points
    fig.add_trace(go.Scatter3d(
        x=X_np[:, 0], y=X_np[:, 1], z=X_np[:, 2],
        mode="markers",
        marker=dict(size=orig_size, color=orig_color),
        name="original"
    ))


    fig.update_layout(title='Surface and projections',
                      scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis', aspectmode='cube'),
                      legend=dict(itemsizing="constant"),
                      height=800)

    return fig