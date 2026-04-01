#!/usr/bin/env python3
"""
Render turntable animations matching the last plots of:
  compute_geodesics.ipynb, meshes.ipynb, projection_hypersurfaces.ipynb, samplers.ipynb

Requires:
  - kaleido + imageio-ffmpeg (dev group): uv run --group dev python scripts/render_notebook_spin_reel.py
  - Uses bundled ffmpeg from imageio-ffmpeg if system ffmpeg is missing.
  - System ffmpeg/ffprobe on PATH is optional (e.g. for codecs your bundle lacks, such as libwebp_anim).

  Each notebook figure is built once; only the 3D camera changes per frame. Frames are exported with
  ``plotly.io.write_images`` (one Kaleido/Chromium session) instead of ``write_image`` per frame.

Examples:
  uv run --group dev python scripts/render_notebook_spin_reel.py --quick --frames 24
  uv run --group dev python scripts/render_notebook_spin_reel.py --no-webp --out-dir artifacts/notebook_spin
  uv run --group dev python scripts/render_notebook_spin_reel.py --segment-webp   # also spin_*.webp under out-dir

By default the merged reel is written as animated WebP to docs/assets/ (MP4 segments + merged MP4
stay under --out-dir for the xfade pipeline). Use --no-webp to skip the docs WebP.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# JAX env before jax import (mirrors notebooks)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Default merged animation for README / MkDocs (animated WebP embeds cleanly in static sites).
DOCS_MERGED_WEBP = Path("docs/assets/notebook_reel_merged.webp")


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    sys.exit(
        "ffmpeg not found. Install ffmpeg on PATH or install the dev group "
        "(imageio-ffmpeg bundles a minimal ffmpeg): uv sync --group dev"
    )


def _ffprobe_exe(ffmpeg_exe: str) -> str | None:
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    cand = Path(ffmpeg_exe).parent / "ffprobe"
    return str(cand) if cand.is_file() else None


def _ffprobe_duration(path: Path, ffprobe: str) -> float:
    r = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip())


def _camera_state(fig) -> tuple[dict, dict, dict]:
    cam = fig.layout.scene.camera
    if cam is None:
        eye = dict(x=1.25, y=1.25, z=1.25)
        center = dict(x=0.0, y=0.0, z=0.0)
        up = dict(x=0.0, y=0.0, z=1.0)
        return eye, center, up
    eye = cam.eye
    center = cam.center
    up = cam.up
    if eye is None:
        eye = dict(x=1.25, y=1.25, z=1.25)
    else:
        d = {k: eye[k] for k in ("x", "y", "z")}
        if any(v is None for v in d.values()):
            eye = dict(x=1.25, y=1.25, z=1.25)
        else:
            eye = {k: float(d[k]) for k in ("x", "y", "z")}
    if center is None:
        center = dict(x=0.0, y=0.0, z=0.0)
    else:
        dc = {k: center[k] for k in ("x", "y", "z")}
        if any(v is None for v in dc.values()):
            center = dict(x=0.0, y=0.0, z=0.0)
        else:
            center = {k: float(dc[k]) for k in ("x", "y", "z")}
    if up is None:
        up = dict(x=0.0, y=0.0, z=1.0)
    else:
        du = {k: up[k] for k in ("x", "y", "z")}
        if any(v is None for v in du.values()):
            up = dict(x=0.0, y=0.0, z=1.0)
        else:
            up = {k: float(du[k]) for k in ("x", "y", "z")}
    return eye, center, up


def _orbit_eye(base_eye: dict, center: dict, azimuth_deg: float) -> dict:
    """Camera eye after rotating base_eye around vertical z through center by azimuth_deg."""
    vx = base_eye["x"] - center["x"]
    vy = base_eye["y"] - center["y"]
    vz = base_eye["z"] - center["z"]
    r = math.hypot(vx, vy)
    theta0 = math.atan2(vy, vx)
    theta = theta0 + math.radians(azimuth_deg)
    return dict(
        x=center["x"] + r * math.cos(theta),
        y=center["y"] + r * math.sin(theta),
        z=center["z"] + vz,
    )


def apply_azimuth(fig, azimuth_deg: float) -> None:
    """Rotate the camera from its *current* pose by ``azimuth_deg`` around z (additive per call)."""
    eye0, center, up = _camera_state(fig)
    eye = _orbit_eye(eye0, center, azimuth_deg)
    fig.update_layout(scene=dict(camera=dict(eye=eye, center=center, up=up)))


def render_spin_mp4(fig, mp4_path: Path, *, n_frames: int, fps: int, scale: int = 1) -> None:
    import copy

    import plotly.graph_objects as go
    import plotly.io as pio

    assert isinstance(fig, go.Figure)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    base_eye, center, up = _camera_state(fig)
    fd = fig.to_dict()
    shared_data = fd["data"]
    layout0 = fd["layout"]

    with tempfile.TemporaryDirectory(prefix="spin_frames_") as tmp:
        td = Path(tmp)
        paths = [td / f"frame_{i:05d}.png" for i in range(n_frames)]
        fig_specs: list[dict] = []
        for i in range(n_frames):
            az = 360.0 * i / n_frames
            eye = _orbit_eye(base_eye, center, az)
            layout = copy.deepcopy(layout0)
            scene = layout.setdefault("scene", {})
            cam = scene.setdefault("camera", {})
            cam["eye"] = eye
            cam["center"] = center
            cam["up"] = up
            fig_specs.append({"data": shared_data, "layout": layout})

        # One persistent Kaleido/Chromium session for all frames (much faster than N× write_image).
        pio.write_images(fig_specs, paths, scale=scale, validate=False)
        pattern = str(td / "frame_%05d.png")
        cmd = [
            _ffmpeg_exe(),
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)


def mp4_to_webp(mp4_path: Path, webp_path: Path) -> None:
    # libwebp_anim is common on brew ffmpeg; bundled ffmpeg may lack it
    webp_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(mp4_path),
        "-c:v",
        "libwebp_anim",
        "-quality",
        "82",
        "-loop",
        "0",
        str(webp_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(
            r.stderr
            or r.stdout
            or "ffmpeg webp encode failed (try a build with libwebp_anim).\n"
        )
        r.check_returncode()


def merge_xfade(
    mp4_paths: list[Path],
    out_path: Path,
    fade_s: float,
    *,
    clip_duration_s: float | None = None,
) -> None:
    if not mp4_paths:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(mp4_paths) == 1:
        shutil.copy2(mp4_paths[0], out_path)
        return
    if clip_duration_s is not None:
        T = clip_duration_s
    else:
        ffprobe = _ffprobe_exe(_ffmpeg_exe())
        if ffprobe is None:
            sys.exit("ffprobe not found; pass clip duration or install ffprobe next to ffmpeg.")
        T = _ffprobe_duration(mp4_paths[0], ffprobe)
        for p in mp4_paths[1:]:
            if abs(_ffprobe_duration(p, ffprobe) - T) > 0.05:
                sys.stderr.write(
                    f"Warning: clip durations differ ({p}); xfade assumes equal length {T}s.\n"
                )
    cur = "[0:v]"
    accum = T
    parts: list[str] = []
    for i in range(1, len(mp4_paths)):
        nxt = f"[{i}:v]"
        offset = max(accum - fade_s, 0.0)
        out = f"[vx{i}]" if i < len(mp4_paths) - 1 else "[vout]"
        parts.append(f"{cur}{nxt}xfade=transition=fade:duration={fade_s}:offset={offset:.6f}{out}")
        accum = accum + T - fade_s
        cur = out
    fc = ";".join(parts)
    cmd = [_ffmpeg_exe(), "-y"]
    for p in mp4_paths:
        cmd.extend(["-i", str(p)])
    cmd.extend(
        [
            "-filter_complex",
            fc,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# --- figure builders (mirror last notebook cells) ---------------------------------

# Single size for all segments so ffmpeg xfade can concatenate streams.
PLOT_W_DEFAULT = 500
PLOT_H_DEFAULT = 400


def _style_like_mesh_plotly(fig) -> None:
    """Strip axes panes, grid, and legend to match ``plot_mesh_plotly`` (e.g. torus)."""
    axis = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        showspikes=False,
        title="",
        zeroline=False,
    )
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectmode="cube",
            bgcolor="white",
        ),
        paper_bgcolor="white",
    )


def build_fig_geodesics(*, quick: bool) -> "object":
    import jax
    import jax.numpy as jnp
    from jnlr.geodesics.compute import GeodesicSolver
    from jnlr.utils.manifolds import f_ackley as f_expl
    from jnlr.utils.plot_utils import plot_mesh_plotly
    from jnlr.utils.samplers import sample

    ranges = ((-1.3, 1.3), (-1.3, 1.3))
    n_mesh = 400 if quick else 1000
    gs = GeodesicSolver(f_expl, n_samples=n_mesh, ranges=ranges)

    x0 = jnp.array([-1.0, -1.0])
    z0 = jnp.hstack([x0, f_expl(x0)])

    vol_n = 800 if quick else 5000
    samples = sample(phi=f_expl, method="volume", n_samples=vol_n, bounds=ranges)
    gs_graph = GeodesicSolver(f_expl, samples=samples, ranges=ranges, method="graph")

    n_cloud = 20 if quick else 100
    perturbations = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (n_cloud, 2)) + jnp.array([1.3, 1.3])
    z_tilde = jnp.vstack([jnp.hstack([x0 + pert, f_expl(x0 + pert)]) for pert in perturbations])

    gamma_gs, _ = gs_graph.geodesic(z_tilde, jnp.tile(z0, len(z_tilde)).reshape(-1, 3))

    title = (
        "Geodesics — graph method on Ackley (compute_geodesics.ipynb)"
    )
    fig = plot_mesh_plotly(
        **gs.mesh,
        title=title,
        lines=gamma_gs,
        show_edges=True,
        points=jnp.vstack([z0[None, :], z_tilde]),
        line_color="red",
        width=PLOT_W_DEFAULT,
        height=PLOT_H_DEFAULT,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=48), title=dict(y=0.96))
    _style_like_mesh_plotly(fig)
    return fig


def build_fig_meshes(*, quick: bool) -> "object":
    import jax.numpy as jnp
    from jnlr.utils.meshes import get_mesh
    from jnlr.utils.plot_utils import plot_mesh_plotly

    R, r = 2.0, 0.7
    nu = nv = 24 if quick else 50

    def phi_torus(U):
        u, v = U
        cu, su = jnp.cos(u), jnp.sin(u)
        cv, sv = jnp.cos(v), jnp.sin(v)
        x = (R + r * cv) * cu
        y = (R + r * cv) * su
        z = r * sv
        return jnp.stack([x, y, z])

    V, F = get_mesh(
        phi_torus,
        "explicit",
        nu=nu,
        nv=nv,
        grid_ranges=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
    )
    title = "Meshes — explicit torus (meshes.ipynb)"
    fig = plot_mesh_plotly(V, F, title=title, width=PLOT_W_DEFAULT, height=PLOT_H_DEFAULT)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=48), title=dict(y=0.96))
    _style_like_mesh_plotly(fig)
    return fig


def build_fig_projection(*, quick: bool) -> "object":
    import jax.numpy as jnp
    import numpy as np
    from jnlr.reconcile import make_solver_alm_optax as make_solver
    from jnlr.utils.plot_utils import plot_3d_projection

    NGRID = 28 if quick else 50
    n_samples = 40 if quick else 100

    def f_paraboloid(v):
        x, y = v[0], v[1]
        return x**2 + y**2

    def f_implicit_paraboloid(v):
        z = v[2]
        return f_paraboloid(v[:2]) - z

    np.random.seed(0)
    X = np.random.normal(size=(n_samples, 3), scale=0.2) + np.array([0.9, 0.9, 0])[None, :]

    title = "Projection — paraboloid + KDE (projection_hypersurfaces.ipynb)"
    fig = plot_3d_projection(
        X,
        f_paraboloid,
        show_kde=True,
        round_cutoff=None,
        solver_builder=make_solver,
        plot_history=True,
        n_iterations=4,
        n_grid=NGRID,
        lo=-2 * np.ones(2),
        hi=2 * np.ones(2),
        n_isolines=7,
        width=PLOT_W_DEFAULT,
        height=PLOT_H_DEFAULT,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=48),
        title=dict(text=title, y=0.97),
    )
    _style_like_mesh_plotly(fig)
    return fig


def build_fig_samplers(*, quick: bool) -> "object":
    from jnlr.utils.implicit_hypersurfaces import surface_b
    from jnlr.utils.plot_utils import plot_3d_projection
    from jnlr.utils.samplers import langevin_implicit

    n = 400 if quick else 10000
    burn = 20 if quick else 100
    Y_lang = langevin_implicit(
        surface_b,
        n_samples=n,
        burn=burn,
        thin=1,
        sigma=0.3,
        lam=0,
        kappa=0,
        R=5.0,
        tol=1e-2,
    )
    title = "Sampling — Langevin on implicit surface B (samplers.ipynb)"
    fig = plot_3d_projection(Y_lang, width=PLOT_W_DEFAULT, height=PLOT_H_DEFAULT)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=48),
        title=dict(text=title, y=0.97),
    )
    _style_like_mesh_plotly(fig)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Turntable renders for JNLR example notebooks.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/notebook_spin"),
        help="Output directory for segment and merged MP4 files",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=90, help="Frames per full 360° rotation")
    parser.add_argument("--scale", type=int, default=1, help="Kaleido scale factor (2 = 2x resolution)")
    parser.add_argument("--fade", type=float, default=0.6, help="Crossfade duration in seconds (merged video)")
    parser.add_argument(
        "--no-webp",
        action="store_true",
        help="Do not write animated WebP to docs/assets/ (only merged MP4 under --out-dir).",
    )
    parser.add_argument(
        "--docs-merged",
        type=Path,
        default=DOCS_MERGED_WEBP,
        help="Path for the merged animated WebP written for documentation (default: docs/assets/...).",
    )
    parser.add_argument(
        "--segment-webp",
        action="store_true",
        help="Also encode each segment as .webp next to its .mp4 under --out-dir.",
    )
    parser.add_argument("--quick", action="store_true", help="Faster lower-quality compute for testing")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-geodesics", action="store_true")
    parser.add_argument("--skip-meshes", action="store_true")
    parser.add_argument("--skip-projection", action="store_true")
    parser.add_argument("--skip-samplers", action="store_true")
    args = parser.parse_args()

    clip_dur = args.frames / float(args.fps)
    if args.fade >= clip_dur and not args.skip_merge:
        sys.exit(f"--fade ({args.fade}s) must be less than clip duration ({clip_dur:.4f}s = frames/fps).")

    _ffmpeg_exe()

    segments: list[tuple[str, Path, object]] = []

    if not args.skip_geodesics:
        print("Building geodesics figure…")
        fig = build_fig_geodesics(quick=args.quick)
        out = args.out_dir / "spin_geodesics.mp4"
        print(f"Rendering {out} ({args.frames} frames)…")
        render_spin_mp4(fig, out, n_frames=args.frames, fps=args.fps, scale=args.scale)
        segments.append(("geodesics", out, fig))

    if not args.skip_meshes:
        print("Building meshes figure…")
        fig = build_fig_meshes(quick=args.quick)
        out = args.out_dir / "spin_meshes.mp4"
        print(f"Rendering {out}…")
        render_spin_mp4(fig, out, n_frames=args.frames, fps=args.fps, scale=args.scale)
        segments.append(("meshes", out, fig))

    if not args.skip_projection:
        print("Building projection figure…")
        fig = build_fig_projection(quick=args.quick)
        out = args.out_dir / "spin_projection.mp4"
        print(f"Rendering {out}…")
        render_spin_mp4(fig, out, n_frames=args.frames, fps=args.fps, scale=args.scale)
        segments.append(("projection", out, fig))

    if not args.skip_samplers:
        print("Building samplers figure…")
        fig = build_fig_samplers(quick=args.quick)
        out = args.out_dir / "spin_samplers.mp4"
        print(f"Rendering {out}…")
        render_spin_mp4(fig, out, n_frames=args.frames, fps=args.fps, scale=args.scale)
        segments.append(("samplers", out, fig))

    mp4_list = [p for _, p, _ in segments]

    if args.segment_webp:
        for name, mp4_path, _ in segments:
            webp_path = mp4_path.with_suffix(".webp")
            print(f"Encoding segment WebP {webp_path}…")
            try:
                mp4_to_webp(mp4_path, webp_path)
            except subprocess.CalledProcessError:
                sys.stderr.write(f"Skipping WebP for {name} (encoder failed).\n")

    if not args.skip_merge and len(mp4_list) >= 1:
        merged_mp4 = args.out_dir / "notebook_reel_merged.mp4"
        print(f"Merging → {merged_mp4} (fade {args.fade}s)…")
        merge_xfade(mp4_list, merged_mp4, fade_s=args.fade, clip_duration_s=clip_dur)

        if not args.no_webp:
            dest = args.docs_merged
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"Encoding docs WebP → {dest}…")
            try:
                mp4_to_webp(merged_mp4, dest)
            except subprocess.CalledProcessError:
                fb = dest.with_suffix(".mp4")
                sys.stderr.write(
                    f"libwebp_anim encode failed; copying merged MP4 to {fb} instead.\n"
                )
                shutil.copy2(merged_mp4, fb)

    print("Done.")


if __name__ == "__main__":
    main()
