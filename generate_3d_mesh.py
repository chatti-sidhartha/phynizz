"""
generate_3d_mesh.py — Generate a 3D mesh image from an input photo.

Usage
-----
    # Activate venv first:
    #   .\\venv\\Scripts\\activate

    python generate_3d_mesh.py test_images/full_body.jpg
    python generate_3d_mesh.py test_images/full_body.jpg --output-dir output --gender male
    python generate_3d_mesh.py test_images/full_body.jpg --device cpu

Output
------
    output/mesh_3d_<basename>.png   — multi-view solid mesh image (Front / Side / 3D)
    output/mesh_3d_<basename>.obj   — Wavefront OBJ for use in Blender / MeshLab
"""

from __future__ import annotations

import os
import sys
import argparse
import types
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe on Windows / headless
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── make sure project modules are importable regardless of CWD ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'modules'))


# ── chumpy monkey-patch ──────────────────────────────────────────────────────
# SMPL v1.1 .pkl files embed chumpy arrays.  We intercept them at pickle load
# time and transparently convert them to plain numpy arrays.
#
# IMPORTANT: _Ch must NOT subclass np.ndarray — doing so causes
#   numpy.ndarray.__new__() failures during unpickling because numpy tries to
#   call __new__(cls, <shape_tuple>) which conflicts with our (x=None) sig.
#   Using a plain Python class that implements __array__ is safe.

def _make_chumpy_stub() -> types.ModuleType:
    """Return a minimal 'chumpy' stub that satisfies SMPL .pkl imports."""

    class _Ch:
        """Stand-in for chumpy.Ch — stores any value, converts to ndarray on demand."""
        def __init__(self, x=None, *args, **kwargs):
            self._arr = np.asarray(x) if x is not None else np.zeros(0)

        def __array__(self, dtype=None):
            return self._arr if dtype is None else self._arr.astype(dtype)

        # ndarray-like interface
        @property
        def shape(self): return self._arr.shape
        @property
        def dtype(self): return self._arr.dtype
        def __len__(self): return len(self._arr)
        def __getitem__(self, key): return self._arr[key]
        def __setitem__(self, key, val): self._arr[key] = val
        def __repr__(self): return f'Ch({self._arr!r})'

    def _ndfactory(*args, **kwargs):
        return np.asarray(args[0]) if args else np.zeros(0)

    # top-level chumpy
    chumpy = types.ModuleType('chumpy')
    setattr(chumpy, 'Ch', _Ch)
    sys.modules['chumpy'] = chumpy

    # chumpy.ch
    ch_sub = types.ModuleType('chumpy.ch')
    setattr(ch_sub, 'Ch', _Ch)
    for name in ('SparseMatrix', 'MatVecMult', 'dot', 'sum', 'maximum',
                 'negative', 'power', 'abs', 'ceil', 'floor', 'sqrt',
                 'exp', 'log', 'sin', 'cos', 'tan', 'arctan2'):
        setattr(ch_sub, name, _ndfactory)
    chumpy.ch = ch_sub  # type: ignore[attr-defined]
    sys.modules['chumpy.ch'] = ch_sub

    # other sub-packages that some SMPL variants reference
    for sub in ('utils', 'reordering', 'linalg', 'logic', 'indexing',
                'minimize', 'optimization'):
        sub_mod = types.ModuleType(f'chumpy.{sub}')
        setattr(chumpy, sub, sub_mod)
        sys.modules[f'chumpy.{sub}'] = sub_mod

    return chumpy


if 'chumpy' not in sys.modules:
    _make_chumpy_stub()
# ────────────────────────────────────────────────────────────────────────────

from modules.hmr_estimator import HMREstimator
from modules.smpl_utils import SMPLBodyModel


# ════════════════════════════════════════════════════════════════════════════
#  Core pipeline
# ════════════════════════════════════════════════════════════════════════════

def generate_3d_mesh(
    image_path: str,
    output_dir: str = 'output',
    gender: str = 'male',
    device: str = 'cpu',
) -> dict:
    """
    Generate 3D SMPL mesh from a single person image.

    Parameters
    ----------
    image_path : str
        Path to the input image (JPG / PNG).
    output_dir : str
        Directory where output files are written.
    gender : str
        'male' or 'female' (selects SMPL body model).
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    dict with keys:
        png_path  - path to the rendered mesh PNG
        obj_path  - path to the exported OBJ file
        vertices  - (6890, 3) ndarray in metres
        faces     - (13776, 3) ndarray of triangle indices
    """

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ── 1. HMR inference ────────────────────────────────────────────────────
    print(f"\n[1/4] Running HMR inference on '{os.path.basename(image_path)}'...")
    hmr = HMREstimator(device=device)
    hmr_result = hmr.estimate(image_path)

    pose_np  = hmr_result['pose']   # (72,)
    shape_np = hmr_result['shape']  # (10,)
    print(f"      Pose shape : {pose_np.shape}   Shape params : {shape_np.shape}")

    # ── 2. Generate SMPL mesh ───────────────────────────────────────────────
    print("[2/4] Generating SMPL mesh...")
    import torch
    smpl = SMPLBodyModel(device=device)

    pose_t  = torch.from_numpy(pose_np).float().unsqueeze(0)
    shape_t = torch.from_numpy(shape_np).float().unsqueeze(0)
    trans_t = torch.zeros(1, 3).float()

    smpl_out = smpl.forward(betas=shape_t, pose=pose_t, trans=trans_t, gender=gender)
    vertices = smpl_out['vertices']  # (6890, 3)  — metres
    faces    = smpl_out['faces']     # (13776, 3) — int
    print(f"      Vertices : {vertices.shape}  |  Faces : {faces.shape}")

    # ── 3. Render multi-view PNG ────────────────────────────────────────────
    print("[3/4] Rendering multi-view mesh image...")
    png_path = _render_mesh_png(vertices, faces, image_path, output_dir, base)

    # ── 4. Export OBJ ──────────────────────────────────────────────────────
    print("[4/4] Exporting OBJ file...")
    obj_path = _export_obj(vertices, faces, output_dir, base)

    print(f"\nDone!")
    print(f"    PNG  ->  {png_path}")
    print(f"    OBJ  ->  {obj_path}")

    return dict(png_path=png_path, obj_path=obj_path, vertices=vertices, faces=faces)


# ════════════════════════════════════════════════════════════════════════════
#  Rendering helpers
# ════════════════════════════════════════════════════════════════════════════

def _render_mesh_png(
    vertices: np.ndarray,
    faces: np.ndarray,
    image_path: str,
    output_dir: str,
    base: str,
) -> str:
    """
    Render mesh with a shaded surface (Poly3DCollection) in three views.

    Uses matplotlib LightSource for realistic Lambertian shading.
    Downsamples faces to keep rendering fast (~5 000 triangles).
    """

    # ── centre & normalise to [-1, 1] ───────────────────────────────────────
    verts = vertices - vertices.mean(axis=0)
    scale = np.abs(verts).max()
    verts = verts / (scale + 1e-9)

    # ── subsample faces for speed (keep ~5 000 triangles) ───────────────────
    n_faces_target = 5_000
    if len(faces) > n_faces_target:
        idx = np.random.choice(len(faces), n_faces_target, replace=False)
        faces_draw = faces[idx]
    else:
        faces_draw = faces

    # Triangle arrays: (N, 3, 3)
    tris = verts[faces_draw]

    # ── per-face height coloring (plasma colormap, feet=dark → head=bright) ─
    mean_y   = tris[:, :, 1].mean(axis=1)          # average Y of each triangle
    y_norm   = (mean_y - mean_y.min()) / (mean_y.ptp() + 1e-9)
    cmap     = plt.cm.plasma
    fc_rgba  = cmap(y_norm)                         # (N, 4) RGBA

    # ── lightsource for shading ──────────────────────────────────────────────
    ls = LightSource(azdeg=225, altdeg=45)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 7), facecolor='#0d0d0d')
    plt.subplots_adjust(left=0.02, right=0.93, top=0.92, bottom=0.05, wspace=0.1)

    # Load & show the original photo on the left
    from PIL import Image as PILImage
    try:
        orig_img = PILImage.open(image_path).convert('RGB')
        ax_img = fig.add_axes([0.01, 0.08, 0.18, 0.84])          # [left, bot, w, h]
        ax_img.imshow(orig_img)
        ax_img.axis('off')
        ax_img.set_title('Input Image', color='white', fontsize=11, pad=6)
    except Exception:
        pass

    views = [
        dict(elev=0,   azim=180,  title='Front View'),
        dict(elev=0,   azim=90,   title='Side View'),
        dict(elev=20,  azim=225,  title='3D View'),
    ]

    # Place 3 mesh subplots to the right of the photo
    for col, view in enumerate(views):
        # Manually position: starts at x=0.22
        left = 0.22 + col * 0.245
        ax = fig.add_axes([left, 0.06, 0.22, 0.88], projection='3d')

        poly = Poly3DCollection(tris, closed=False, linewidths=0)
        poly.set_facecolor(fc_rgba)
        poly.set_alpha(0.92)
        ax.add_collection3d(poly)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        ax.view_init(elev=view['elev'], azim=view['azim'])
        ax.set_facecolor('#0d0d0d')
        ax.grid(False)
        ax.set_axis_off()

        ax.set_title(view['title'], color='white', fontsize=12,
                     fontweight='bold', pad=4)

    # ── colour-bar ───────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Height (normalised)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    # ── title ────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'3D SMPL Body Mesh   -   {os.path.basename(image_path)}',
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    out_path = os.path.join(output_dir, f'mesh_3d_{base}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _export_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: str,
    base: str,
) -> str:
    """Write a Wavefront OBJ for use in Blender / MeshLab."""

    out_path = os.path.join(output_dir, f'mesh_3d_{base}.obj')
    with open(out_path, 'w') as fh:
        fh.write(f'# SMPL mesh generated by generate_3d_mesh.py\n')
        fh.write(f'# Vertices: {len(vertices)}   Faces: {len(faces)}\n\n')

        for v in vertices:
            fh.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

        fh.write('\n')
        for f in faces:
            # OBJ indices are 1-based
            fh.write(f'f {f[0]+1} {f[1]+1} {f[2]+1}\n')

    return out_path


# ════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate a 3D SMPL mesh image from a single-person photo.'
    )
    parser.add_argument('image_path', help='Path to input image (JPG / PNG)')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--gender', '-g', default='male',
                        choices=['male', 'female'],
                        help='Body gender for SMPL model (default: male)')
    parser.add_argument('--device', '-d', default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Computation device (default: cpu)')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Image not found: {args.image_path}")
        sys.exit(1)

    result = generate_3d_mesh(
        image_path=args.image_path,
        output_dir=args.output_dir,
        gender=args.gender,
        device=args.device,
    )

    print(f"\nPNG mesh  : {result['png_path']}")
    print(f"OBJ file  : {result['obj_path']}")


if __name__ == '__main__':
    main()
