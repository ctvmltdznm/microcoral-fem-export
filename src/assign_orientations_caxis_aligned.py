#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assign_orientations_caxis_aligned.py
=====================================
VARIANT: c-axis aligned with needle morphological direction.

In this model phi1/Phi are computed directly from each needle's growth
direction vector (= morphological long axis = crystallographic c-axis).
phi2 is assigned via the aragonite twin model (±half_angle per variant).

LIMITATION (noted for future reference):
Within a sclerodermite, needles radiate outward in different directions,
so adjacent-needle c-axis differences are typically 30–60°.  When
computing full 3D misorientation between adjacent needles, the c-axis
angular spread dominates over the phi2 twin signal, washing out the
twin peaks in misorientation histograms.

This variant is physically appropriate IF:
  - The sclerodermite is genuinely a polycrystalline aggregate where
    each needle has its own independent c-axis direction
  - The EBSD twin peaks come from WITHIN a single needle (pixel-to-pixel
    within one crystal domain), not between adjacent needles

See also: assign_orientations.py (grain-shared c-axis variant) where all
needles in a sclerodermite share one c-axis direction and differ only in
phi2, which produces clean twin peaks in adjacent-needle misorientation
histograms analogous to the EBSD Mackenzie plot.

Author: nk03 / pipeline

Physically correct crystallographic orientation assignment for synthetic
aragonite microstructures.

Model
-----
Aragonite is orthorhombic (point group mmm).  Within a sclerodermite all
needles share an approximate c-axis direction (= needle long axis), but the
a/b axes are related by discrete twin operations.

Euler angles follow the Bunge ZXZ convention (degrees), consistent with
MTEX, Channel 5, and the existing export_to_exodus() function.

Twin model (from EBSD Mackenzie plot)
--------------------------------------
Within each sclerodermite a base phi2 is drawn once (uniform 0-360 deg).
This phi2_base defines the mirror plane of the twin system.

Each needle then draws a VARIANT TYPE and a SIGN (+ or -):

    variant type     half-angle      misorientation between ± variants
    ───────────────  ──────────      ──────────────────────────────────
    low-angle        ~5.5°           ~11°   (broad, intra-grain scatter)
    mirror twin      26.2°           52.4°  (mirror-reflected (110))
    (310) twin       28.6°           57.2°
    (110) twin       31.9°           63.8°  (dominant peak)

    phi2 = phi2_base  ±  half_angle  +  N(0, sigma)

This ensures that two needles of OPPOSITE sign (one at +31.9°, one at -31.9°)
produce 63.8° misorientation, while two needles of the SAME sign produce ~0°.
This is the correct physics: the twin boundary is the mirror plane.

Inter-sclerodermite orientation
--------------------------------
phi2_base for each sclerodermite is uniform random in [0, 360°].
No constraint between adjacent sclerodermites (organic matter decouples them).

Usage (CLI)
-----------
    python assign_orientations.py --plot --n-grains 20 --needles-per-grain 25
    python assign_orientations.py --p-twin110 0.5 --sigma-twin 1.0 --plot

Author: nk03 / pipeline
"""

from __future__ import annotations
import argparse
import dataclasses
import sys
from typing import List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Parameter dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class OrientationParams:
    """
    All tunable parameters for the orientation assignment model.
    Defaults calibrated from the EBSD Mackenzie plot (aragonite coral).
    """

    # ── Twin variant probabilities (must sum to 1.0) ──────────────────────────
    # These are fractions of needles assigned to each twin TYPE.
    # Each type has two sub-variants (+ and -) drawn with equal probability.
    p_lowangle: float = 0.10   # low-angle GB / intra-grain scatter, peak ~11°
    p_twin110:  float = 0.45   # dominant (110) twin, peak at 63.8°
    p_twin310:  float = 0.25   # (310) twin, peak at 57.2°
    p_mirror:   float = 0.20   # mirror-reflected (110) twin, peak at 52.4°

    # ── Half-angles for each twin type (degrees) ──────────────────────────────
    # phi2 = phi2_base ± half_angle
    # Misorientation between opposite variants = 2 × half_angle
    half_angle_lowangle: float = 5.5    # → misori ~11°
    half_angle_twin110:  float = 31.9   # → misori 63.8°
    half_angle_twin310:  float = 28.6   # → misori 57.2°
    half_angle_mirror:   float = 26.2   # → misori 52.4°

    # ── Gaussian scatter around each variant position (degrees) ───────────────
    sigma_intra: float = 2.0   # scatter for low-angle / same-variant pairs
    sigma_twin:  float = 1.5   # scatter around twin variant positions

    def validate(self) -> None:
        total = self.p_lowangle + self.p_twin110 + self.p_twin310 + self.p_mirror
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Variant probabilities must sum to 1.0, got {total:.6f}."
            )

    def cumulative_probs(self) -> np.ndarray:
        return np.cumsum([
            self.p_lowangle,
            self.p_twin110,
            self.p_twin310,
            self.p_mirror,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Core orientation functions
# ──────────────────────────────────────────────────────────────────────────────

def _c_axis_to_euler(direction: np.ndarray) -> tuple[float, float]:
    """
    Convert needle direction (= crystal c-axis) to Bunge phi1, Phi (degrees).
        phi1 in [0, 360),  Phi in [0, 180]
    """
    c = direction / np.linalg.norm(direction)
    phi1 = float(np.degrees(np.arctan2(c[1], c[0])) % 360.0)
    Phi  = float(np.degrees(np.arccos(np.clip(c[2], -1.0, 1.0))))
    return phi1, Phi


def _draw_phi2(phi2_base: float, rng: np.random.Generator,
               params: OrientationParams) -> float:
    """
    Draw phi2 for one needle using the ±half_angle twin model.

    Variant type is drawn from probabilities; sign (+ or -) is 50/50.
    Gaussian scatter is added around the variant position.
    """
    cumprobs = params.cumulative_probs()
    u = rng.uniform(0.0, 1.0)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0

    if u < cumprobs[0]:
        half = params.half_angle_lowangle
        sigma = params.sigma_intra
    elif u < cumprobs[1]:
        half = params.half_angle_twin110
        sigma = params.sigma_twin
    elif u < cumprobs[2]:
        half = params.half_angle_twin310
        sigma = params.sigma_twin
    else:
        half = params.half_angle_mirror
        sigma = params.sigma_twin

    delta = sign * half + rng.normal(0.0, sigma)
    return (phi2_base + delta) % 360.0


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def assign_orientations(
    center_properties: list,
    params: Optional[OrientationParams] = None,
    seed: Optional[int] = None,
) -> list:
    """
    Assign crystallographic Euler angles to all needles.

    Adds keys  'phi1', 'Phi', 'phi2', 'phi2_base'  to each needle dict.

    Parameters
    ----------
    center_properties : list
        Output of radial_needles_more_2d().
    params : OrientationParams, optional
        Model parameters. Defaults to EBSD-calibrated values.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    center_properties (modified in-place and returned)
    """
    if params is None:
        params = OrientationParams()
    params.validate()

    rng = np.random.default_rng(seed)

    n_grains  = len(center_properties)
    n_needles = sum(len(c['needles']) for c in center_properties)
    print(f"\nAssigning orientations: {n_needles} needles in {n_grains} sclerodermites")

    for center in center_properties:
        phi2_base = rng.uniform(0.0, 360.0)   # one per sclerodermite

        for needle in center['needles']:
            direction = np.asarray(needle['direction'], dtype=float)
            phi1, Phi = _c_axis_to_euler(direction)
            phi2 = _draw_phi2(phi2_base, rng, params)

            needle['phi1']      = phi1
            needle['Phi']       = Phi
            needle['phi2']      = phi2
            needle['phi2_base'] = phi2_base

    _print_summary(center_properties)
    return center_properties


def build_needle_euler_lookup(center_properties: list) -> dict:
    """
    Build  {needle_id: {'phi1': float, 'Phi': float, 'phi2': float}}
    after assign_orientations() has been called.
    Drop-in replacement for the old random phi2 logic in export_to_exodus().
    """
    lookup = {}
    for center in center_properties:
        for needle in center['needles']:
            if 'phi1' not in needle:
                raise RuntimeError(
                    f"Needle {needle['id']} missing phi1/Phi/phi2. "
                    "Call assign_orientations() first."
                )
            lookup[needle['id']] = {
                'phi1': needle['phi1'],
                'Phi':  needle['Phi'],
                'phi2': needle['phi2'],
            }
    return lookup


# ──────────────────────────────────────────────────────────────────────────────
# Verification / plotting
# ──────────────────────────────────────────────────────────────────────────────

def _misori_mmm(rot_i, rot_j, sym_ops) -> float:
    """Minimum misorientation angle (deg) under orthorhombic mmm symmetry."""
    delta = rot_i.inv() * rot_j
    return min(np.degrees((sym * delta).magnitude()) for sym in sym_ops)


def _build_sym_ops():
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('ZXZ', [
        [0,   0,   0  ],   # identity
        [180, 0,   0  ],   # 2-fold Z
        [0,   180, 0  ],   # 2-fold X
        [180, 180, 0  ],   # 2-fold Y
    ], degrees=True)


def compute_misorientation_slices(
    center_properties: list,
    volume: np.ndarray,
    needle_volume: np.ndarray,
    n_pairs: int = 20000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    Compute intra-grain misorientation angles from 2D cross-section slices
    of the voxel volume — mimicking what EBSD sees in a planar scan.

    For each of the three mid-plane slices (XY, XZ, YZ), find all adjacent
    voxel pairs that belong to the SAME grain, then compute their misorientation.

    Returns
    -------
    dict with keys 'XY', 'XZ', 'YZ', each an array of misorientation angles.
    """
    from scipy.spatial.transform import Rotation

    # Build euler lookup
    euler_lookup = {}   # needle_id -> Rotation object
    sym_ops = _build_sym_ops()

    for center in center_properties:
        for needle in center['needles']:
            if 'phi1' not in needle:
                raise RuntimeError("Call assign_orientations() first.")
            r = Rotation.from_euler(
                'ZXZ', [needle['phi1'], needle['Phi'], needle['phi2']],
                degrees=True
            )
            euler_lookup[needle['id']] = r

    ni, nj, nk = volume.shape
    rng = np.random.default_rng(seed)
    results = {}

    # Helper: misorientation from two voxel indices
    def misori_voxels(idx1, idx2):
        nid1 = needle_volume[idx1]
        nid2 = needle_volume[idx2]
        gid1 = volume[idx1]
        gid2 = volume[idx2]
        # Only intra-grain pairs
        if gid1 != gid2 or gid1 == 0:
            return None
        if nid1 not in euler_lookup or nid2 not in euler_lookup:
            return None
        if nid1 == nid2:
            return None
        return _misori_mmm(euler_lookup[nid1], euler_lookup[nid2], sym_ops)

    # ── XY mid-plane (z = nk//2) ──────────────────────────────────────────────
    k_mid = nk // 2
    xy_angles = []
    for _ in range(n_pairs * 5):   # oversample to get n_pairs valid pairs
        i = rng.integers(0, ni - 1)
        j = rng.integers(0, nj - 1)
        # Check +X and +Y neighbours
        for di, dj in [(1, 0), (0, 1)]:
            a = misori_voxels((i, j, k_mid), (i+di, j+dj, k_mid))
            if a is not None:
                xy_angles.append(a)
        if len(xy_angles) >= n_pairs:
            break
    results['XY'] = np.array(xy_angles[:n_pairs])

    # ── XZ mid-plane (y = nj//2) ──────────────────────────────────────────────
    j_mid = nj // 2
    xz_angles = []
    for _ in range(n_pairs * 5):
        i = rng.integers(0, ni - 1)
        k = rng.integers(0, nk - 1)
        for di, dk in [(1, 0), (0, 1)]:
            a = misori_voxels((i, j_mid, k), (i+di, j_mid, k+dk))
            if a is not None:
                xz_angles.append(a)
        if len(xz_angles) >= n_pairs:
            break
    results['XZ'] = np.array(xz_angles[:n_pairs])

    # ── YZ mid-plane (x = ni//2) ──────────────────────────────────────────────
    i_mid = ni // 2
    yz_angles = []
    for _ in range(n_pairs * 5):
        j = rng.integers(0, nj - 1)
        k = rng.integers(0, nk - 1)
        for dj, dk in [(1, 0), (0, 1)]:
            a = misori_voxels((i_mid, j, k), (i_mid, j+dj, k+dk))
            if a is not None:
                yz_angles.append(a)
        if len(yz_angles) >= n_pairs:
            break
    results['YZ'] = np.array(yz_angles[:n_pairs])

    return results


def compute_misorientation_3d(
    center_properties: list,
    n_pairs: int = 30000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    3D misorientation: intra-grain (neighbour) vs random pairs.
    Returns (intra_angles, random_angles).
    """
    from scipy.spatial.transform import Rotation
    sym_ops = _build_sym_ops()

    grain_rotations = []
    all_rotations   = []

    for center in center_properties:
        grain_rots = []
        for needle in center['needles']:
            r = Rotation.from_euler(
                'ZXZ', [needle['phi1'], needle['Phi'], needle['phi2']],
                degrees=True
            )
            grain_rots.append(r)
            all_rotations.append(r)
        grain_rotations.append(grain_rots)

    rng = np.random.default_rng(seed)

    # Intra-grain pairs
    intra = []
    attempts = 0
    while len(intra) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        g = grain_rotations[rng.integers(0, len(grain_rotations))]
        if len(g) < 2:
            continue
        i, j = rng.choice(len(g), 2, replace=False)
        intra.append(_misori_mmm(g[i], g[j], sym_ops))

    # Random pairs
    n_all = len(all_rotations)
    random_pairs = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n_all, 2)
        if i != j:
            random_pairs.append(_misori_mmm(all_rotations[i], all_rotations[j], sym_ops))

    return np.array(intra), np.array(random_pairs)


def plot_verification(
    center_properties: list,
    volume: Optional[np.ndarray] = None,
    needle_volume: Optional[np.ndarray] = None,
    n_pairs: int = 20000,
    seed: int = 0,
    output_file: str = 'synthetic_mackenzie.png',
    params: Optional[OrientationParams] = None,
) -> None:
    """
    Three-panel verification plot:
      Left  — 3D intra-grain vs random-pair misorientation
      Centre — planar cross-section misorientation (XY/XZ/YZ slices), if volume provided
      Right  — reference EBSD-style Mackenzie peak positions
    """
    import matplotlib.pyplot as plt

    if params is None:
        params = OrientationParams()

    twin_angles = [
        (params.half_angle_lowangle * 2, 'Low-angle', 'C0'),
        (params.half_angle_mirror   * 2, 'Mirror',    'C1'),
        (params.half_angle_twin310  * 2, '(310)',      'C2'),
        (params.half_angle_twin110  * 2, '(110)',      'C3'),
    ]

    has_volume = volume is not None and needle_volume is not None
    ncols = 3 if has_volume else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5))

    bins120 = np.linspace(0, 120, 61)

    # ── Panel A: 3D misorientation ────────────────────────────────────────────
    print("Computing 3D misorientation...")
    intra, random_pairs = compute_misorientation_3d(center_properties, n_pairs, seed)

    ax = axes[0]
    ax.hist(random_pairs, bins=bins120, density=True, alpha=0.40,
            color='orange', label='Random pairs (inter-grain)')
    ax.hist(intra,        bins=bins120, density=True, alpha=0.65,
            color='steelblue', label='Intra-grain pairs (3D)')
    ymax = ax.get_ylim()[1]
    for angle, label, col in twin_angles:
        ax.axvline(angle, color='firebrick', lw=1.0, ls='--')
        ax.text(angle+0.4, ymax*0.93, label, rotation=90, va='top',
                fontsize=7, color='firebrick')
    ax.set_xlabel('Misorientation angle (°)')
    ax.set_ylabel('Density')
    ax.set_title('3D misorientation\n(note: c-axis spread dilutes peaks)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 120)

    # ── Panel B: 2D slice misorientation (EBSD-like) ─────────────────────────
    if has_volume:
        print("Computing planar slice misorientation (mimicking EBSD)...")
        slice_data = compute_misorientation_slices(
            center_properties, volume, needle_volume, n_pairs, seed
        )
        ax2 = axes[1]
        colors_slices = {'XY': 'steelblue', 'XZ': 'seagreen', 'YZ': 'orchid'}
        for plane, angles in slice_data.items():
            if len(angles) > 0:
                ax2.hist(angles, bins=bins120, density=True, alpha=0.5,
                         color=colors_slices[plane], label=f'{plane} slice')
        ymax2 = ax2.get_ylim()[1]
        for angle, label, col in twin_angles:
            ax2.axvline(angle, color='firebrick', lw=1.0, ls='--')
            ax2.text(angle+0.4, ymax2*0.93, label, rotation=90, va='top',
                     fontsize=7, color='firebrick')
        ax2.set_xlabel('Misorientation angle (°)')
        ax2.set_ylabel('Density')
        ax2.set_title('2D planar slices — adjacent voxels\n(same grain, mimics EBSD scan)')
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 120)
        ref_ax = axes[2]
    else:
        ref_ax = axes[1]

    # ── Reference panel: EBSD peak positions ─────────────────────────────────
    ax3 = ref_ax
    ax3.set_xlim(0, 120)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Misorientation angle (°)')
    ax3.set_title('Expected EBSD peak positions\n(from your Mackenzie plot)')
    ax3.set_yticks([])
    for angle, label, col in twin_angles:
        ax3.axvline(angle, color=col, lw=2.0)
        ax3.text(angle, 0.85, f'{label}\n{angle:.1f}°', ha='center',
                 va='top', fontsize=9, color=col,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    # Sketch the expected peak shape (schematic)
    x = np.linspace(0, 120, 500)
    y = np.zeros_like(x)
    for angle, _, _ in twin_angles:
        sigma = 1.5 if angle > 20 else 5.0
        y += np.exp(-0.5*((x - angle)/sigma)**2)
    y /= y.max()
    ax3.fill_between(x, y*0.6, alpha=0.15, color='steelblue')
    ax3.plot(x, y*0.6, color='steelblue', lw=1.5, label='Schematic')
    ax3.legend(fontsize=8)

    fig.suptitle('Aragonite Orientation Model Verification', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(center_properties: list) -> None:
    phi1_v, Phi_v, phi2_v = [], [], []
    for c in center_properties:
        for n in c['needles']:
            phi1_v.append(n['phi1'])
            Phi_v.append(n['Phi'])
            phi2_v.append(n['phi2'])
    phi1, Phi, phi2 = np.array(phi1_v), np.array(Phi_v), np.array(phi2_v)
    print(f"  phi1: [{phi1.min():.1f}, {phi1.max():.1f}]°  mean={phi1.mean():.1f}°")
    print(f"  Phi:  [{Phi.min():.1f},  {Phi.max():.1f}]°  mean={Phi.mean():.1f}°")
    print(f"  phi2: [{phi2.min():.1f}, {phi2.max():.1f}]°  mean={phi2.mean():.1f}°")
    print(f"  Unique phi2: {len(np.unique(phi2.round(4)))}")


def _make_test_microstructure(n_grains: int, needles_per_grain: int,
                               resolution: int, seed: int):
    """
    Build a minimal synthetic microstructure (center_properties + volume arrays)
    for CLI testing, without requiring the full radial_needles_more_2d().
    """
    rng = np.random.default_rng(seed)
    center_properties = []
    volume      = np.zeros((resolution, resolution, resolution), dtype=int)
    needle_vol  = np.zeros((resolution, resolution, resolution), dtype=int)
    needle_id   = 1

    domain = float(resolution)

    for g in range(n_grains):
        grain_id = g + 1
        # Random centre position
        cx, cy, cz = rng.uniform(5, domain-5, 3)

        phi   = rng.uniform(0, 2*np.pi, needles_per_grain)
        cos_t = rng.uniform(-1, 1, needles_per_grain)
        theta = np.arccos(cos_t)
        dirs  = np.column_stack([
            np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)
        ])

        needles = []
        for d in dirs:
            length = rng.uniform(resolution*0.10, resolution*0.25)
            width  = length / rng.uniform(8, 15)

            # Paint voxels along needle
            t_vals = np.linspace(0, 1, int(length/width*3))
            w_vox  = max(1, int(width))
            for t in t_vals:
                pos = np.array([cx, cy, cz]) + d * t * length
                vi, vj, vk = (np.round(pos)).astype(int)
                for di in range(-w_vox, w_vox+1):
                    for dj in range(-w_vox, w_vox+1):
                        for dk in range(-w_vox, w_vox+1):
                            if di*di+dj*dj+dk*dk <= w_vox*w_vox:
                                ii,jj,kk = vi+di, vj+dj, vk+dk
                                if 0<=ii<resolution and 0<=jj<resolution and 0<=kk<resolution:
                                    volume[ii,jj,kk]     = grain_id
                                    needle_vol[ii,jj,kk] = needle_id

            needles.append({
                'id': needle_id, 'grain_id': grain_id,
                'direction': d, 'length': length, 'width': width,
                'aspect_ratio': length/width,
            })
            needle_id += 1

        center_properties.append({
            'id': grain_id,
            'position': np.array([cx, cy, cz]),
            'needles': needles,
        })

    return center_properties, volume, needle_vol


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group('Twin variant probabilities (must sum to 1.0)')
    g.add_argument('--p-lowangle', type=float, default=0.10)
    g.add_argument('--p-twin110',  type=float, default=0.45)
    g.add_argument('--p-twin310',  type=float, default=0.25)
    g.add_argument('--p-mirror',   type=float, default=0.20)

    g2 = p.add_argument_group('Half-angles for each twin type (degrees)')
    g2.add_argument('--half-lowangle', type=float, default=5.5)
    g2.add_argument('--half-twin110',  type=float, default=31.9)
    g2.add_argument('--half-twin310',  type=float, default=28.6)
    g2.add_argument('--half-mirror',   type=float, default=26.2)

    g3 = p.add_argument_group('Gaussian scatter (degrees)')
    g3.add_argument('--sigma-intra', type=float, default=2.0)
    g3.add_argument('--sigma-twin',  type=float, default=1.5)

    p.add_argument('--plot',       action='store_true')
    p.add_argument('--plot-file',  type=str, default='synthetic_mackenzie.png')
    p.add_argument('--n-pairs',    type=int, default=20000)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--n-grains',   type=int, default=8)
    p.add_argument('--needles-per-grain', type=int, default=20)
    p.add_argument('--resolution', type=int, default=40,
                   help='Voxel resolution for test volume (default: 40)')
    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    params = OrientationParams(
        p_lowangle          = args.p_lowangle,
        p_twin110           = args.p_twin110,
        p_twin310           = args.p_twin310,
        p_mirror            = args.p_mirror,
        half_angle_lowangle = args.half_lowangle,
        half_angle_twin110  = args.half_twin110,
        half_angle_twin310  = args.half_twin310,
        half_angle_mirror   = args.half_mirror,
        sigma_intra         = args.sigma_intra,
        sigma_twin          = args.sigma_twin,
    )
    try:
        params.validate()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ORIENTATION ASSIGNMENT — ARAGONITE TWIN MODEL")
    print("=" * 60)
    print(f"\nTwin half-angles:  low={params.half_angle_lowangle}° → misori {2*params.half_angle_lowangle}°")
    print(f"                   mirror={params.half_angle_mirror}° → misori {2*params.half_angle_mirror}°")
    print(f"                   (310)={params.half_angle_twin310}° → misori {2*params.half_angle_twin310}°")
    print(f"                   (110)={params.half_angle_twin110}° → misori {2*params.half_angle_twin110}°")
    print(f"Probabilities:     low={params.p_lowangle}  mirror={params.p_mirror}  "
          f"(310)={params.p_twin310}  (110)={params.p_twin110}")
    print(f"Scatter:           sigma_intra={params.sigma_intra}°  sigma_twin={params.sigma_twin}°")

    print(f"\nGenerating test microstructure ({args.n_grains} grains × "
          f"{args.needles_per_grain} needles, {args.resolution}³ voxels)...")
    center_props, volume, needle_vol = _make_test_microstructure(
        args.n_grains, args.needles_per_grain, args.resolution, args.seed
    )

    assign_orientations(center_props, params, seed=args.seed)

    if args.plot:
        plot_verification(
            center_props,
            volume=volume,
            needle_volume=needle_vol,
            n_pairs=args.n_pairs,
            seed=args.seed,
            output_file=args.plot_file,
            params=params,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
