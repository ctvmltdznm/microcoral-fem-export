#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assign_orientations.py
======================
Physically correct crystallographic orientation assignment for synthetic
aragonite microstructures.

Model: grain-shared c-axis
---------------------------
Within a sclerodermite (grain), all needles share ONE crystallographic
c-axis direction.  This reflects the observation that a sclerodermite is
a single-crystal domain (or a coherently twinned domain), not a
polycrystal where each needle has its own c-axis.

The "flower" morphology seen in SEM/CT — needles radiating in different
directions — is a MORPHOLOGICAL feature, not a crystallographic one.
The needle long axis does NOT necessarily coincide with the c-axis of
the whole grain; rather, one shared c-axis exists for the grain, and the
individual needles grow at various angles to it.

Euler angles (Bunge ZXZ, degrees):
    phi1, Phi  — orientation of the shared c-axis for this sclerodermite
                 drawn once per grain: phi1 ~ U(0,360), Phi ~ arccos(U(-1,1))
    phi2       — rotation of a/b axes around c, varies per needle via
                 the aragonite twin model (±half_angle variants)

This produces twin peaks in adjacent-needle misorientation histograms
because adjacent needles within a grain share the same c-axis — only
phi2 differs — exactly analogous to EBSD adjacent-pixel pairs.

Twin model (from EBSD Mackenzie plot, aragonite coral)
-------------------------------------------------------
Per sclerodermite:  phi1_base, Phi_base  ~ uniform on sphere
                    phi2_base            ~ U(0, 360)

Per needle:         draw variant type (probability-weighted) + sign (±)
                    phi2 = phi2_base ± half_angle + N(0, sigma)

Variant    half_angle   misorientation between ± variants   EBSD peak
---------  ----------   --------------------------------     ---------
Low-angle  5.5°         11°                                  broad
Mirror     26.2°        52.4°
(310)      28.6°        57.2°
(110)      31.9°        63.8°  ← dominant

See also: assign_orientations_caxis_aligned.py — alternative where
phi1/Phi come from each needle's morphological direction vector.

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
    All tunable parameters.  Defaults calibrated from EBSD Mackenzie plot.
    """
    # Twin variant probabilities (must sum to 1.0)
    p_lowangle: float = 0.10
    p_twin110:  float = 0.45
    p_twin310:  float = 0.25
    p_mirror:   float = 0.20

    # Half-angles: phi2 = phi2_base ± half_angle  → misori = 2 * half_angle
    half_angle_lowangle: float = 5.5
    half_angle_twin110:  float = 31.9
    half_angle_twin310:  float = 28.6
    half_angle_mirror:   float = 26.2

    # Gaussian scatter around each variant position (degrees)
    sigma_intra: float = 2.0
    sigma_twin:  float = 1.5

    def validate(self) -> None:
        total = self.p_lowangle + self.p_twin110 + self.p_twin310 + self.p_mirror
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Variant probabilities must sum to 1.0, got {total:.6f}."
            )

    def cumulative_probs(self) -> np.ndarray:
        return np.cumsum([
            self.p_lowangle, self.p_twin110,
            self.p_twin310,  self.p_mirror,
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _random_caxis_euler(rng: np.random.Generator) -> tuple[float, float]:
    """
    Draw a uniformly random c-axis direction on the unit sphere.
    Returns (phi1, Phi) in degrees (Bunge ZXZ convention).
    """
    phi1 = float(rng.uniform(0.0, 360.0))
    Phi  = float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0))))
    return phi1, Phi


def _draw_phi2(phi2_base: float, rng: np.random.Generator,
               params: OrientationParams) -> float:
    """
    Draw phi2 for one needle using the ±half_angle twin model.
    Sign (+/-) is 50/50.  Gaussian scatter applied around the variant.
    """
    cumprobs = params.cumulative_probs()
    u    = rng.uniform(0.0, 1.0)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0

    if u < cumprobs[0]:
        half, sigma = params.half_angle_lowangle, params.sigma_intra
    elif u < cumprobs[1]:
        half, sigma = params.half_angle_twin110,  params.sigma_twin
    elif u < cumprobs[2]:
        half, sigma = params.half_angle_twin310,  params.sigma_twin
    else:
        half, sigma = params.half_angle_mirror,   params.sigma_twin

    return (phi2_base + sign * half + rng.normal(0.0, sigma)) % 360.0


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

    Adds keys 'phi1', 'Phi', 'phi2', 'phi1_base', 'Phi_base', 'phi2_base'
    to each needle dict.

    phi1 and Phi are SHARED across all needles of the same sclerodermite.
    phi2 varies per needle according to the twin model.

    NOTE: needle['direction'] (the morphological growth direction) is NOT
    used for the crystallographic orientation in this model.  It is still
    stored in the dict for mesh geometry purposes.

    Parameters
    ----------
    center_properties : list
        Output of radial_needles_more_2d().
    params : OrientationParams, optional
        Model parameters.  None → EBSD-calibrated defaults.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    center_properties (modified in-place, returned for convenience)
    """
    if params is None:
        params = OrientationParams()
    params.validate()

    rng = np.random.default_rng(seed)

    n_grains  = len(center_properties)
    n_needles = sum(len(c['needles']) for c in center_properties)
    print(f"\nAssigning orientations: {n_needles} needles in "
          f"{n_grains} sclerodermites  [grain-shared c-axis model]")

    for center in center_properties:
        # One shared c-axis direction per sclerodermite
        phi1_base, Phi_base = _random_caxis_euler(rng)
        phi2_base = rng.uniform(0.0, 360.0)

        for needle in center['needles']:
            phi2 = _draw_phi2(phi2_base, rng, params)

            needle['phi1']      = phi1_base    # shared within grain
            needle['Phi']       = Phi_base     # shared within grain
            needle['phi2']      = phi2         # varies per needle (twin)
            needle['phi1_base'] = phi1_base
            needle['Phi_base']  = Phi_base
            needle['phi2_base'] = phi2_base

    _print_summary(center_properties)
    return center_properties


def build_needle_euler_lookup(center_properties: list) -> dict:
    """
    Build {needle_id: {'phi1': float, 'Phi': float, 'phi2': float}}
    after assign_orientations() has been called.
    Drop-in for the old random phi2 logic in export_to_exodus().
    """
    lookup = {}
    for center in center_properties:
        for needle in center['needles']:
            if 'phi1' not in needle:
                raise RuntimeError(
                    f"Needle {needle['id']} missing Euler angles. "
                    "Call assign_orientations() first."
                )
            lookup[needle['id']] = {
                'phi1': needle['phi1'],
                'Phi':  needle['Phi'],
                'phi2': needle['phi2'],
            }
    return lookup


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────

def _build_sym_ops():
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler('ZXZ', [
        [0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]
    ], degrees=True)


def _misori_mmm(rot_i, rot_j, sym_ops) -> float:
    delta = rot_i.inv() * rot_j
    return min(np.degrees((s * delta).magnitude()) for s in sym_ops)


def compute_misorientation_3d(
    center_properties: list,
    n_pairs: int = 30000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute intra-grain vs random-pair misorientation (3D, mmm symmetry).
    Returns (intra_angles, random_angles).

    With the grain-shared c-axis model, intra-grain pairs have identical
    phi1/Phi, so their misorientation is purely a phi2 difference — the
    twin peaks should be clearly visible even in 3D misorientation.
    """
    from scipy.spatial.transform import Rotation
    sym_ops = _build_sym_ops()

    grain_rotations, all_rotations = [], []
    for center in center_properties:
        g = []
        for needle in center['needles']:
            r = Rotation.from_euler(
                'ZXZ', [needle['phi1'], needle['Phi'], needle['phi2']],
                degrees=True
            )
            g.append(r)
            all_rotations.append(r)
        grain_rotations.append(g)

    rng = np.random.default_rng(seed)

    # Intra-grain: same grain, different needle
    intra, attempts = [], 0
    while len(intra) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        g = grain_rotations[rng.integers(0, len(grain_rotations))]
        if len(g) < 2:
            continue
        i, j = rng.choice(len(g), 2, replace=False)
        intra.append(_misori_mmm(g[i], g[j], sym_ops))

    # Random pairs
    n_all = len(all_rotations)
    rand = [_misori_mmm(all_rotations[rng.integers(0, n_all)],
                        all_rotations[rng.integers(0, n_all)], sym_ops)
            for _ in range(n_pairs)]

    return np.array(intra), np.array(rand)


def plot_verification(
    center_properties: list,
    n_pairs: int = 30000,
    seed: int = 0,
    output_file: str = 'synthetic_mackenzie.png',
    params: Optional[OrientationParams] = None,
) -> None:
    """
    Two-panel verification plot.

    Left  — 3D intra-grain misorientation vs random pairs.
             With grain-shared c-axis, intra-grain pairs have identical
             phi1/Phi → misorientation = f(phi2 only) → twin peaks visible.
    Right  — phi2 difference within grain (direct twin model check).
    """
    import matplotlib.pyplot as plt
    if params is None:
        params = OrientationParams()

    twin_peaks = [
        (params.half_angle_lowangle * 2, 'Low-angle\n11°',  'C0'),
        (params.half_angle_mirror   * 2, 'Mirror\n52.4°',   'C1'),
        (params.half_angle_twin310  * 2, '(310)\n57.2°',    'C2'),
        (params.half_angle_twin110  * 2, '(110)\n63.8°',    'C3'),
    ]

    print(f"\nComputing misorientation distribution ({n_pairs} pairs)...")
    intra, rand = compute_misorientation_3d(center_properties, n_pairs, seed)

    # phi2 differences within grains
    phi2_deltas = []
    rng2 = np.random.default_rng(seed + 99)
    for _ in range(n_pairs):
        center = center_properties[rng2.integers(0, len(center_properties))]
        ns = center['needles']
        if len(ns) < 2:
            continue
        i, j = rng2.choice(len(ns), 2, replace=False)
        d = abs(ns[i]['phi2'] - ns[j]['phi2']) % 90.0
        phi2_deltas.append(d)
    phi2_deltas = np.array(phi2_deltas)

    bins     = np.arange(0, 121, 1)
    bins_phi2= np.arange(0, 91, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(rand,  bins=bins, density=True, alpha=0.40, color='orange',
            label=f'Random pairs (n={len(rand):,})')
    ax.hist(intra, bins=bins, density=True, alpha=0.70, color='steelblue',
            label=f'Intra-grain pairs (n={len(intra):,})')
    ymax = ax.get_ylim()[1]
    for ang, lbl, col in twin_peaks:
        ax.axvline(ang, color='firebrick', lw=1.2, ls='--')
        ax.text(ang + 0.4, ymax * 0.93, lbl, rotation=90,
                va='top', fontsize=8, color='firebrick')
    ax.set_xlabel('Misorientation angle (deg)')
    ax.set_ylabel('Density')
    ax.set_title('3D misorientation — intra-grain vs random\n'
                 '(grain-shared c-axis: peaks should be visible)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 120)

    ax2 = axes[1]
    ax2.hist(phi2_deltas, bins=bins_phi2, density=True, alpha=0.75,
             color='steelblue', label=f'|Δphi2| mod 90°  (n={len(phi2_deltas):,})')
    ymax2 = ax2.get_ylim()[1]
    for ang, lbl, col in twin_peaks:
        a2 = ang % 90.0
        if a2 > 45:
            a2 = 90 - a2
        ax2.axvline(a2, color='firebrick', lw=1.2, ls='--')
        ax2.text(a2 + 0.4, ymax2 * 0.93, lbl.split('\n')[0], rotation=90,
                 va='top', fontsize=8, color='firebrick')
    ax2.set_xlabel('|phi2_A − phi2_B| mod 90° (deg)')
    ax2.set_ylabel('Density')
    ax2.set_title('phi2 difference within sclerodermite\n(direct twin model check)')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 90)

    fig.suptitle('Synthetic Aragonite — Orientation Verification\n'
                 '[grain-shared c-axis model]',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(center_properties: list) -> None:
    phi1_v, Phi_v, phi2_v = [], [], []
    for c in center_properties:
        for n in c['needles']:
            phi1_v.append(n['phi1'])
            Phi_v.append(n['Phi'])
            phi2_v.append(n['phi2'])
    phi1 = np.array(phi1_v)
    Phi  = np.array(Phi_v)
    phi2 = np.array(phi2_v)
    # phi1/Phi vary by grain, not by needle; count unique grain values
    unique_phi1 = len(np.unique(phi1.round(4)))
    n_grains    = len(center_properties)
    print(f"  phi1: [{phi1.min():.1f}, {phi1.max():.1f}]°  "
          f"({unique_phi1} unique = {n_grains} grains, shared per grain)")
    print(f"  Phi:  [{Phi.min():.1f},  {Phi.max():.1f}]°")
    print(f"  phi2: [{phi2.min():.1f}, {phi2.max():.1f}]°  "
          f"({len(np.unique(phi2.round(4)))} unique values across needles)")


def _make_test_center_properties(n_grains, needles_per_grain, seed):
    rng = np.random.default_rng(seed)
    center_properties = []
    needle_id = 1
    for g in range(n_grains):
        needles = []
        phi   = rng.uniform(0, 2 * np.pi, needles_per_grain)
        cos_t = rng.uniform(-1, 1, needles_per_grain)
        theta = np.arccos(cos_t)
        for i in range(needles_per_grain):
            d = np.array([np.sin(theta[i]) * np.cos(phi[i]),
                          np.sin(theta[i]) * np.sin(phi[i]),
                          np.cos(theta[i])])
            needles.append({
                'id': needle_id, 'grain_id': g + 1,
                'direction': d,
                'length': rng.uniform(10, 30),
                'width':  rng.uniform(1, 3),
                'aspect_ratio': rng.uniform(8, 15),
            })
            needle_id += 1
        center_properties.append({
            'id': g + 1,
            'position': rng.uniform(0, 100, 3),
            'needles': needles,
        })
    return center_properties


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

    g2 = p.add_argument_group('Half-angles (degrees)')
    g2.add_argument('--half-lowangle', type=float, default=5.5)
    g2.add_argument('--half-twin110',  type=float, default=31.9)
    g2.add_argument('--half-twin310',  type=float, default=28.6)
    g2.add_argument('--half-mirror',   type=float, default=26.2)

    g3 = p.add_argument_group('Gaussian scatter (degrees)')
    g3.add_argument('--sigma-intra', type=float, default=2.0)
    g3.add_argument('--sigma-twin',  type=float, default=1.5)

    p.add_argument('--plot',              action='store_true')
    p.add_argument('--plot-file',         type=str, default='synthetic_mackenzie.png')
    p.add_argument('--n-pairs',           type=int, default=30000)
    p.add_argument('--seed',              type=int, default=42)
    p.add_argument('--n-grains',          type=int, default=12)
    p.add_argument('--needles-per-grain', type=int, default=20)
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

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
    print("ORIENTATION ASSIGNMENT — grain-shared c-axis model")
    print("=" * 60)
    print(f"Half-angles: low={params.half_angle_lowangle}°  "
          f"mirror={params.half_angle_mirror}°  "
          f"(310)={params.half_angle_twin310}°  "
          f"(110)={params.half_angle_twin110}°")
    print(f"Scatter:     sigma_intra={params.sigma_intra}°  "
          f"sigma_twin={params.sigma_twin}°")

    cp = _make_test_center_properties(args.n_grains, args.needles_per_grain, args.seed)
    assign_orientations(cp, params, seed=args.seed)

    if args.plot:
        plot_verification(cp, n_pairs=args.n_pairs, seed=args.seed,
                          output_file=args.plot_file, params=params)
    print("\nDone.")


if __name__ == '__main__':
    main()