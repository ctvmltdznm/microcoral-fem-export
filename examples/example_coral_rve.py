#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_coral_rve.py
====================
Main example for the MOOSE pipeline.

Edit the parameters below and run:

    cd examples
    python example_coral_rve.py

Then convert for MOOSE:

    cd ..
    python src/convert_to_moose.py examples/aragonite.e examples/aragonite_moose.e --verify
"""

import os
import sys

# Allow running from the examples/ directory without installing the package.
# Adds the src/ folder to the Python path so imports work regardless of
# where the script is called from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_microstructure_export import (
    radial_needles_more_2d,
    export_to_exodus,
)
from assign_orientations import OrientationParams


# ── Parameters ────────────────────────────────────────────────────────────────

# Geometry
NUM_CENTERS              = 45         # sclerodermites (grains)
DOMAIN_SIZE              = 10.0       # physical size in µm
NEEDLE_LENGTH_RANGE      = (1.0, 4.0) # µm
NEEDLES_PER_CENTER_RANGE = (10, 35)
RESOLUTION               = 100        # voxels per side; 100^3 = 1M elements

# Structure type
QUASI_2D             = True   # True: layered (matches EBSD cross-sections)
                               # False: isotropic 3D
Z_CONSTRAINT_FACTOR  = 0.1    # spread of needles in Z; 0 = fully in-plane

# Orientation model — twin parameters calibrated from EBSD Mackenzie plot
# All probabilities must sum to 1.0
ORIENTATION_PARAMS = OrientationParams(
    p_lowangle = 0.10,   # broad peak ~11°
    p_twin110  = 0.45,   # dominant peak at 63.8°  (110 twin)
    p_twin310  = 0.25,   # peak at 57.2°            (310 twin)
    p_mirror   = 0.20,   # peak at 52.4°            (mirror of 110)

    # half_angle: phi2 = phi2_base ± half_angle  →  misorientation = 2 × half_angle
    half_angle_lowangle = 5.5,
    half_angle_twin110  = 31.9,
    half_angle_twin310  = 28.6,
    half_angle_mirror   = 26.2,

    # Gaussian scatter around each twin variant (degrees)
    # Smaller sigma → sharper, better-separated peaks in Mackenzie plot
    sigma_intra = 2.0,
    sigma_twin  = 1.0,
)

ORIENTATION_SEED = 42       # set to None for stochastic runs
OUTPUT_FILE      = 'aragonite.e'


# ── Generate ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("GENERATING ARAGONITE MICROSTRUCTURE")
print("=" * 60)
print(f"  Sclerodermites:    {NUM_CENTERS}")
print(f"  Domain size:       {DOMAIN_SIZE} µm")
print(f"  Resolution:        {RESOLUTION}^3 = {RESOLUTION**3:,} voxels")
print(f"  Quasi-2D:          {QUASI_2D}")

volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers              = NUM_CENTERS,
    domain_size              = DOMAIN_SIZE,
    needle_length_range      = NEEDLE_LENGTH_RANGE,
    needles_per_center_range = NEEDLES_PER_CENTER_RANGE,
    resolution               = RESOLUTION,
    z_constraint_factor      = Z_CONSTRAINT_FACTOR,
    quasi_2d                 = QUASI_2D,
)

n_grains  = len(center_properties)
n_needles = sum(len(c['needles']) for c in center_properties)
print(f"\nGenerated: {n_grains} grains, {n_needles} needles")


# ── Export ────────────────────────────────────────────────────────────────────

print(f"\nExporting to {OUTPUT_FILE} ...")
print("  (Euler angles will be assigned via twin model)")

export_to_exodus(
    volume, needle_volume,
    domain_size        = DOMAIN_SIZE,
    filename           = OUTPUT_FILE,
    center_properties  = center_properties,
    orientation_params = ORIENTATION_PARAMS,
    orientation_seed   = ORIENTATION_SEED,
)

print(f"\nDone. Next step:")
print(f"  cd ..")
print(f"  python src/convert_to_moose.py examples/{OUTPUT_FILE} "
      f"examples/aragonite_moose.e --verify")
