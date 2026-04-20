from enhanced_microstructure_export import (
    radial_needles_more_2d,
    export_to_exodus,
)
from assign_orientations_caxis_aligned import OrientationParams

# 1. Generate microstructure
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers            = 45,          # sclerodermites (grains)
    domain_size            = 10.0,        # physical size (µm)
    needle_length_range    = (1.0, 4.0),  # µm
    needles_per_center_range = (10, 35),
    resolution             = 100,         # voxels per side
    z_constraint_factor    = 0.1,         # quasi-2D (small z spread)
    quasi_2d               = True,
)

# 2+3. Export with physically correct orientations
params = OrientationParams(
    # Twin probabilities (must sum to 1.0)
    p_lowangle = 0.10,   # low-angle GB / intra-grain scatter (~11°)
    p_twin110  = 0.45,   # (110) twin — dominant peak at 63.8°
    p_twin310  = 0.25,   # (310) twin at 57.2°
    p_mirror   = 0.20,   # mirror twin at 52.4°

    # Half-angles: phi2 = phi2_base ± half_angle
    half_angle_lowangle = 5.5,    # → misorientation ~11°
    half_angle_twin110  = 31.9,   # → misorientation 63.8°
    half_angle_twin310  = 28.6,   # → misorientation 57.2°
    half_angle_mirror   = 26.2,   # → misorientation 52.4°

    # Gaussian scatter around each variant position
    sigma_intra = 1.0,   # scatter for low-angle peak
    sigma_twin  = 0.1,   # scatter for twin peaks (tighter = sharper peaks)
)

export_to_exodus(
    volume, needle_volume, domain_size=10.0,
    filename='aragonite.e',
    center_properties=center_properties,
    orientation_params=params,
    orientation_seed=42,          # set for reproducibility
)