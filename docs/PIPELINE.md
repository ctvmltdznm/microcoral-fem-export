# Pipeline Documentation

End-to-end guide for generating synthetic aragonite microstructures and
converting them to MOOSE-ready meshes with cohesive zone interface
classification. Also covers the legacy Abaqus path and the planned GAN path.

---

## Overview

Three paths exist. The **MOOSE path** is the primary one for this paper.

```
                        ┌── MOOSE path (primary) ───────────────────────────────┐
                        │                                                       │
MD/DFT properties       │  examples/example_coral_rve.py                        │
       │                │    radial_needles_more_2d()  → voxel RVE              │
       ▼                │    assign_orientations()     → twin model (integrated)│
elastic_tensor_         │    export_to_exodus()        → single-block .e        │
converter.py            │                  │                                    │
       │                │                  ▼                                    │
       ▼                │  src/convert_to_moose.py                              │
Engineering             │    → multi-block .e   (1 block per needle or grain)   │
constants               │    → mesh_info.json   (bounds + block-to-grain map)   │
                        │    → interface_map.json (all interfaces typed)        │
                        │    → _moose.i snippet (Mesh + InterfaceKernels + BCs) │
                        └───────────────────────────────────────────────────────┘

                        ┌── Abaqus / legacy path ───────────────────────────────┐
                        │                                                       │
                        │  examples/practical_examples.py                       │
                        │    radial_needles_more_2d()  → voxel RVE              │
                        │    export_to_abaqus_enhanced() → .inp with BCs        │
                        │    export_vtk_unstructured() → .vtk for ParaView      │
                        │    [optional] convert2e.py   → .e from VTK            │
                        └───────────────────────────────────────────────────────┘

                        ┌── GAN path (future) ──────────────────────────────────┐
                        │                                                       │
                        │  GAN-generated segmented volume  →  src/convert2e.py  │
                        │    [pending] orientation assignment from morphology   │
                        │    same convert_to_moose.py step as MOOSE path        │
                        └───────────────────────────────────────────────────────┘
```

---

## MOOSE path — step by step

### Step 1: Generate microstructure and export Exodus

Edit `examples/example_coral_rve.py` with your parameters and run:

```bash
cd examples
python example_coral_rve.py
```

Key parameters:

```python
# Geometry
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers              = 45,         # sclerodermites
    domain_size              = 10.0,       # physical size in µm
    needle_length_range      = (1.0, 4.0), # µm
    needles_per_center_range = (10, 35),
    resolution               = 100,        # voxels per side; 100^3 = 1M elements
    quasi_2d                 = True,       # True: layered; False: isotropic 3D
    z_constraint_factor      = 0.1,        # z-spread of needles (0 = fully in-plane)
)

# Orientations — twin model calibrated from EBSD Mackenzie plot
params = OrientationParams(
    p_lowangle = 0.10,   # broad peak ~11°
    p_twin110  = 0.45,   # dominant peak at 63.8°  (110 twin)
    p_twin310  = 0.25,   # peak at 57.2°            (310 twin)
    p_mirror   = 0.20,   # peak at 52.4°            (mirror of 110)
    sigma_twin = 1.0,    # scatter around peaks; reduce to sharpen
)

export_to_exodus(
    volume, needle_volume, domain_size=10.0,
    filename='aragonite.e',
    center_properties=center_properties,
    orientation_params=params,
    orientation_seed=42,
)
```

Output `aragonite.e` contains element variables:
`grain_id`, `needle_id`, `euler_phi1`, `euler_Phi`, `euler_phi2`

phi1/Phi are shared within each sclerodermite (one c-axis per grain).
phi2 varies per needle according to the twin model.

### Step 2: Convert for MOOSE

```bash
cd ..
python src/convert_to_moose.py examples/aragonite.e examples/aragonite_moose.e --verify
```

The `--verify` flag generates `aragonite_moose_mackenzie.png` — compare it
against your EBSD Mackenzie plot to confirm the orientation model is correct.

Other useful flags:

```bash
# Grain-level blocks instead of needle-level (fewer interfaces, faster solve)
python src/convert_to_moose.py aragonite.e aragonite_moose.e --organize-by grain_id

# Custom CZM material names
python src/convert_to_moose.py aragonite.e aragonite_moose.e \
    --czm-intra CZM_Protein --czm-inter CZM_Water
```

Outputs:

| File | Content |
|---|---|
| `aragonite_moose.e` | Multi-block Exodus; one block per needle (or grain) |
| `aragonite_moose_mesh_info.json` | Mesh bounds + block-to-grain mapping |
| `aragonite_moose_interface_map.json` | All interfaces typed intra/inter-grain |
| `aragonite_moose_moose.i` | Paste-ready MOOSE snippet |
| `aragonite_moose_mackenzie.png` | Orientation verification (`--verify` only) |

### Step 3: Run in MOOSE

Paste `_moose.i` into your simulation input. Add `[Materials]`:

```ini
[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Modules/TensorMechanics/Master]
  [all]
    strain = FINITE
    add_variables = true
  []
[]

[Materials]
  [elasticity]
    type = ComputeElasticityTensorCoupled   # from crystal_ort repo
    euler_angle_1 = euler_phi1
    euler_angle_2 = euler_Phi
    euler_angle_3 = euler_phi2
    ...
  []
  [czm_intra]
    type = HomogenizedExponentialCZMMaterial
    base_name = 'CZM_IntraGrain'
    boundary = '...'   # list all intra-grain interface names from interface_map.json
    # Parameters from Kvashin et al. 2026 (needle-needle, organic matrix)
  []
  [czm_inter]
    type = HomogenizedExponentialCZMMaterial
    base_name = 'CZM_InterGrain'
    boundary = '...'
    # Parameters for inter-sclerodermite boundary (water/protein layer)
  []
[]
```

---

## Orientation model details

### Physical basis

Aragonite is orthorhombic (mmm symmetry). Within a sclerodermite, all needles
share one crystallographic c-axis direction — the sclerodermite is a single
coherent crystal domain, not a polycrystal. The flower-like morphology seen
in SEM/CT is a morphological feature, not a crystallographic one.

The model assigns:
- One `(phi1_base, Phi_base)` per sclerodermite — uniform random on sphere
- One `phi2_base` per sclerodermite — uniform 0–360°
- Per needle: `phi2 = phi2_base ± half_angle + N(0, sigma)`

| Variant | half_angle | Misorientation | EBSD peak |
|---|---|---|---|
| Low-angle | 5.5° | ~11° | broad |
| Mirror twin | 26.2° | 52.4° | |
| (310) twin | 28.6° | 57.2° | |
| (110) twin | 31.9° | 63.8° | dominant |

### Alternative model

`src/assign_orientations_caxis_aligned.py` implements an earlier model where
phi1/Phi come from each needle's morphological direction vector (c-axis = needle
long axis). This is physically appropriate if the sclerodermite is a polycrystal
with each needle having its own independent c-axis. See the file header for a
discussion of when each model applies.

### Verification

```bash
# Quick standalone check — synthetic Mackenzie plot
python src/assign_orientations.py \
    --plot --n-grains 100 --needles-per-grain 120 --sigma-twin 0.5

# Exhaustive adjacent-voxel scan on a filled volume
python verification/compute_misori_plot.py

# Verification built into conversion step
python src/convert_to_moose.py aragonite.e aragonite_moose.e --verify
```

---

## Interface classification

`convert_to_moose.py` reads `grain_id` from element variables to type each
block-pair interface:

- **`intra_grain`** — both blocks have the same `grain_id`
  → needle-needle within one sclerodermite → organic matrix / protein bonding
- **`inter_grain`** — blocks have different `grain_id`
  → across sclerodermite boundaries → water layer / protein

`interface_map.json`:
```json
{
  "block3_block7": {
    "type": "inter_grain",
    "grain_A": 1,
    "grain_B": 2,
    "n_faces": 142
  },
  "block3_block4": {
    "type": "intra_grain",
    "grain_A": 1,
    "grain_B": 1,
    "n_faces": 88
  }
}
```

---

## Legacy path — Abaqus / old MOOSE workflow

Before the current pipeline existed, two separate intermediate steps were used
to prepare files for MOOSE:

**Old workflow (three steps):**
```
enhanced_microstructure_export.py  →  .vtk  (with c/a/b axis vectors per element)
        │
        convert2e.py  →  single-block .e  (computed Euler angles from axis vectors)
        │
        assign_exodus_blocks.py  →  multi-block .e  (reorganised by grain/needle ID)
        │
        [manual]  →  MOOSE input  (ParsedGenerateSideset written by hand)
```

Both `convert2e.py` and `assign_exodus_blocks.py` are now deleted — their
functionality is fully covered by `export_to_exodus()` (with integrated
orientation assignment) and `convert_to_moose.py` (block conversion + interface
detection + MOOSE snippet generation).

For the **Abaqus export path** (generating `.inp` files), see
`examples/practical_examples.py` and `docs/COMPLETE_WORKFLOW.md`.
This path uses `export_to_abaqus_enhanced()` and `microstructure_utils.py`
and remains fully functional.

---

## GAN path — future work

The planned workflow for GAN-generated microstructures:

```
GAN-generated volume
  → segmented labels (grain_id / needle_id per voxel, as numpy array or VTK)
  → [pending] assign_orientations_from_morphology.py
      PCA on voxel coordinates per needle → fit c-axis direction
      same twin model as assign_orientations.py
  → export_to_exodus() → single-block .e with Euler angles
  → convert_to_moose.py → MOOSE-ready outputs (same as primary path)
```

**What is already in place:**
- `src/assign_orientations.py` — twin model works independently of how the
  morphology was generated; it only needs `center_properties` with needle
  direction vectors, which the morphology fitting step would provide
- `src/convert_to_moose.py` — works on any single-block Exodus, regardless
  of how it was generated

**What is not yet implemented:**
- `assign_orientations_from_morphology.py` — PCA-based c-axis fitting for
  segmented volumes that lack `center_properties` (no needle direction vectors
  from the parametric generator); this is the key missing piece for the GAN path
- GAN training / inference code
- 2D-to-3D reconstruction from experimental EBSD

**What we expect from GAN collaborators:**
- Segmented 3D volume as numpy array or VTK with `grain_id` and `needle_id`
  labels per voxel — no pre-computed orientation data required
- The orientation assignment step handles crystallography separately from
  morphology, so the GAN only needs to get the geometry right

**Note on `convert2e.py` (removed):** An earlier script `convert2e.py` converted
VTK files that contained full rotation matrix data (c/a/b axis vectors per element)
to Exodus. It required pre-computed orientation data in the VTK and is now
superseded: the parametric path uses `export_to_exodus()` directly, and the GAN
path will use `assign_orientations_from_morphology.py` on raw segmented labels.

---

## Mesh sizes and performance

| RVE | Grains | Needles | Elements | `convert_to_moose` | MOOSE runtime |
|---|---|---|---|---|---|
| Small (paper) | 10–20 | 100–300 | ~100k | seconds | minutes |
| Medium | 45 | ~1000 | ~1M | ~1 min | hours |
| Large | 100+ | ~5000 | ~10M | ~10 min | days |

For large meshes: `--organize-by grain_id` reduces block count and interface
kernel count significantly. CZM solve is the dominant cost in MOOSE.

---

## CLI quick reference

### `src/assign_orientations.py`
```
--p-lowangle / --p-twin110 / --p-twin310 / --p-mirror  FLOAT  (probabilities, sum=1)
--half-lowangle / --half-twin110 / --half-twin310 / --half-mirror  FLOAT  (degrees)
--sigma-intra / --sigma-twin  FLOAT  (scatter in degrees)
--n-grains / --needles-per-grain / --resolution / --seed  INT
--plot / --plot-file / --n-pairs
```

### `src/convert_to_moose.py`
```
input.e output.e
--organize-by  needle_id | grain_id        [needle_id]
--czm-intra    NAME                        [CZM_IntraGrain]
--czm-inter    NAME                        [CZM_InterGrain]
--verify       generate _mackenzie.png
--force        overwrite without prompt
--no-moose-snippet
```
