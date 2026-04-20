# Script Reference

Descriptions of all scripts in `src/`, `examples/`, and `verification/`.

---

## src/

### `enhanced_microstructure_export.py`

**Role:** Volume generation and Exodus export. The core of the pipeline.

**History:** Original script. Initially exported Abaqus INP and VTK only.
Exodus export (`export_to_exodus`) was added later; it now integrates
orientation assignment via `assign_orientations.py` and writes Euler angles
as element variables directly. The patched version of `export_to_exodus()`
replaced the earlier random phi2 assignment with the physically correct twin
model.

**Key functions:**

`radial_needles_more_2d(num_centers, domain_size, needle_length_range,
needles_per_center_range, resolution, z_constraint_factor, quasi_2d)`
- Generates voxelised RVE with radially-growing needle structures
- Returns `volume` (grain_id per voxel), `needle_volume` (needle_id per voxel),
  `center_properties` (needle directions, lengths, aspect ratios)
- `quasi_2d=True`: nucleation centres distributed in layers, needles fan in XY
- `quasi_2d=False`: fully isotropic 3D structure

`export_to_exodus(volume, needle_volume, domain_size, filename,
center_properties, orientation_params, orientation_seed)`
- Calls `assign_orientations()` internally before writing
- Writes single-block Exodus with element variables:
  `grain_id`, `needle_id`, `euler_phi1`, `euler_Phi`, `euler_phi2`
- `orientation_params=None` uses EBSD-calibrated defaults

`export_to_abaqus_enhanced(...)` — Abaqus INP with orientations and BCs

`export_vtk_unstructured(...)` — VTK with orientation data for ParaView

---

### `assign_orientations.py`

**Role:** Crystallographic orientation assignment using the aragonite twin model.
Integrated into `export_to_exodus()` but also usable standalone for verification.

**Model — grain-shared c-axis:**
All needles within a sclerodermite share one c-axis direction (phi1, Phi).
phi2 varies per needle by selecting a twin variant:

```
phi2 = phi2_base ± half_angle + N(0, sigma)
```

| Variant | half_angle | Misorientation | Peak in EBSD |
|---|---|---|---|
| Low-angle | 5.5° | 11° | broad |
| Mirror | 26.2° | 52.4° | |
| (310) | 28.6° | 57.2° | |
| (110) | 31.9° | 63.8° | dominant |

**Key classes/functions:**

`OrientationParams` — dataclass with all tunable parameters (probabilities,
half-angles, scatter sigmas). All defaults calibrated from EBSD data.

`assign_orientations(center_properties, params, seed)` — assigns phi1, Phi,
phi2 to every needle dict in-place.

`build_needle_euler_lookup(center_properties)` — returns
`{needle_id: {phi1, Phi, phi2}}` for use in `export_to_exodus`.

`plot_verification(center_properties, ...)` — produces synthetic Mackenzie
plot comparing intra-grain vs random-pair misorientation.

**CLI:**
```bash
python src/assign_orientations.py --plot --n-grains 100 --needles-per-grain 120 \
    --sigma-twin 0.5 --seed 42
```
See `docs/PIPELINE.md` for full CLI reference.

---

### `assign_orientations_caxis_aligned.py`

**Role:** Alternative orientation model where phi1/Phi come from each needle's
morphological direction vector (c-axis = needle long axis).

**When to use:** This model is physically appropriate if the sclerodermite is a
polycrystal where each needle has an independent c-axis, rather than a single
coherent crystal domain. In the adjacent-element misorientation histogram, the
c-axis angular spread between neighbouring needles (typically 30–60°) dominates
the full 3D misorientation, burying the phi2 twin signal. The grain-shared model
(`assign_orientations.py`) reproduces the EBSD Mackenzie peaks more cleanly.

Both models are preserved because the correct physics depends on the actual coral
structure, which is still an open question pending full 3D EBSD reconstruction.

**API:** Identical to `assign_orientations.py` — same function signatures,
same `OrientationParams`, same CLI flags.

---

### `convert_to_moose.py`

**Role:** Converts a single-block Exodus file (from `export_to_exodus`) into a
fully MOOSE-ready set of outputs. This is the Task 1 script.

**History:** Created to replace the manual workflow that used `convert2e.py`
plus handwritten `ParsedGenerateSideset` blocks. Automates: block reorganisation,
interface detection and classification, mesh bound extraction, MOOSE snippet
generation, and orientation verification.

**What it does:**
1. Reads single-block Exodus with element variables
2. Splits into one element block per needle_id (or grain_id)
3. Detects all internal face-sharing block pairs (hex face adjacency)
4. Classifies each interface as `intra_grain` or `inter_grain` using `grain_id`
5. Computes mesh bounds for `ParsedGenerateSideset` coordinates
6. Writes multi-block Exodus, `mesh_info.json`, `interface_map.json`,
   `_moose.i` snippet
7. Optionally generates Mackenzie misorientation plot (`--verify`)

**CLI:**
```bash
python src/convert_to_moose.py input.e output.e [--organize-by needle_id|grain_id]
    [--czm-intra NAME] [--czm-inter NAME] [--verify] [--force]
```

---

### `convert2e.py`

**Role:** Converts VTK files (with orientation data) to Exodus format.

**History:** This was the original MOOSE conversion tool, created when the
pipeline was: generate → VTK → convert2e.py → patch manually. It reorganises
elements by grain_id or needle_id into separate Exodus blocks, and computes
Euler angles from c/a/b axis vectors stored in the VTK.

**Current relevance:**
- Superseded by `convert_to_moose.py` for the parametric generation path
- Still needed for the GAN path: GAN-generated volumes will likely be produced
  as VTK or numpy arrays, and `convert2e.py` handles the VTK → Exodus step
- Useful for re-converting existing `.vtk` files from before the new pipeline

**CLI:**
```bash
python src/convert2e.py microstructure.vtk -o mesh.e --organize-by needle_id
```

**Required VTK fields:** `c_axis_x/y/z`, `a_axis_x/y/z`, `b_axis_x/y/z`,
`grain_id`, `needle_id`.

---

### `elastic_tensor_converter.py`

**Role:** Converts 6×6 stiffness matrices (from MD/DFT) to engineering constants
needed by MOOSE (`ComputeElasticityTensorCoupled`) or Abaqus.

**Key functions:**
- `stiffness_to_compliance(C)` — C → S matrix (inversion)
- `compliance_to_engineering_constants(S)` — extract E1/E2/E3, nu12/13/23, G12/13/23
- `validate_orthotropic_symmetry(constants)` — check nu_ij/E_i = nu_ji/E_j
- `format_for_abaqus(constants, unit='MPa')` — ready-to-paste string

**Standalone use:**
```bash
python src/elastic_tensor_converter.py
# Runs aragonite example by default and prints constants
```

---

### `microstructure_utils.py`

**Role:** Utility functions for the Abaqus workflow. Not required for the MOOSE
pipeline.

**Functions:**
- `generate_random_texture(num_grains, texture_type)` — random / cube / Goss
- `euler_to_rotation_matrix(phi1, Phi, phi2)` — Bunge ZXZ → 3×3 matrix
- `find_grain_boundary_faces(volume)` — detect grain boundary faces in voxel array
- `compute_misorientation(R1, R2)` — misorientation angle between two rotation matrices
- `assign_cohesive_properties(angle)` — misorientation-based CZM properties (Abaqus)
- `create_abaqus_cohesive_section(filename, faces, orientations)` — write Abaqus CZM section
- `analyze_microstructure_stats(volume, needle_volume)` — grain/needle size statistics
- `export_orientation_data(center_properties, filename)` — export CSV of needle directions

---

## examples/

### `example_coral_rve.py`

**Role:** Main example for the MOOSE pipeline. Edit parameters at the top and run.
Produces `aragonite.e` ready for `convert_to_moose.py`.

**Usage:**
```bash
cd examples
python example_coral_rve.py
# then:
cd ..
python src/convert_to_moose.py examples/aragonite.e examples/aragonite_moose.e --verify
```

### `practical_examples.py`

**Role:** Demonstrates the Abaqus export path. Three examples:
1. Aragonite coral — quasi-2D structure, orthotropic properties, Exodus export
2. BCC polycrystal — 3D structure, random texture, Abaqus export
3. Grain boundaries with cohesive zones — boundary detection, Abaqus CZM section

Uses `microstructure_utils.py`. Not required for the MOOSE pipeline.

### `aragonite_constants.txt`

Pre-computed engineering constants for aragonite (from `elastic_tensor_converter.py`).
Copy values into `example_coral_rve.py` or MOOSE input as needed.

### Output files (`tiny_aragonite.*`)

Example outputs from `practical_examples.py` (example 1, small test run).
Included for reference; not tracked in production runs.

---

## verification/

### `compute_misori_plot.py`

**Role:** Standalone script that generates an exhaustive adjacent-voxel
misorientation histogram from a filled microstructure volume. Mimics EBSD
pixel-pair analysis without needing an Exodus file.

Produces two panels:
- Left: all adjacent cross-needle same-grain pairs (full volume)
- Right: zoomed 0–90°, 0.5° bins

**Usage:** Edit `RESOLUTION`, `N_GRAINS`, `NEEDLES_GRAIN` at the top and run:
```bash
python verification/compute_misori_plot.py
# → adjacent_misori.png
```

Useful for tuning `sigma_twin` and checking that twin peaks appear at the correct
angles before committing to a full RVE run.
