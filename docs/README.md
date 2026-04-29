# microcoral-fem-export

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python toolkit for generating synthetic aragonite microstructures and converting
them to MOOSE-ready finite element meshes with automatic cohesive zone interface
classification.

Developed for cold-water coral biomechanics (aragonite), but applicable to other
hierarchical biominerals (bone, nacre) and polycrystalline metals.

---

## Repository structure

```
microcoral-fem-export/
├── src/
│   ├── enhanced_microstructure_export.py     # Volume generation + Exodus export
│   ├── assign_orientations.py                # Aragonite twin orientation model
│   ├── assign_orientations_caxis_aligned.py  # Alternative: c-axis = needle direction
│   ├── convert_to_moose.py                   # Block conversion + interface detection
│   ├── elastic_tensor_converter.py           # 6x6 stiffness → engineering constants
│   └── microstructure_utils.py               # Utility functions (Abaqus path)
│
├── examples/
│   ├── example_coral_rve.py                  # Main MOOSE workflow — start here
│   └── practical_examples.py                 # Abaqus export + cohesive zones
│
├── verification/
│   └── compute_misori_plot.py                # Standalone Mackenzie plot
│
└── docs/
    ├── PIPELINE.md                           # Full pipeline documentation
    ├── SCRIPTS.md                            # All scripts described
    └── COMPLETE_WORKFLOW.md                  # Abaqus workflow (legacy)
```

---

## Quick start

### Install

```bash
git clone https://github.com/ctvmltdznm/microcoral-fem-export.git
cd microcoral-fem-export
pip install -r docs/requirements.txt
```

### Generate a microstructure

Edit parameters at the top of the example, then:

```bash
cd examples
python example_coral_rve.py
```

This produces `aragonite.e` — single-block Exodus with element variables
`grain_id`, `needle_id`, `euler_phi1`, `euler_Phi`, `euler_phi2`.

### Convert for MOOSE

```bash
cd ..
python src/convert_to_moose.py examples/aragonite.e examples/aragonite_moose.e --verify
```

The `--verify` flag produces `aragonite_moose_mackenzie.png` — compare it against
your EBSD Mackenzie plot to confirm the orientation model is correct.

### Check outputs

```
examples/
  aragonite_moose.e                     one block per needle, all element vars preserved
  aragonite_moose_mesh_info.json        mesh bounds + block-to-grain mapping
  aragonite_moose_interface_map.json    every interface typed intra_grain / inter_grain
  aragonite_moose_moose.i              paste-ready [Mesh], [Materials], [BCs]
  aragonite_moose_mackenzie.png         orientation verification (with --verify)
```

### Verify orientation model (optional, before a production run)

```bash
python src/assign_orientations.py --plot --n-grains 100 --needles-per-grain 120 --sigma-twin 0.5
```

Compare `synthetic_mackenzie.png` against your EBSD Mackenzie plot. Twin peaks
should appear at 52.4°, 57.2°, 63.8°.

### Run in MOOSE

Paste `aragonite_moose_moose.i` into your simulation input, then modify
`[GlobalParams]` and `[Materials]` with
your CZM parameters. See `docs/PIPELINE.md` for a complete MOOSE input template.

---

## What the pipeline does

```
example_coral_rve.py
  radial_needles_more_2d()    — parametric voxel RVE
  assign_orientations()        — twin model, integrated into export
  export_to_exodus()           — single-block Exodus with Euler angles
          |
          v
convert_to_moose.py
  organise elements into blocks (one per needle or grain)
  detect face-sharing block pairs → classify intra/inter-grain
  extract mesh bounds for ParsedGenerateSideset
  write mesh_info.json, interface_map.json, _moose.i snippet
  (optional) compute adjacent-element misorientation → Mackenzie plot
```

---

## Orientation model summary

Within a sclerodermite, all needles share one c-axis direction. phi2 (rotation
of a/b axes around c) varies per needle by drawing a twin variant:

| Variant | Misorientation | Default probability |
|---|---|---|
| Low-angle | ~11° | 10% |
| Mirror twin | 52.4° | 20% |
| (310) twin | 57.2° | 25% |
| (110) twin (dominant) | 63.8° | 45% |

Probabilities and scatter widths calibrated from EBSD Mackenzie plot data
(aragonite coral). All parameters are tunable via `OrientationParams`.

An alternative model where c-axis = needle morphological direction is available
in `src/assign_orientations_caxis_aligned.py`. See `docs/SCRIPTS.md` for a
discussion of when each model applies.

---

## Interface classification

Every internal block-pair interface is typed automatically:

| Type | Condition | Physical meaning |
|---|---|---|
| `intra_grain` | same grain_id | Needle-needle within one sclerodermite |
| `inter_grain` | different grain_id | Across sclerodermite boundaries |

The MOOSE snippet assigns different `HomogenizedExponentialCZM` material blocks
to each type. You supply the CZM parameters for each type in `[Materials]`.

---

## Two paths

**MOOSE path (this paper):**
`example_coral_rve.py` → `convert_to_moose.py` → MOOSE

**Abaqus path (also supported):**
`practical_examples.py` → Abaqus INP
See `docs/COMPLETE_WORKFLOW.md`.

---

## Dependencies

```bash
pip install -r docs/requirements.txt
# numpy scipy matplotlib pyvista meshio netCDF4
```

---

## Known limitations

- **Twin model is phi2-only.** Full orthorhombic twin rotations also affect
  phi1/Phi slightly. The phi2-only approximation is justified for needles within
  one sclerodermite (nearly co-axial c-axes).
- **No inter-sclerodermite twin constraint.** Adjacent sclerodermites receive
  independent base orientations.
- **GAN integration not yet complete.** The key missing piece is
  `assign_orientations_from_morphology.py` — PCA-based c-axis fitting for
  segmented volumes (grain_id/needle_id labels) that lack the needle direction
  vectors produced by the parametric generator. Once implemented, the rest of
  the pipeline (`export_to_exodus` + `convert_to_moose.py`) applies unchanged.
- **Small RVE in this paper.** Production-scale interface-resolved simulations
  are noted as future work.

---

## License

MIT — see [LICENSE](LICENSE).
