> **Note:** This document describes the **Abaqus export workflow** — generating
> INP files with material orientations and boundary conditions for use in Abaqus,
> plus VTK export for ParaView visualisation.
>
> For the **MOOSE workflow** (multi-block Exodus + CZM interface classification),
> see [`PIPELINE.md`](PIPELINE.md) instead.
> 
---

# Complete Workflow Guide

This guide explains the full MD→FEM workflow for the **Abaqus path**,
using `enhanced_microstructure_export.py`, `microstructure_utils.py`, and
`elastic_tensor_converter.py`.

Note: `convert2e.py` was previously part of this workflow (VTK → Exodus
conversion for MOOSE). It has been removed and its functionality is now
handled by `convert_to_moose.py`, which additionally provides interface
detection and automatic MOOSE snippet generation. See `PIPELINE.md`.

---

## Overview

### The Abaqus Pipeline

```
MD/DFT Simulations and/or Experiments
       ↓
[elastic_tensor_converter.py]  ← Convert 6×6 matrix to E, ν, G
       ↓
Engineering Constants
       ↓
[enhanced_microstructure_export.py]  ← Generate + export with BCs
       ↓
Abaqus INP + VTK files
       ↓
Run simulations in Abaqus
  (or use PIPELINE.md for MOOSE)
```

### Three Core Scripts (Abaqus path)

1. **`elastic_tensor_converter.py`** - Stiffness to engineering constants
2. **`enhanced_microstructure_export.py`** - Core microstructure generation + Abaqus/VTK export
3. **`microstructure_utils.py`** - Advanced operations (texture, GBs, cohesive zones)

Each script is modular and can be used independently or as part of the complete workflow.

---

## The Three Core Scripts

### 1. `elastic_tensor_converter.py`

**Purpose:** Convert elastic tensors from atomistic simulations to FEM-ready engineering constants.

**Input:** 6×6 stiffness matrix C (GPa)
**Output:** E₁, E₂, E₃, ν₁₂, ν₁₃, ν₂₃, G₁₂, G₁₃, G₂₃

**Key Functions:**
```python
stiffness_to_compliance(C)                  # C → S matrix
compliance_to_engineering_constants(S)      # Extract E, ν, G
validate_orthotropic_symmetry(constants)    # Check validity
format_for_abaqus(constants, unit='MPa')   # Ready-to-paste
```

**When to use:**
- You have MD simulation results
- You have DFT calculations
- You have experimental stiffness data
- You need to convert between representations

### 2. `enhanced_microstructure_export.py`

**Purpose:** Generate synthetic microstructures and export to Abaqus INP, VTK, or Exodus II
with automatic orientation and boundary condition assignment.

**Input:** Microstructure parameters (size, grains, resolution)
**Output:** Abaqus INP, VTK, Exodus II files

**Key Functions:**
```python
radial_needles_more_2d(...)                # Generate microstructure
export_to_abaqus_enhanced(...)             # Export with BCs
export_vtk_unstructured(...)               # VTK with orientations
export_to_exodus(...)                      # Direct Exodus export (MOOSE path)
```

**When to use:**
- Generate representative volume elements (RVEs)
- Create synthetic microstructures
- Need automatic material orientation assignment
- Want ready-to-run FEM models

### 3. `microstructure_utils.py`

**Purpose:** Advanced operations on microstructures (texture, grain boundaries, cohesive zones).

**Input:** Microstructure arrays and parameters
**Output:** Orientations, grain boundaries, cohesive properties

**Key Functions:**
```python
generate_random_texture(...)               # Crystallographic texture
euler_to_rotation_matrix(...)              # Orientation handling
find_grain_boundary_faces(...)             # Detect boundaries
compute_misorientation(...)                # GB characterization
create_abaqus_cohesive_section(...)        # CZM for GBs
```

**When to use:**
- Need specific crystallographic textures
- Study grain boundary properties
- Implement cohesive zone models in Abaqus
- Analyze microstructure statistics

---

## Complete Workflow

### Workflow 1: From MD to Abaqus (Aragonite Coral)

**Step 1: Run MD Simulations**
```bash
# Your LAMMPS/other MD code
# Output: elastic_constants.dat (6×6 matrix)
```

**Step 2: Convert Elastic Tensor**
```python
from src.elastic_tensor_converter import *

# Read your MD output
C_aragonite = np.array([
    [171.8,  57.5,  30.2,   0.0,   0.0,   0.0],
    [ 57.5, 106.7,  46.9,   0.0,   0.0,   0.0],
    [ 30.2,  46.9,  84.2,   0.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,  42.1,   0.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,  31.1,   0.0],
    [  0.0,   0.0,   0.0,   0.0,   0.0,  46.6]
])

S = stiffness_to_compliance(C_aragonite)
constants = compliance_to_engineering_constants(S)
valid, errors = validate_orthotropic_symmetry(constants)

if valid:
    print("Symmetry validated")
    abaqus_line = format_for_abaqus(constants, unit='MPa')
    with open('aragonite_constants.txt', 'w') as f:
        f.write(abaqus_line)
```

**Step 3: Generate Microstructure**
```python
from src.enhanced_microstructure_export import radial_needles_more_2d

volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=45,
    domain_size=200,
    needle_length_range=(20, 40),
    needles_per_center_range=(10, 35),
    resolution=100,
    z_constraint_factor=0.1,
    quasi_2d=True
)
```

**Step 4: Export with Properties and BCs**
```python
from src.enhanced_microstructure_export import export_to_abaqus_enhanced

material_props = {
    'needle_material': {
        'name': 'Aragonite',
        'type': 'orthotropic',
        'constants': [
            constants['E1']*1000, constants['E2']*1000, constants['E3']*1000,
            constants['nu12'], constants['nu13'], constants['nu23'],
            constants['G12']*1000, constants['G13']*1000, constants['G23']*1000
        ]
    }
}

export_to_abaqus_enhanced(
    volume, needle_volume, center_properties, 200,
    'aragonite_coral_tension.inp',
    material_properties=material_props,
    boundary_conditions={'type': 'tension_z', 'displacement': 2.0}
)
```

**Step 5: Export VTK for Visualisation (optional)**
```python
from src.enhanced_microstructure_export import export_vtk_unstructured

export_vtk_unstructured(
    volume, needle_volume, 200,
    'aragonite_coral.vtk',
    center_properties=center_properties
)
# Open in ParaView — color by grain_id or euler_Phi
```

**Step 6: Run in Abaqus**
```bash
abaqus job=aragonite_coral_tension cpus=4
abaqus viewer odb=aragonite_coral_tension.odb
```

---

### Workflow 2: BCC Polycrystal for MOOSE

**Step 1: Convert Iron Properties**
```python
from src.elastic_tensor_converter import *

C11, C12, C44 = 233.0, 135.0, 118.0  # GPa

C_bcc = np.array([
    [C11, C12, C12,  0.0,  0.0,  0.0],
    [C12, C11, C12,  0.0,  0.0,  0.0],
    [C12, C12, C11,  0.0,  0.0,  0.0],
    [0.0, 0.0, 0.0,  C44,  0.0,  0.0],
    [0.0, 0.0, 0.0,  0.0,  C44,  0.0],
    [0.0, 0.0, 0.0,  0.0,  0.0,  C44]
])

S = stiffness_to_compliance(C_bcc)
constants = compliance_to_engineering_constants(S)
```

**Step 2: Generate Polycrystal**
```python
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=100,
    domain_size=1000,
    needle_length_range=(100, 200),
    needles_per_center_range=(20, 40),
    resolution=200,
    z_constraint_factor=0.5,
    quasi_2d=False
)
```

**Step 3: Export to Exodus (for MOOSE)**
```python
from src.enhanced_microstructure_export import export_to_exodus
from src.assign_orientations import OrientationParams

export_to_exodus(
    volume, needle_volume, domain_size=1000,
    filename='bcc_polycrystal.e',
    center_properties=center_properties,
    orientation_params=OrientationParams(),
    orientation_seed=42
)
```

**Step 4: Convert for MOOSE**
```bash
python src/convert_to_moose.py bcc_polycrystal.e bcc_polycrystal_moose.e --verify
```
See `PIPELINE.md` for the complete MOOSE workflow.

---

### Workflow 3: Grain Boundaries with Cohesive Zones (Abaqus)

**Step 1: Generate Microstructure and Orientations**
```python
from src.microstructure_utils import generate_random_texture, euler_to_rotation_matrix

volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=10, domain_size=100,
    needle_length_range=(10, 20),
    needles_per_center_range=(10, 20),
    resolution=50, quasi_2d=False
)

grain_ids = np.unique(volume); grain_ids = grain_ids[grain_ids > 0]
orientations = generate_random_texture(len(grain_ids), texture_type='random')

grain_orientations = {}
for gid, (phi1, Phi, phi2) in zip(grain_ids, orientations):
    grain_orientations[gid] = euler_to_rotation_matrix(phi1, Phi, phi2)
```

**Step 2: Find Grain Boundaries**
```python
from src.microstructure_utils import find_grain_boundary_faces, compute_misorientation

boundary_faces = find_grain_boundary_faces(volume)

low_angle_gbs, high_angle_gbs = [], []
for face in boundary_faces:
    g1, g2 = face[3], face[4]
    if g1 in grain_orientations and g2 in grain_orientations:
        angle = np.degrees(compute_misorientation(grain_orientations[g1],
                                                   grain_orientations[g2]))
        (low_angle_gbs if angle < 15 else high_angle_gbs).append(face)
```

**Step 3: Create Cohesive Zones**
```python
from src.microstructure_utils import create_abaqus_cohesive_section

create_abaqus_cohesive_section(
    'gb_cohesive_zones.inp',
    boundary_faces,
    grain_orientations
)
```

---

## Script-by-Script Details

### Detailed: `elastic_tensor_converter.py`

#### Mathematical Background

```
S = C⁻¹

E₁ = 1/S₁₁,  E₂ = 1/S₂₂,  E₃ = 1/S₃₃
ν₁₂ = -S₁₂·E₁,  ν₁₃ = -S₁₃·E₁,  ν₂₃ = -S₂₃·E₂
G₁₂ = 1/S₆₆,  G₁₃ = 1/S₅₅,  G₂₃ = 1/S₄₄
```

Validation: ν₁₂/E₁ = ν₂₁/E₂, ν₁₃/E₁ = ν₃₁/E₃, ν₂₃/E₂ = ν₃₂/E₃

#### Examples Included
1. **Aragonite (orthorhombic)** - Experimental + simulation C matrix
2. **BCC Iron (cubic)** - With Voigt isotropic approximation
3. **Custom input** - Template for your data

---

### Detailed: `enhanced_microstructure_export.py`

#### Key Algorithm: Radial Needle Growth

1. **Nucleation**: Place random centres with minimum distance constraint
2. **Growth**: Needles grow radially from each centre
3. **Direction**: Quasi-2D (layered) or full 3D sphere distribution
4. **Filling**: Distance transform fills empty voxels to nearest needle
5. **Orientation**: Euler angles assigned via twin model (see `assign_orientations.py`)

#### Resolution Guide

| Resolution | Elements | Use |
|---|---|---|
| 20³ | ~8k | Quick testing |
| 50³ | ~125k | Development |
| 100³ | ~1M | Production |
| 200³ | ~8M | High resolution |

#### Boundary Condition Types

| Type | Fixed | Applied |
|---|---|---|
| `tension_z` | Bottom_Z (all DOF) | Top_Z (Z-disp) |
| `compression_z` | Bottom_Z (all DOF) | Top_Z (-Z-disp) |
| `shear_xy` | Bottom_Z (all DOF) | Top_Z (X-disp) |
| `biaxial_xy` | Bottom faces | Top_X, Top_Y |
| `custom` | User-defined | User-defined |

---

### Detailed: `microstructure_utils.py`

#### Texture Generation

```python
orientations = generate_random_texture(num_grains, 'random')  # uniform
orientations = generate_random_texture(num_grains, 'cube')    # {001}<100>
orientations = generate_random_texture(num_grains, 'goss')    # {110}<001>
```

#### Euler Angles (Bunge ZXZ convention)

```
φ₁: First rotation about Z  [0, 2π]
Φ:  Rotation about X'        [0, π]
φ₂: Second rotation about Z'' [0, 2π]
```

#### Misorientation Classification

```python
delta_R = R2 @ R1.T
theta   = arccos((trace(delta_R) - 1) / 2)
# Low-angle: θ < 15°,  High-angle: θ ≥ 15°
```

---

## Summary

**Three scripts, Abaqus workflow:**

1. **elastic_tensor_converter.py** - Stiffness to engineering constants
2. **enhanced_microstructure_export.py** - Generate → export with BCs
3. **microstructure_utils.py** - Advanced operations (texture, GBs)

**For MOOSE workflow:** see `PIPELINE.md` — uses `assign_orientations.py`
and `convert_to_moose.py` instead.

---
