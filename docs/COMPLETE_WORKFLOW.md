# Complete Workflow Guide

This guide explains the full MD→FEM workflow using all four core scripts in the toolkit.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [The Four Core Scripts](#the-four-core-scripts)
3. [Complete Workflow](#complete-workflow)
4. [Script-by-Script Details](#script-by-script-details)
5. [Practical Examples](#practical-examples)
6. [Integration Patterns](#integration-patterns)

---

## Overview

### The Complete Pipeline

```
MD/DFT Simulations
       ↓
[elastic_tensor_converter.py]  ← Convert 6×6 matrix to E, ν, G
       ↓
Engineering Constants
       ↓
[enhanced_microstructure_export.py]  ← Generate + export with BCs
       ↓
Abaqus INP + VTK files
       ↓
[convert2e.py]  ← Convert to Exodus if needed
       ↓
Multiple FEM formats ready
       ↓
Run simulations in Abaqus/MOOSE
```

### Why Four Scripts?

1. **`elastic_tensor_converter.py`** - Bridges MD/DFT and FEM
2. **`enhanced_microstructure_export.py`** - Core microstructure generation
3. **`microstructure_utils.py`** - Advanced operations (texture, GBs)
4. **`convert2e.py`** - Format flexibility (VTK→Exodus)

Each script is modular and can be used independently or as part of the complete workflow.

---

## The Four Core Scripts

### 1. `elastic_tensor_converter.py`

**Purpose:** Convert elastic tensors from atomistic simulations to FEM-ready engineering constants.

**Input:** 6×6 stiffness matrix C (GPa) from MD/DFT  
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

**Purpose:** Generate synthetic microstructures and export to FEM formats with automatic orientation and boundary condition assignment.

**Input:** Microstructure parameters (size, grains, resolution)  
**Output:** Abaqus INP, VTK, Exodus II files

**Key Functions:**
```python
radial_needles_more_2d(...)                # Generate microstructure
export_to_abaqus_enhanced(...)             # Export with BCs
export_vtk_unstructured(...)               # VTK with orientations
export_to_exodus(...)                      # Direct Exodus export
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
- Implement cohesive zone models
- Analyze microstructure statistics

### 4. `convert2e.py`

**Purpose:** Convert between mesh formats with proper element block organization for ParaView/MOOSE.

**Input:** VTK, Abaqus INP, or FEBio files  
**Output:** Exodus II format (.e)

**Key Features:**
- Reorganizes elements by grain_id/needle_id
- Creates proper element blocks for ParaView
- Command-line interface
- Automatic field detection

**When to use:**
- Need Exodus format for MOOSE
- Want proper ParaView visualization
- Converting between FEM formats
- Organizing existing meshes by material ID

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

# Convert to engineering constants
S = stiffness_to_compliance(C_aragonite)
constants = compliance_to_engineering_constants(S)
valid, errors = validate_orthotropic_symmetry(constants)

if valid:
    print("✓ Symmetry validated")
    print(f"E1 = {constants['E1']:.3f} GPa")
    print(f"E2 = {constants['E2']:.3f} GPa")
    print(f"E3 = {constants['E3']:.3f} GPa")
    
    # Save for later
    abaqus_line = format_for_abaqus(constants, unit='MPa')
    with open('aragonite_constants.txt', 'w') as f:
        f.write(abaqus_line)
```

**Step 3: Generate Microstructure**
```python
from src.enhanced_microstructure_export import radial_needles_more_2d

# Generate quasi-2D coral structure
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=45,              # ~45 nucleation sites
    domain_size=200,             # 200 microns
    needle_length_range=(20, 40), # Realistic needle lengths
    needles_per_center_range=(10, 35),
    resolution=100,              # High resolution
    z_constraint_factor=0.1,     # Quasi-2D layering
    quasi_2d=True
)

print(f"Generated {len(center_properties)} grains")
print(f"Total needles: {sum(len(c['needles']) for c in center_properties)}")
```

**Step 4: Export with Properties and BCs**
```python
from src.enhanced_microstructure_export import export_to_abaqus_enhanced

# Define material from MD constants
material_props = {
    'needle_material': {
        'name': 'Aragonite',
        'type': 'orthotropic',
        'constants': [
            constants['E1']*1000,  # Convert GPa → MPa
            constants['E2']*1000,
            constants['E3']*1000,
            constants['nu12'],
            constants['nu13'],
            constants['nu23'],
            constants['G12']*1000,
            constants['G13']*1000,
            constants['G23']*1000
        ]
    }
}

# Apply 1% strain
strain = 0.01
displacement = 200 * strain  # 2.0 microns

# Export complete model
export_to_abaqus_enhanced(
    volume, needle_volume, center_properties, 200,
    'aragonite_coral_tension.inp',
    material_properties=material_props,
    boundary_conditions={
        'type': 'tension_z',
        'displacement': displacement
    }
)
```

**Step 5: Visualize (Optional)**
```python
from src.enhanced_microstructure_export import export_vtk_unstructured

export_vtk_unstructured(
    volume, needle_volume, 200,
    'aragonite_coral.vtk',
    center_properties=center_properties
)

# Open in ParaView:
# - Color by grain_id
# - Color by euler_Phi (c-axis orientation)
# - Color by boundary_flag (boundary nodes)
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

# BCC Iron stiffness
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

# For cubic, also compute isotropic approximation
K = (C11 + 2*C12) / 3
G = (C11 - C12 + 3*C44) / 5
E = 9*K*G / (3*K + G)
nu = (3*K - 2*G) / (2*(3*K + G))

print(f"Isotropic approximation: E={E:.1f} GPa, ν={nu:.3f}")
```

**Step 2: Generate Polycrystal**
```python
from src.enhanced_microstructure_export import radial_needles_more_2d

# Full 3D polycrystalline structure
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=100,             # 100 grains
    domain_size=1000,            # 1 mm cube
    needle_length_range=(100, 200),  # Equiaxed grains
    needles_per_center_range=(20, 40),
    resolution=200,              # 5 micron voxels
    z_constraint_factor=0.5,     # Full 3D
    quasi_2d=False
)
```

**Step 3: Assign Texture**
```python
from src.microstructure_utils import generate_random_texture, euler_to_rotation_matrix

# Get unique grain IDs
grain_ids = np.unique(volume)
grain_ids = grain_ids[grain_ids > 0]

# Generate random texture
orientations = generate_random_texture(
    len(grain_ids),
    texture_type='random'  # or 'cube', 'goss'
)

# Store as rotation matrices
grain_orientations = {}
for gid, (phi1, Phi, phi2) in zip(grain_ids, orientations):
    grain_orientations[gid] = euler_to_rotation_matrix(phi1, Phi, phi2)

print(f"Assigned orientations to {len(grain_ids)} grains")
```

**Step 4: Export to VTK First**
```python
from src.enhanced_microstructure_export import export_vtk_unstructured

export_vtk_unstructured(
    volume, needle_volume, 1000,
    'bcc_polycrystal.vtk',
    center_properties=center_properties
)
```

**Step 5: Convert to Exodus**
```python
from src.convert2e import MeshConverter

# Convert with proper block organization
converter = MeshConverter(
    'bcc_polycrystal.vtk',
    'bcc_polycrystal.e',
    organize_by='grain_id'  # Organize by grain
)

success = converter.convert()

if success:
    print("✓ Ready for MOOSE")
    print("  Each grain is a separate element block")
    print("  Assign materials in MOOSE input file")
```

**Step 6: Use in MOOSE**
```
[Mesh]
  file = bcc_polycrystal.e
[]

[Materials]
  [./grain_material]
    type = ComputeElasticityTensor
    # Assign per-block material properties
    # Use grain_id to identify blocks
  [../]
[]
```

---

### Workflow 3: Grain Boundary Cohesive Zones (to be done)

**Step 1: Generate and Orient**
```python
from src.enhanced_microstructure_export import radial_needles_more_2d
from src.microstructure_utils import generate_random_texture, euler_to_rotation_matrix

# Generate microstructure
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=20, domain_size=100,
    needle_length_range=(10, 20), needles_per_center_range=(10, 20),
    resolution=50, quasi_2d=False
)

# Assign orientations
grain_ids = np.unique(volume)[1:]
orientations = generate_random_texture(len(grain_ids), 'random')

grain_orientations = {}
for gid, (phi1, Phi, phi2) in zip(grain_ids, orientations):
    grain_orientations[gid] = euler_to_rotation_matrix(phi1, Phi, phi2)
```

**Step 2: Find Grain Boundaries**
```python
from src.microstructure_utils import find_grain_boundary_faces, compute_misorientation

# Detect all grain boundary faces
boundary_faces = find_grain_boundary_faces(volume)
print(f"Found {len(boundary_faces)} GB faces")

# Classify by misorientation
low_angle_gbs = []
high_angle_gbs = []
special_gbs = []  # e.g., Σ3, Σ5, etc.

for face in boundary_faces:
    grain1_id, grain2_id = face[3], face[4]
    
    R1 = grain_orientations[grain1_id]
    R2 = grain_orientations[grain2_id]
    
    misori = compute_misorientation(R1, R2)
    angle_deg = np.rad2deg(misori)
    
    if angle_deg < 15:
        low_angle_gbs.append(face)
    elif 58 < angle_deg < 62:  # ~60° = Σ3 twin
        special_gbs.append(face)
    else:
        high_angle_gbs.append(face)

print(f"Low-angle GBs: {len(low_angle_gbs)}")
print(f"High-angle GBs: {len(high_angle_gbs)}")
print(f"Special GBs (Σ3): {len(special_gbs)}")
```

**Step 3: Create Cohesive Zones**
```python
from src.microstructure_utils import create_abaqus_cohesive_section

# Create cohesive zone model
create_abaqus_cohesive_section(
    'gb_cohesive_zones.inp',
    boundary_faces,
    grain_orientations
)

# This will create:
# - Cohesive element definitions
# - Material properties based on misorientation
# - Low-angle GBs: stronger
# - High-angle GBs: weaker
```

---

## Script-by-Script Details

### Detailed: `elastic_tensor_converter.py`

#### Purpose
Bridges the gap between atomistic simulations (MD/DFT) and continuum FEM by converting between different representations of elastic properties.

#### Mathematical Background

**Stiffness matrix C (Voigt notation):**
```
σ = C ε

where σ = stress (6×1), ε = strain (6×1), C = stiffness (6×6)
```

**Compliance matrix S:**
```
ε = S σ

S = C⁻¹
```

**Engineering constants extraction:**
```
E₁ = 1/S₁₁
E₂ = 1/S₂₂
E₃ = 1/S₃₃

ν₁₂ = -S₁₂ · E₁
ν₁₃ = -S₁₃ · E₁
ν₂₃ = -S₂₃ · E₂

G₁₂ = 1/S₆₆
G₁₃ = 1/S₅₅
G₂₃ = 1/S₄₄
```

#### Validation
Must satisfy orthotropic symmetry:
```
ν₁₂/E₁ = ν₂₁/E₂
ν₁₃/E₁ = ν₃₁/E₃
ν₂₃/E₂ = ν₃₂/E₃
```

#### Examples Included
1. **Aragonite (orthorhombic)** - From DFT calculations
2. **BCC Iron (cubic)** - With isotropic approximation
3. **Custom input** - Template for your data

#### Integration with MD
```python
# From LAMMPS elastic compute:
# Read elastic constant matrix output
# Convert using this script
# Use in Abaqus/MOOSE
```

---

### Detailed: `enhanced_microstructure_export.py`

#### Purpose
Core module for generating synthetic microstructures with automatic material orientation and boundary condition assignment.

#### Key Algorithm: Radial Needle Growth

1. **Nucleation**: Place random centers with minimum distance
2. **Growth**: Needles grow radially from each center
3. **Direction**: Quasi-2D (layered) or full 3D distribution
4. **Filling**: Distance transform fills empty spaces
5. **Orientation**: Each needle's direction → material c-axis

#### Microstructure Parameters

**`num_centers`**: Number of grains
- Small (5-10): Quick testing
- Medium (20-50): Realistic microstructures
- Large (100+): Statistical studies

**`domain_size`**: Physical size in microns
- Match your experimental scale
- Consider computational cost

**`needle_length_range`**: (min, max) in microns
- Aragonite: (20, 40)
- BCC equiaxed: (100, 200)

**`resolution`**: Voxels per dimension
- 20³: Testing (~8k elements)
- 50³: Development (~125k elements)
- 100³: Production (~1M elements)
- 200³: High resolution (~8M elements)

**`quasi_2d`**: Structure type
- True: Layered (matches EBSD cross-sections)
- False: Full 3D

#### Material Orientation Assignment

For each needle:
1. Primary axis (a₁) = needle direction (normalized)
2. Secondary axis (a₂) = orthogonal to a₁ and global Z
3. Tertiary axis (a₃) = a₁ × a₂

Written as Abaqus `*Orientation` card:
```
*Orientation, name=Orient1
a1x, a1y, a1z, a2x, a2y, a2z
1, 0
```

#### Boundary Conditions

Five predefined types:
1. **tension_z**: Fix bottom, pull top
2. **compression_z**: Fix bottom, compress top
3. **shear_xy**: Fix bottom, shear top
4. **biaxial_xy**: Pull in X and Y
5. **custom**: User-defined combination

All create node sets: `Bottom_Z`, `Top_Z`, `Bottom_X`, `Top_X`, `Bottom_Y`, `Top_Y`

---

### Detailed: `microstructure_utils.py`

#### Purpose
Advanced operations: texture generation, grain boundary analysis, cohesive zones.

#### Texture Generation

**Random texture:**
```python
orientations = generate_random_texture(num_grains, 'random')
# Uniform distribution in orientation space
```

**Cube texture:** `{001}<100>`
```python
orientations = generate_random_texture(num_grains, 'cube')
# Concentrated around cube orientation with scatter
```

**Goss texture:** `{110}<001>`
```python
orientations = generate_random_texture(num_grains, 'goss')
# Typical for rolled steels
```

#### Euler Angles

**Bunge convention (ZXZ):**
```
φ₁: First rotation about Z (0 to 2π)
Φ: Rotation about X' (0 to π)
φ₂: Second rotation about Z'' (0 to 2π)
```

**Conversion to rotation matrix:**
```python
R = euler_to_rotation_matrix(phi1, Phi, phi2, convention='bunge')
# Returns 3×3 orthogonal matrix
```

#### Grain Boundary Detection

**Algorithm:**
```python
boundary_faces = find_grain_boundary_faces(volume)
# Returns list of (elem1_ijk, elem2_ijk, normal, grain1_id, grain2_id)
```

For each internal face:
- Check if grain_id differs between neighbors
- Store face location and grain IDs
- Use for cohesive zone insertion

#### Misorientation Calculation

**Formula:**
```python
ΔR = R₂ @ R₁ᵀ
θ = arccos((trace(ΔR) - 1) / 2)
```

**Classification:**
- Low-angle: θ < 15°
- High-angle: θ ≥ 15°
- Special: Near CSL values (Σ3=60°, Σ5=36.9°, etc.)


---

### Detailed: `convert2e.py`

#### Purpose
Convert meshes to Exodus II format with proper element block organization for ParaView and MOOSE.

#### Key Feature: Block Reorganization

**Problem:** VTK files often have one large element block
**Solution:** Reorganize into separate blocks by grain_id/needle_id

**Before:**
```
Block 1: All 1,000,000 elements
  grain_id: [1, 1, 1, 2, 2, 2, 3, 3, 3, ...]
```

**After:**
```
Block 1 (grain_id=1): 50,000 elements
Block 2 (grain_id=2): 45,000 elements
Block 3 (grain_id=3): 52,000 elements
...
```

**Result:** ParaView can color by block, MOOSE can assign materials per block

#### Usage

**Command line:**
```bash
# Auto-detect field
python convert2e.py microstructure.vtk

# Specify field
python convert2e.py model.vtk --organize-by needle_id

# Custom output
python convert2e.py model.vtk -o output.e --organize-by grain_id
```

**Python API:**
```python
from src.convert2e import MeshConverter

converter = MeshConverter('model.vtk', 'model.e', organize_by='grain_id')
success = converter.convert()
```

#### Supported Input Formats
- VTK (.vtk, .vtu, .vti)
- Abaqus INP (.inp)
- FEBio (.feb)

#### Output
- Exodus II (.e, .exo)
- Compatible with ParaView, MOOSE, Cubit

---

## Summary

**Four scripts, complete workflow:**

1. **elastic_tensor_converter.py** - MD/DFT → engineering constants
2. **enhanced_microstructure_export.py** - Generate → export with BCs
3. **microstructure_utils.py** - Advanced operations (texture, GBs)
4. **convert2e.py** - Format conversion → MOOSE/ParaView

**Typical usage:**
1. Run MD → get C matrix
2. Convert C → engineering constants
3. Generate microstructure
4. Export with properties and BCs
5. Convert to needed format
6. Run FEM simulation

**All modular:** Use any script independently or in combination.

---
