# Microstructure FEM Export Toolkit

**Complete workflow for synthetic microstructure generation, material property conversion, and multi-format FEM export**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Python toolkit for computational materials science, providing:
- Synthetic microstructure generation (quasi-2D and 3D)
- Elastic tensor to engineering constants conversion
- Multi-format export (Abaqus, Exodus, VTK)
- Automatic material orientation assignment
- Ready-to-run boundary conditions
- Complete MD-to-FEM workflow

Developed for biomineralized structures (aragonite coral) and polycrystalline metals (BCC iron/steel).

---

## Features

### Microstructure Generation
- Radial needle structures (quasi-2D or full 3D)
- Grain and needle ID tracking
- Configurable grain sizes and aspect ratios
- Quasi-2D layered structures for EBSD comparison

### Material Property Handling
- **Elastic tensor converter**: 6×6 stiffness matrix → engineering constants
- Orthotropic and isotropic materials
- Automatic c-axis alignment for anisotropic materials
- MD/DFT integration ready

### Export Formats
- **Abaqus INP**: With orientations and boundary conditions
- **Exodus II**: For MOOSE framework
- **VTK**: For ParaView visualization
- **NumPy**: For custom post-processing

### Advanced Features
- Automatic boundary node sets (6 faces)
- Ready-to-run mechanical tests (tension, compression, shear, biaxial)
- Grain boundary analysis and cohesive zones
- Misorientation calculations
- Multi-format mesh conversion

---

## Repository Structure

```
microstructure-fem-export/
├── README.md                           # This file
├── LICENSE                             
├── requirements.txt                    # Dependencies
│
├── src/                                # Source code
│   ├── enhanced_microstructure_export.py   # Main export (with BC)
│   ├── microstructure_utils.py             # Utility functions
│   ├── elastic_tensor_converter.py         # Tensor 2 constants converter
│   └── convert2e.py                        # VTK 2 Exodus converter
│
├── examples/                           # Example scripts
│   └── practical_examples.py               # Quick start, examples on coral, optional bcc iron
│   
└── docs/                               # Documentation
    └── COMPLETE_WORKFLOW.md               # Details + workflow
 
```

---

## 🚀 Quick Start (30 seconds)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/microstructure-fem-export.git
cd microstructure-fem-export

# Install dependencies
pip install -r requirements.txt
```

### Minimal Example

```python
from src.enhanced_microstructure_export import (
    radial_needles_more_2d,
    export_to_abaqus_enhanced
)

# Generate microstructure
volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=10, domain_size=100, needle_length_range=(10, 20),
    needles_per_center_range=(10, 20), resolution=30
)

# Export with automatic boundary conditions
export_to_abaqus_enhanced(
    volume, needle_volume, center_properties, 100,
    'model.inp',
    boundary_conditions={'type': 'tension_z', 'displacement': 1.0}
)

# Ready to run: abaqus job=model
```

---

## 🔬 Complete Workflow

### Step 1: Convert Elastic Properties

```python
from src.elastic_tensor_converter import (
    stiffness_to_compliance,
    compliance_to_engineering_constants,
    validate_orthotropic_symmetry,
    format_for_abaqus
)

# Your 6×6 stiffness matrix in GPa
C_matrix = np.array([
    [171.8,  57.5,  30.2,   0.0,   0.0,   0.0],
    [ 57.5, 106.7,  46.9,   0.0,   0.0,   0.0],
    [ 30.2,  46.9,  84.2,   0.0,   0.0,   0.0],
    [  0.0,   0.0,   0.0,  42.1,   0.0,   0.0],
    [  0.0,   0.0,   0.0,   0.0,  31.1,   0.0],
    [  0.0,   0.0,   0.0,   0.0,   0.0,  46.6]
])

# Convert to engineering constants
S = stiffness_to_compliance(C_matrix)
constants = compliance_to_engineering_constants(S)
valid, errors = validate_orthotropic_symmetry(constants)

# Get Abaqus-ready format (check whether syntax formatting is correct)
abaqus_line = format_for_abaqus(constants, unit='MPa')
print(abaqus_line)
# Output: E1, E2, E3, nu12, nu13, nu23, G12, G13
 G23
```

### Step 2: Generate Microstructure

```python
from src.enhanced_microstructure_export import radial_needles_more_2d

volume, needle_volume, center_properties, _ = radial_needles_more_2d(
    num_centers=45,
    domain_size=200,
    needle_length_range=(20, 40),
    needles_per_center_range=(10, 35),
    resolution=100,
    quasi_2d=True
)
```

### Step 3: Export with Properties and BCs

```python
from src.enhanced_microstructure_export import export_to_abaqus_enhanced

material_props = {
    'needle_material': {
        'name': 'Aragonite',
        'type': 'orthotropic',
        'constants': [
            constants['E1']*1000,  constants['E2']*1000,  constants['E3']*1000,
            constants['nu12'], constants['nu13'], constants['nu23'],
            constants['G12']*1000, constants['G13']*1000, constants['G23']*1000
        ]
    }
}

export_to_abaqus_enhanced(
    volume, needle_volume, center_properties, 200,
    'complete_model.inp',
    material_properties=material_props,
    boundary_conditions={'type': 'tension_z', 'displacement': 2.0}
)
```

### Step 4: Convert to Other Formats

```python
from src.enhanced_microstructure_export import export_vtk_unstructured
from src.convert2e import MeshConverter

# Export to VTK
export_vtk_unstructured(
    volume, needle_volume, 200, 'model.vtk',
    center_properties=center_properties
)

# Convert VTK to Exodus for MOOSE (from cmd: python convert2e.py your_setup.vtk  --organize-by needle_id (or grain_id // or authomatically)
converter = MeshConverter('model.vtk', 'model.e', organize_by='grain_id')
converter.convert()
```

---

## 📚 Documentation

### Quick Links
- **[Complete Workflow](docs/COMPLETE_WORKFLOW.md)** - End-to-end examples

### Key Modules

#### 1. `enhanced_microstructure_export.py`
Main module for microstructure generation and export.

**Key Functions:**
- `radial_needles_more_2d()` - Generate synthetic microstructures
- `export_to_abaqus_enhanced()` - Export with BCs and orientations
- `export_vtk_unstructured()` - VTK with orientation data
- `export_to_exodus()` - Direct Exodus II export

#### 2. `elastic_tensor_converter.py`
Convert elastic tensors to FEM-ready constants.

**Key Functions:**
- `stiffness_to_compliance()` - C to S matrix
- `compliance_to_engineering_constants()` - Extract E, nu, G
- `validate_orthotropic_symmetry()` - Check validity
- `format_for_abaqus()` - Ready-to-paste format

#### 3. `convert2e.py`
Universal mesh converter to Exodus II format.

**Key Features:**
- Reorganizes elements by grain_id/needle_id
- Proper ParaView block organization
- Supports VTK, Abaqus INP, FEBio formats

#### 4. `microstructure_utils.py`
Utility functions for advanced operations.

**Key Functions:**
- `generate_random_texture()` - Crystallographic orientations
- `euler_to_rotation_matrix()` - Orientation handling
- `find_grain_boundary_faces()` - GB detection
- `compute_misorientation()` - GB characterization
- `create_abaqus_cohesive_section()` - CZM for GBs

---

## 🎯 Use Cases

### Cold-Water Coral Biomechanics

```python
# Generate quasi-2D aragonite structure
volume, needle_volume, props, _ = radial_needles_more_2d(
    num_centers=45, domain_size=200,
    needle_length_range=(20, 40),
    needles_per_center_range=(10, 35),
    resolution=100, quasi_2d=True
)

# Use MD-derived orthotropic properties
# (c-axis automatically aligned with needle direction)
material_props = {
    'needle_material': {
        'name': 'Aragonite',
        'type': 'orthotropic',
        'constants': [...]  # From elastic_tensor_converter
    }
}

# Export for mechanical testing
export_to_abaqus_enhanced(..., boundary_conditions={'type': 'tension_z', 'displacement': 2.0})
```

### Grain Boundary Engineering (BCC Iron/Steel)

```python
# Generate polycrystalline structure
volume, needle_volume, props, _ = radial_needles_more_2d(
    num_centers=100, domain_size=1000,
    resolution=200, quasi_2d=False  # Full 3D
)

# Assign random texture
from src.microstructure_utils import generate_random_texture
orientations = generate_random_texture(num_grains, 'random')

# Find and characterize grain boundaries
boundary_faces = find_grain_boundary_faces(volume)
# Apply misorientation-dependent cohesive zones
```

### MOOSE Framework Integration

```python
# Generate and export to Exodus
export_to_exodus(volume, needle_volume, domain_size, 'model.e')

# Or convert existing VTK
converter = MeshConverter('model.vtk', organize_by='grain_id')
converter.convert()  # Creates model.e

# Use in MOOSE input file:
# [Mesh]
#   file = model.e
# []
```

---

## 📊 Boundary Condition Types

| Type | Description | Fixed | Applied |
|------|-------------|-------|---------|
| `tension_z` | Uniaxial tension | Bottom_Z (all DOFs) | Top_Z (Z-disp) |
| `compression_z` | Uniaxial compression | Bottom_Z (all DOFs) | Top_Z (-Z-disp) |
| `shear_xy` | Simple shear | Bottom_Z (all DOFs) | Top_Z (X-disp, Z-constrained) |
| `biaxial_xy` | Biaxial tension | Bottom faces | Top_X, Top_Y |
| `custom` | User-defined | As specified | As specified |

All models export with 6 boundary node sets: `Bottom_Z`, `Top_Z`, `Bottom_X`, `Top_X`, `Bottom_Y`, `Top_Y`.

---

## 🔧 Requirements

### Core Requirements
```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
pyvista >= 0.32
```

### Optional Requirements
```
meshio >= 5.0  # For Exodus II export
```

Install all:
```bash
pip install -r requirements.txt
```

---

