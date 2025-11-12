# Microstructure FEM Export Toolkit

**Complete workflow for synthetic microstructure generation, material property conversion, and multi-format FEM export**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Python toolkit for computational materials science, providing:
- 🏗️ Synthetic microstructure generation (quasi-2D and 3D)
- 🔄 Elastic tensor to engineering constants conversion
- 📤 Multi-format export (Abaqus, Exodus, VTK)
- ⚙️ Automatic material orientation assignment
- 🎯 Ready-to-run boundary conditions
- 🔗 Complete MD-to-FEM workflow

Developed for biomineralized structures (aragonite coral) and polycrystalline metals (BCC iron/steel).

---

## ⭐ Features

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

## 📦 Repository Structure

```
microstructure-fem-export/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Dependencies
│
├── src/                                # Source code
│   ├── enhanced_microstructure_export.py   # Main export (with BC)
│   ├── microstructure_utils.py             # Utility functions
│   ├── elastic_tensor_converter.py         # Tensor→constants converter
│   ├── convert2e.py                        # VTK→Exodus converter
│   └── __init__.py                         # Package init
│
├── examples/                           # Example scripts
│   ├── 01_minimal_example.py               # Quick start
│   ├── 02_boundary_conditions.py           # All BC types
│   ├── 03_aragonite_coral.py               # Realistic coral
│   ├── 04_bcc_polycrystal.py               # Polycrystalline iron
│   ├── 05_material_properties.py           # Elastic tensor usage
│   └── 06_complete_workflow.py             # Full MD→FEM pipeline
│
├── docs/                               # Documentation
│   ├── 01_GETTING_STARTED.md               # Installation and basics
│   ├── 02_MICROSTRUCTURE_GENERATION.md     # Microstructure guide
│   ├── 03_MATERIAL_PROPERTIES.md           # Material definition
│   ├── 04_BOUNDARY_CONDITIONS.md           # BC application
│   ├── 05_EXPORT_FORMATS.md                # Format-specific guides
│   ├── 06_COMPLETE_WORKFLOW.md             # End-to-end examples
│   ├── 07_API_REFERENCE.md                 # Complete API
│   └── 08_TROUBLESHOOTING.md               # Common issues
│
├── tests/                              # Tests (optional)
│   └── test_export.py
│
└── output/                             # Output directory (gitignored)
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

### Step 1: Convert MD/DFT Elastic Properties

```python
from src.elastic_tensor_converter import (
    stiffness_to_compliance,
    compliance_to_engineering_constants,
    validate_orthotropic_symmetry,
    format_for_abaqus
)

# Your 6×6 stiffness matrix from MD/DFT (GPa)
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

# Get Abaqus-ready format
abaqus_line = format_for_abaqus(constants, unit='MPa')
print(abaqus_line)
# Output: E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
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

# Convert VTK to Exodus for MOOSE
converter = MeshConverter('model.vtk', 'model.e', organize_by='grain_id')
converter.convert()
```

---

## 📚 Documentation

### Quick Links
- **[Getting Started](docs/01_GETTING_STARTED.md)** - Installation and first steps
- **[Material Properties](docs/03_MATERIAL_PROPERTIES.md)** - Using elastic tensor converter
- **[Boundary Conditions](docs/04_BOUNDARY_CONDITIONS.md)** - Automatic BC application
- **[Complete Workflow](docs/06_COMPLETE_WORKFLOW.md)** - End-to-end examples
- **[API Reference](docs/07_API_REFERENCE.md)** - Complete function documentation

### Key Modules

#### 1. `enhanced_microstructure_export.py`
Main module for microstructure generation and export.

**Key Functions:**
- `radial_needles_more_2d()` - Generate synthetic microstructures
- `export_to_abaqus_enhanced()` - Export with BCs and orientations
- `export_vtk_unstructured()` - VTK with orientation data
- `export_to_exodus()` - Direct Exodus II export

#### 2. `elastic_tensor_converter.py`
Convert elastic tensors from MD/DFT to FEM-ready constants.

**Key Functions:**
- `stiffness_to_compliance()` - C → S matrix
- `compliance_to_engineering_constants()` - Extract E, ν, G
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

## 📖 Examples

### Example 1: Minimal (30 seconds)

```bash
cd examples
python 01_minimal_example.py
```

Generates:
- `minimal_model.inp` - Abaqus file with node sets
- Creates ~8,000 elements
- Node sets available for manual BC setup

### Example 2: All Boundary Conditions

```bash
python 02_boundary_conditions.py
```

Generates 5 models with different BCs:
- Tension, compression, shear, biaxial, custom

### Example 3: Realistic Aragonite Coral

```bash
python 03_aragonite_coral.py
```

Generates:
- Complete model with MD properties
- 1% tensile strain applied
- Ready to submit to Abaqus
- VTK for visualization

### Example 4: Material Property Workflow

```bash
python 05_material_properties.py
```

Shows complete workflow:
- Define stiffness matrix
- Convert to engineering constants
- Validate symmetry
- Generate microstructure
- Export with properties

---

## 📈 Performance

| Resolution | Elements | Export Time | File Size | Use Case |
|------------|----------|-------------|-----------|----------|
| 20³ | ~8k | <1s | ~1 MB | Quick testing |
| 50³ | ~125k | ~5s | ~15 MB | Development |
| 100³ | ~1M | ~30s | ~120 MB | Production |
| 200³ | ~8M | ~5min | ~1 GB | High resolution |

Tested on: Intel i7, 16GB RAM

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 👥 Authors

**Nikolai K.**  
Postdoctoral Researcher  
Clausthal University of Technology  
Computational Materials Physics Group

---

## 🙏 Acknowledgments

- Computational Materials Physics group, TU Clausthal
- DFG (German Research Foundation)
- Universitat Politècnica de Catalunya
- Marta Pena Fernandez (original microCT-to-microFE framework)

---

## 📧 Contact

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/microstructure-fem-export/issues)
- **Email**: nikolai.k@tu-clausthal.de
- **Research Gate**: [Your profile]

---

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{microstructure_fem_export_2024,
  author = {Nikolai K.},
  title = {Microstructure FEM Export Toolkit},
  year = {2024},
  publisher = {GitHub},
  institution = {Clausthal University of Technology},
  url = {https://github.com/yourusername/microstructure-fem-export}
}
```

---

## 🔗 Related Projects

- [MOOSE Framework](https://mooseframework.inl.gov/)
- [MTEX](https://mtex-toolbox.github.io/) - EBSD analysis
- [Dream.3D](http://dream3d.bluequartz.net/) - Microstructure generation

---

## 🗺️ Roadmap

- [ ] GUI for interactive microstructure generation
- [ ] Integration with Dream.3D
- [ ] Automatic periodic boundary conditions
- [ ] Support for more crystal systems (FCC, HCP)
- [ ] Machine learning for property prediction
- [ ] Automated mesh refinement at grain boundaries

---

**⭐ If you find this useful, please star the repository!**

**🐛 Found a bug? [Report it](https://github.com/yourusername/microstructure-fem-export/issues)**

**💡 Have a suggestion? [Let us know](https://github.com/yourusername/microstructure-fem-export/discussions)**
