# -*- coding: utf-8 -*-
"""
Practical Examples - Using the Enhanced Microstructure Export Tools

This script demonstrates three common use cases:
1. Aragonite coral microstructure with orthotropic needles
2. BCC polycrystalline iron with random texture
3. Adding grain boundary cohesive zones

@author: nk03
"""

import numpy as np
from enhanced_microstructure_export import (
    radial_needles_more_2d,
    export_to_abaqus_enhanced,
    export_to_exodus,
    export_vtk_unstructured,
    visualize_slices
)
from microstructure_utils import (
    generate_random_texture,
    euler_to_rotation_matrix,
    find_grain_boundary_faces,
    create_abaqus_cohesive_section,
    analyze_microstructure_stats,
    export_orientation_data
)


def example1_aragonite_coral():
    """
    Example 1: Aragonite coral microstructure
    - Quasi-2D flower-like structure
    - Orthotropic material properties
    - Automatic needle orientation assignment
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: ARAGONITE CORAL MICROSTRUCTURE")
    print("="*70)
    
    # Parameters matching cold-water coral aragonite
    domain_size = 10  # 200 microns
    num_centers = 4   # 45 nucleation centers (grains)
    needle_length_range = (10, 50)  # 20-40 microns
    needles_per_center_range = (5, 25)
    resolution = 20
    
    print("\nGenerating microstructure...")
    volume, needle_volume, center_properties, mesh = radial_needles_more_2d(
        num_centers, domain_size, needle_length_range, needles_per_center_range,
        resolution, z_constraint_factor=0.1, quasi_2d=False
    )
    
    # MD-derived aragonite properties
    material_props = {
        'needle_material': {
            'name': 'Aragonite',
            'type': 'orthotropic',
            # From your MD simulations: E1 (a-axis), E2 (b-axis), E3 (c-axis)
            'constants': [
                140.432e3,   # E1 (MPa)
                70.297e3,   # E2 (MPa)
                63.413e3,  # E3 (MPa) - stiffest along c-axis
                0.5048,      # nu12
                0.0775,      # nu13
                0.4664,      # nu23
                46.600e3,   # G12 (MPa)
                31.100e3,    # G13 (MPa)
                42.100e3    # G23 (MPa)
            ]
        }
    }
    needle_count = 0
    print('First 6 needles\' properties:')
    for center in center_properties:
        for needle in center['needles']:
            needle_count += 1
            direction = needle['direction']
            mag = np.linalg.norm(direction)
            print(f"   Needle {needle['id']}: direction={direction}, |c|={mag:.6f}")
            if needle_count >= 6:  # Just show first 3
                break
        if needle_count >= 6:
            break

#    print(f"\n   Total needles with direction data: {needle_count}")
    print(f"   ✓ c-axis is aligned with needle direction (33 component)")

    print("\nExporting to Abaqus...")
    export_to_abaqus_enhanced(
        volume, needle_volume, center_properties, domain_size,
        'tiny_aragonite.inp',
        material_properties=material_props
    )
    
    print("\nExporting to VTK for visualization...")
    export_vtk_unstructured(
        volume, needle_volume, domain_size,
        'tiny_aragonite.vtk',
        center_properties=center_properties
    )
    
    print("\nCreating visualization...")
    visualize_slices(volume, domain_size, 'tiny_aragonite')
    
    print("\nExporting orientation data...")
    export_orientation_data(center_properties, 'tiny_orientations.csv')
    
    print("\nExporting to exodus...")
    export_to_exodus(
                volume, needle_volume, domain_size,
                'tiny_aragonite.e',
                center_properties=center_properties
            )
    
#    try:
#        export_to_exodus(
#            volume, needle_volume, domain_size,
#            'tiny_aragonite.e',
#            center_properties=center_properties
#        )
#    except Exception as e:
#        print(f"Could not export to Exodus (meshio required): {e}")
        
    # Statistics
    print("\n" + "-"*70)
    print("MICROSTRUCTURE STATISTICS")
    print("-"*70)
    stats = analyze_microstructure_stats(volume, needle_volume)
    print(f"Number of grains: {stats['num_grains']}")
    print(f"Number of needles: {stats['num_needles']}")
    print(f"Mean grain size: {stats['mean_grain_size']:.1f} voxels")
    print(f"Mean needle size: {stats['mean_needle_size']:.1f} voxels")
    print(f"Number of GB faces: {stats['num_gb_faces']}")
    
    return volume, needle_volume, center_properties


def example2_bcc_polycrystal():
    """
    Example 2: BCC Polycrystalline Iron
    - 3D equiaxed grain structure
    - Random texture
    - Isotropic material per grain (but different orientations)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: BCC POLYCRYSTALLINE IRON")
    print("="*70)
    
    # Parameters for polycrystalline structure
    domain_size = 1000  # 1 mm cube
    num_centers = 100   # 100 grains
    needle_length_range = (100, 200)  # Equiaxed-ish grains
    needles_per_center_range = (20, 40)
    resolution = 100  # 10 micron voxels
    
    print("\nGenerating microstructure...")
    volume, needle_volume, center_properties, mesh = radial_needles_more_2d(
        num_centers, domain_size, needle_length_range, needles_per_center_range,
        resolution, z_constraint_factor=0.5, quasi_2d=False  # Full 3D
    )
    
    # BCC iron properties (isotropic at grain level)
    material_props = {
        'needle_material': {  # Using 'needle' sets but treating as grains
            'name': 'BCC_Iron',
            'type': 'isotropic',
            'E': 211e3,  # MPa
            'nu': 0.29
        }
    }
    
    print("\nExporting to Abaqus...")
    export_to_abaqus_enhanced(
        volume, needle_volume, center_properties, domain_size,
        'example2_bcc_iron.inp',
        material_properties=material_props
    )
    
    # Generate random crystallographic orientations
    print("\nGenerating random texture...")
    grain_ids = np.unique(volume)
    grain_ids = grain_ids[grain_ids > 0]
    orientations = generate_random_texture(len(grain_ids), texture_type='random')
    
    # Create orientation dictionary
    grain_orientations = {}
    for gid, (phi1, Phi, phi2) in zip(grain_ids, orientations):
        grain_orientations[gid] = euler_to_rotation_matrix(phi1, Phi, phi2)
    
    # Add orientations to INP file
    print("\nAdding crystallographic orientations...")
    _add_grain_orientations_to_inp('example2_bcc_iron.inp', grain_orientations)
    
    print("\nExporting to Exodus for MOOSE...")
    try:
        export_to_exodus(
            volume, needle_volume, domain_size,
            'example2_bcc_iron.e'
        )
    except Exception as e:
        print(f"Could not export to Exodus (meshio required): {e}")
    
    print("\nCreating visualization...")
    visualize_slices(volume, domain_size, 'example2_bcc_iron')
    
    return volume, needle_volume, grain_orientations


def example3_grain_boundaries_with_cohesive():
    """
    Example 3: Adding Grain Boundary Cohesive Zones
    - Generate microstructure
    - Find grain boundaries
    - Assign cohesive properties based on misorientation
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: GRAIN BOUNDARIES WITH COHESIVE ZONES")
    print("="*70)
    
    # Generate a simple microstructure
    domain_size = 100
    num_centers = 10  # Small for demonstration
    needle_length_range = (10, 20)
    needles_per_center_range = (10, 20)
    resolution = 50
    
    print("\nGenerating microstructure...")
    volume, needle_volume, center_properties, mesh = radial_needles_more_2d(
        num_centers, domain_size, needle_length_range, needles_per_center_range,
        resolution, z_constraint_factor=0.3, quasi_2d=False
    )
    
    # Generate orientations
    print("\nGenerating grain orientations...")
    grain_ids = np.unique(volume)
    grain_ids = grain_ids[grain_ids > 0]
    orientations = generate_random_texture(len(grain_ids), texture_type='random')
    
    grain_orientations = {}
    for gid, (phi1, Phi, phi2) in zip(grain_ids, orientations):
        grain_orientations[gid] = euler_to_rotation_matrix(phi1, Phi, phi2)
    
    # Find grain boundaries
    print("\nFinding grain boundaries...")
    boundary_faces = find_grain_boundary_faces(volume)
    print(f"Found {len(boundary_faces)} grain boundary faces")
    
    # Create cohesive zone model
    print("\nCreating cohesive zone model...")
    create_abaqus_cohesive_section(
        'example3_cohesive_zones.inp',
        boundary_faces,
        grain_orientations
    )
    
    # Export base microstructure
    material_props = {
        'needle_material': {
            'name': 'Substrate',
            'type': 'isotropic',
            'E': 200e3,
            'nu': 0.3
        }
    }
    
    export_to_abaqus_enhanced(
        volume, needle_volume, center_properties, domain_size,
        'example3_base.inp',
        material_properties=material_props
    )
    
    print("\nNote: Merge example3_base.inp and example3_cohesive_zones.inp")
    print("      to create complete model with cohesive elements")
    
    return volume, needle_volume, boundary_faces


def _add_grain_orientations_to_inp(filename, grain_orientations):
    """
    Helper function to add grain orientations to Abaqus INP
    """
    # Read file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find where to insert orientations (after materials, before sections)
    insert_idx = 0
    for i, line in enumerate(lines):
        if 'SECTION ASSIGNMENTS' in line:
            insert_idx = i
            break
    
    # Create orientation cards
    orientation_lines = [
        "**\n",
        "** ----------------------------------------------------------------\n",
        "**                    GRAIN ORIENTATIONS\n",
        "** ----------------------------------------------------------------\n"
    ]
    
    for grain_id, R in grain_orientations.items():
        orientation_lines.append(f"*Orientation, name=OrientGrain{grain_id}\n")
        orientation_lines.append(
            f"{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}, "
            f"{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}\n"
        )
        orientation_lines.append("1, 0\n")
    
    # Insert
    lines = lines[:insert_idx] + orientation_lines + lines[insert_idx:]
    
    # Modify solid sections to reference orientations
    # Find needle sections and map them to grain orientations
    for i, line in enumerate(lines):
        if '*Solid Section, elset=Needle' in line:
            # Extract needle ID
            needle_id = int(line.split('Needle')[1].split(',')[0])
            # For this example, map needle to grain (simplified)
            # In reality, you'd need to track which needle belongs to which grain
            grain_id = (needle_id % len(grain_orientations)) + 1
            # Add orientation if not already present
            if 'orientation=' not in line:
                lines[i] = line.rstrip().rstrip(',') + f", orientation=OrientGrain{grain_id}\n"
    
    # Write modified file
    output_file = filename.replace('.inp', '_oriented.inp')
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    print(f"Created oriented model: {output_file}")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("ENHANCED MICROSTRUCTURE EXPORT - PRACTICAL EXAMPLES")
    print("="*70)
    
    # Uncomment the example you want to run:
    
    # Example 1: Aragonite coral (your current work)
    volume1, needle_volume1, center_props1 = example1_aragonite_coral()
    
    # Example 2: BCC polycrystal (for DFG proposal)
    # volume2, needle_volume2, grain_orients = example2_bcc_polycrystal()
    
    # Example 3: Cohesive zones (for GB mechanics)
    # volume3, needle_volume3, boundaries = example3_grain_boundaries_with_cohesive()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - example1_aragonite.inp (Abaqus input with orientations)")
    print("  - example1_aragonite.vtk (VTK for ParaView)")
    print("  - example1_aragonite_slices.png (Visualization)")
    print("  - example1_orientations.csv (Needle directions)")
    print("\nNext steps:")
    print("  1. Visualize in ParaView: File > Open > example1_aragonite.vtk")
    print("  2. Import to Abaqus: File > Import > Input File > example1_aragonite.inp")
    print("  3. Check orientations: View orientation data in example1_orientations.csv")
    print("  4. Modify materials: Edit material_props dictionary in the script")


if __name__ == "__main__":
    main()
