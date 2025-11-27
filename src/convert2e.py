#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTK to Exodus Converter with Proper Euler Angles

Reads VTK with complete orientation data (c_axis, a_axis, b_axis)
and converts to Exodus format with correct Euler angles

Author: Nikolai
Date: 2025
"""

import numpy as np
import meshio
import netCDF4 as nc
import argparse
import sys
from pathlib import Path


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Bunge Euler angles (ZXZ convention)
    
    R should be [a-axis, b-axis, c-axis] as columns
    """
    Phi = np.arccos(np.clip(R[2, 2], -1, 1))
    
    if np.abs(np.sin(Phi)) < 1e-6:
        phi1 = np.arctan2(R[0, 1], R[0, 0])
        phi2 = 0.0
    else:
        phi1 = np.arctan2(R[2, 0] / np.sin(Phi), -R[2, 1] / np.sin(Phi))
        phi2 = np.arctan2(R[0, 2] / np.sin(Phi), R[1, 2] / np.sin(Phi))
    
    return np.degrees(phi1) % 360, np.degrees(Phi), np.degrees(phi2) % 360


def convert_vtk_to_exodus(vtk_file, exodus_file, organize_by='needle_id'):
    """
    Convert VTK to Exodus with proper Euler angles
    
    Parameters:
    -----------
    vtk_file : str
        Input VTK file with c_axis, a_axis, b_axis data
    exodus_file : str  
        Output Exodus file
    organize_by : str
        Field to organize blocks by ('grain_id' or 'needle_id')
    """
    
    print(f"Reading VTK: {vtk_file}")
    mesh = meshio.read(vtk_file)
    
    print(f"  Points: {len(mesh.points)}")
    print(f"  Cells: {sum(len(block.data) for block in mesh.cells)}")
    
    # Check for required fields
    required_fields = [
        'c_axis_x', 'c_axis_y', 'c_axis_z',
        'a_axis_x', 'a_axis_y', 'a_axis_z',
        'b_axis_x', 'b_axis_y', 'b_axis_z',
        'grain_id', 'needle_id'
    ]
    
    missing = [f for f in required_fields if f not in mesh.cell_data]
    if missing:
        print(f"\n✗ ERROR: Missing fields in VTK: {missing}")
        print("  Your VTK must have complete orientation data (c, a, b axes)")
        return False
    
    print(f"  ✓ Has complete orientation data")
    
    # Get cell data arrays (from first block)
    c_x = mesh.cell_data['c_axis_x'][0]
    c_y = mesh.cell_data['c_axis_y'][0]
    c_z = mesh.cell_data['c_axis_z'][0]
    
    a_x = mesh.cell_data['a_axis_x'][0]
    a_y = mesh.cell_data['a_axis_y'][0]
    a_z = mesh.cell_data['a_axis_z'][0]
    
    b_x = mesh.cell_data['b_axis_x'][0]
    b_y = mesh.cell_data['b_axis_y'][0]
    b_z = mesh.cell_data['b_axis_z'][0]
    
    grain_id = mesh.cell_data['grain_id'][0]
    needle_id = mesh.cell_data['needle_id'][0]
    
    # Compute Euler angles for each element
    print(f"\nComputing Euler angles...")
    n_cells = len(c_x)
    euler_phi1 = np.zeros(n_cells)
    euler_Phi = np.zeros(n_cells)
    euler_phi2 = np.zeros(n_cells)
    
    for i in range(n_cells):
        # Build rotation matrix
        c_axis = np.array([c_x[i], c_y[i], c_z[i]])
        a_axis = np.array([a_x[i], a_y[i], a_z[i]])
        b_axis = np.array([b_x[i], b_y[i], b_z[i]])
        
        # R = [a-axis, b-axis, c-axis]
        R = np.column_stack([a_axis, b_axis, c_axis])
        
        # Convert to Euler angles
        phi1, Phi, phi2 = rotation_matrix_to_euler(R)
        
        euler_phi1[i] = phi1
        euler_Phi[i] = Phi
        euler_phi2[i] = phi2
    
    print(f"  ✓ Computed Euler angles")
    print(f"    phi1: [{euler_phi1.min():.1f}, {euler_phi1.max():.1f}]°")
    print(f"    Phi:  [{euler_Phi.min():.1f}, {euler_Phi.max():.1f}]°")
    print(f"    phi2: [{euler_phi2.min():.1f}, {euler_phi2.max():.1f}]°")
    
    # Write basic mesh
    print(f"\nWriting Exodus: {exodus_file}")
    meshio.write(exodus_file, mesh, file_format="exodus")
    
    # Add element variables
    print(f"  Adding element variables...")
    exo = nc.Dataset(exodus_file, 'r+')
    
    try:
        # Create element variable dimension
        num_vars = 5
        exo.createDimension('num_elem_var', num_vars)
        
        # Variable names
        name_var = exo.createVariable('name_elem_var', 'S1', ('num_elem_var', 'len_string'))
        var_names = ['grain_id', 'needle_id', 'euler_phi1', 'euler_Phi', 'euler_phi2']
        
        for i, vname in enumerate(var_names):
            name_str = vname.ljust(33, '\x00')
            name_array = np.array([c.encode('utf-8') for c in name_str], dtype='S1')
            name_var[i, :] = name_array
        
        # Write data for each variable
        var_data = [grain_id, needle_id, euler_phi1, euler_Phi, euler_phi2]
        
        for var_idx, data in enumerate(var_data):
            var_name = f'vals_elem_var{var_idx+1}eb1'
            var = exo.createVariable(var_name, 'f8', ('time_step', 'num_el_in_blk1'))
            var[0, :] = np.array(data, dtype=np.float64)
        
        print(f"  ✓ Wrote {num_vars} element variables")
        
    finally:
        exo.close()
    
    print(f"\n✓ Conversion complete!")
    print(f"\nMOOSE Usage:")
    print(f"  [AuxVariables]")
    print(f"    [euler_phi1] family = MONOMIAL order = CONSTANT []")
    print(f"    [euler_Phi] family = MONOMIAL order = CONSTANT []")
    print(f"    [euler_phi2] family = MONOMIAL order = CONSTANT []")
    print(f"  []")
    print(f"")
    print(f"  [UserObjects]")
    print(f"    [prop_read]")
    print(f"      type = ElementPropertyReadFile")
    print(f"      prop_file_name = '{exodus_file}'")
    print(f"      nprop = 5")
    print(f"    []")
    print(f"  []")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert VTK to Exodus with proper Euler angles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python vtk_to_exodus.py microstructure.vtk -o mesh.e
  
Requires:
  - VTK file with c_axis_x/y/z, a_axis_x/y/z, b_axis_x/y/z
  - meshio and netCDF4 packages
        """
    )
    
    parser.add_argument('input', help='Input VTK file')
    parser.add_argument('-o', '--output', help='Output Exodus file (default: input.e)')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"✗ Error: File not found: {args.input}")
        sys.exit(1)
    
    # Determine output filename
    if args.output:
        output = args.output
    else:
        output = Path(args.input).with_suffix('.e')
    
    # Convert
    try:
        success = convert_vtk_to_exodus(args.input, output)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()