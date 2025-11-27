# -*- coding: utf-8 -*-
"""
Enhanced Microstructure to FE Export
Generates artificial grain boundary networks with needle structures and exports to multiple FE formats

@author: nk03
Enhanced version with improved material handling and Exodus II support
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LinearSegmentedColormap
import os

def generate_nucleation_centers(num_centers, bounds, min_distance=None, z_constraint=None):
    """
    Generate random nucleation centers with optional minimum distance constraint
    and optional z-plane constraint for more 2D-like structures
    """
    centers = []
    attempts = 0
    max_attempts = 10000
    
    while len(centers) < num_centers and attempts < max_attempts:
        if z_constraint is not None:
            # Generate points more constrained in z-direction (quasi-2D)
            z_val = np.random.uniform(z_constraint[0], z_constraint[1])
            point = np.array([
                np.random.uniform(bounds[0], bounds[1]),
                np.random.uniform(bounds[0], bounds[1]),
                z_val
            ])
        else:
            point = np.random.rand(3) * (bounds[1] - bounds[0]) + bounds[0]
        
        if min_distance is not None and centers:
            # Check distance to existing centers
            distances = np.array([np.linalg.norm(point - c) for c in centers])
            if np.min(distances) < min_distance:
                attempts += 1
                continue
        
        centers.append(point)
        attempts = 0
    
    return np.array(centers)

def radial_needles_more_2d(num_centers, domain_size, needle_length_range, 
                           needles_per_center_range, resolution=100,
                           z_constraint_factor=0.2, quasi_2d=True):
    """
    Generate 3D volume with radially-oriented needle crystals growing from nucleation centers,
    with adjustments to create more 2D-like flower patterns and separate grain/needle IDs
    """
    # Create uniform grid
    mesh = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(domain_size/resolution, domain_size/resolution, domain_size/resolution),
        origin=(0, 0, 0)
    )
    
    # Set z-constraint for nucleation centers to create more layered structure
    if quasi_2d:
        # Create multiple z-layers for a more layered structure
        num_layers = min(5, num_centers)  # Limit number of layers
        layer_thickness = domain_size / num_layers
        z_constraints = [(i*layer_thickness, (i+1)*layer_thickness) for i in range(num_layers)]
        
        # Distribute centers among layers
        centers_per_layer = [num_centers // num_layers + (1 if i < num_centers % num_layers else 0) 
                            for i in range(num_layers)]
        
        centers = []
        for layer_idx, count in enumerate(centers_per_layer):
            if count > 0:
                min_distance = domain_size / (count ** (1/2)) * 0.7  # Adjusted for 2D spacing
                layer_centers = generate_nucleation_centers(
                    count, [0, domain_size], min_distance, z_constraints[layer_idx])
                centers.extend(layer_centers)
        
        centers = np.array(centers)
    else:
        # Original 3D distribution
        min_distance = domain_size / (num_centers ** (1/3)) * 0.5
        centers = generate_nucleation_centers(num_centers, [0, domain_size], min_distance)

    # Initialize empty volume and needle ID volume
    volume = np.zeros((resolution, resolution, resolution), dtype=int)  # For grain ID (center ID)
    needle_volume = np.zeros((resolution, resolution, resolution), dtype=int)  # For needle ID
    
    # Properties for each center and its needles
    center_properties = []
    needle_id_counter = 1
    
    # For each nucleation center, generate needles radiating outward
    for center_idx, center in enumerate(centers):
        # Define grain ID as center ID + 1 (to avoid 0)
        grain_id = center_idx + 1
        
        # Number of needles to generate from this center
        num_needles = np.random.randint(needles_per_center_range[0], needles_per_center_range[1]+1)
           
        if quasi_2d:
            # Generate directions primarily in xy plane with small z component for quasi-2D effect
            # Using golden angle for more even distribution in 2D
            golden_angle = np.pi * (3 - np.sqrt(5))
            theta = np.array([i * golden_angle for i in range(num_needles)])
            
            # Create needle directions primarily in XY plane
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Add small z-component variation for slight 3D effect
            z = np.random.uniform(-z_constraint_factor, z_constraint_factor, num_needles)
            
            # Normalize directions
            directions = np.column_stack((x, y, z))
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        else:
            # Original 3D distribution on a sphere
            phi = np.random.uniform(0, 2*np.pi, num_needles)
            costheta = np.random.uniform(-1, 1, num_needles)
            theta = np.arccos(costheta)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            directions = np.column_stack((x, y, z))
        
        # Center properties
        center_prop = {
            'position': center,
            'id': grain_id,
            'needles': []
        }
        
        # Generate each needle from this center
        for dir_idx, direction in enumerate(directions):
            # Needle length
            length = np.random.uniform(needle_length_range[0], needle_length_range[1])
            
            # Aspect ratio (length/width)
            aspect_ratio = np.random.uniform(8, 15)  # Typical for needle-like structures
            width = length / aspect_ratio
            
            # Needle properties
            needle_prop = {
                'id': needle_id_counter,
                'grain_id': grain_id,
                'length': length,
                'width': width,
                'direction': direction,
                'aspect_ratio': aspect_ratio
            }
            
            center_prop['needles'].append(needle_prop)
            
            # Convert to voxel representation
            center_voxel = np.round(center * resolution / domain_size).astype(int)
            
            # Create points along the needle
            t_values = np.linspace(0, 1, int(length/width*3))  # More points for longer needles
            for t in t_values:
                # Position along the needle from center
                pos = center + direction * t * length
                
                # Convert to voxel coordinates
                voxel_pos = np.round(pos * resolution / domain_size).astype(int)
                
                # Check if within bounds
                if (0 <= voxel_pos[0] < resolution and 
                    0 <= voxel_pos[1] < resolution and 
                    0 <= voxel_pos[2] < resolution):
                    
                    # Calculate width at this position (elliptical profile)
                    current_width = width * np.sqrt(1 - (2*t - 1)**2)
                    width_voxels = max(1, int(current_width * resolution / domain_size))
                    
                    # Fill voxels around this point (spherical)
                    for di in range(-width_voxels, width_voxels+1):
                        for dj in range(-width_voxels, width_voxels+1):
                            for dk in range(-width_voxels, width_voxels+1):
                                # Check if within spherical radius
                                if di**2 + dj**2 + dk**2 <= width_voxels**2:
                                    new_i = voxel_pos[0] + di
                                    new_j = voxel_pos[1] + dj
                                    new_k = voxel_pos[2] + dk
                                    
                                    if (0 <= new_i < resolution and 
                                        0 <= new_j < resolution and 
                                        0 <= new_k < resolution):
                                        
                                        volume[new_i, new_j, new_k] = grain_id
                                        needle_volume[new_i, new_j, new_k] = needle_id_counter
            
            needle_id_counter += 1
        
        center_properties.append(center_prop)
    
    # Fill empty spaces using distance transform
    print("Filling empty spaces...")
    empty_mask = (volume == 0)
    if np.any(empty_mask):
        distances, indices = distance_transform_edt(empty_mask, return_indices=True)
        filled_volume = volume.copy()
        filled_needle = needle_volume.copy()
        
        for idx in np.ndindex(empty_mask.shape):
            if empty_mask[idx]:
                nearest_idx = tuple(ind[idx] for ind in indices)
                filled_volume[idx] = volume[nearest_idx]
                filled_needle[idx] = needle_volume[nearest_idx]
        
        volume = filled_volume
        needle_volume = filled_needle
    
    # Add to mesh
    mesh['grain_id'] = volume.flatten(order='F')
    mesh['needle_id'] = needle_volume.flatten(order='F')
    
    return volume, needle_volume, center_properties, mesh

def export_to_abaqus_enhanced(volume, needle_volume, center_properties, domain_size, 
                               filename, material_properties=None, spacing=None, origin=None,
                               boundary_conditions=None):
    """
    Enhanced Abaqus INP export with flexible material definitions and boundary conditions
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D array containing grain IDs
    needle_volume : numpy.ndarray
        3D array containing needle IDs
    center_properties : list
        List of center properties containing needle information
    domain_size : float
        Physical size of the domain
    filename : str
        Output filename
    material_properties : dict, optional
        Dictionary defining materials:
        {
            'grain_material': {'name': 'Material1', 'type': 'isotropic', 'E': value, 'nu': value},
            'needle_material': {'name': 'Aragonite', 'type': 'orthotropic', 'constants': [...]}
        }
    spacing : tuple, optional
        Grid spacing
    origin : tuple, optional
        Grid origin
    boundary_conditions : dict, optional
        Dictionary defining boundary conditions and loading:
        {
            'type': 'tension_z' | 'compression_z' | 'shear_xy' | 'biaxial_xy' | 'custom',
            'displacement': float (for displacement-controlled),
            'force': float (for force-controlled),
            'step_time': float (default 1.0),
            'custom': {
                'fixed': ['Bottom_Z', ...],  # Node sets to fix completely
                'prescribed': [('Top_Z', 3, 1.0), ...]  # (set, DOF, value) tuples
            }
        }
        If None, creates a minimal step with one fixed corner node
    """
    
    # Get mesh dimensions
    ni, nj, nk = volume.shape
    
    # Set default spacing and origin if not provided
    if spacing is None:
        spacing = (domain_size/ni, domain_size/nj, domain_size/nk)
    if origin is None:
        origin = (0, 0, 0)
    
    # Default material properties if not provided
    if material_properties is None:
        material_properties = {
            'needle_material': {
                'name': 'Aragonite',
                'type': 'orthotropic',
                'constants': [38.617e3, 52.212e3, 118.645e3, 0.498, 0.232, 0.106, 
                             34.069e3, 37.36e3, 23.996e3]  # E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
            }
        }
    
    # Create a lookup dictionary for needle properties
    needle_directions = {}
    for center in center_properties:
        for needle in center['needles']:
            needle_directions[needle['id']] = needle['direction']
    
    with open(filename, 'w') as f:
        # Write header
        f.write("*Heading\n")
        f.write("** Radial Needle Microstructure for Abaqus\n")
        f.write("** Generated with enhanced export script\n")
        f.write("**\n")
        
        # Write nodes
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                          NODES\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("*Node\n")
        node_id = 1
        node_ids = np.zeros((ni+1, nj+1, nk+1), dtype=int)
        
        for k in range(nk+1):
            for j in range(nj+1):
                for i in range(ni+1):
                    # Real space coordinates
                    x = i * spacing[0] + origin[0]
                    y = j * spacing[1] + origin[1]
                    z = k * spacing[2] + origin[2]
                    
                    f.write(f"{node_id}, {x:.6f}, {y:.6f}, {z:.6f}\n")
                    node_ids[i, j, k] = node_id
                    node_id += 1
        
        # Write elements (C3D8 - hexahedral)
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                          ELEMENTS\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("*Element, type=C3D8\n")
        elem_id = 1
        elem_to_grain = {}
        elem_to_needle = {}
        
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    # Get the 8 corner nodes for this element (Abaqus ordering)
                    n1 = node_ids[i, j, k]
                    n2 = node_ids[i+1, j, k]
                    n3 = node_ids[i+1, j+1, k]
                    n4 = node_ids[i, j+1, k]
                    n5 = node_ids[i, j, k+1]
                    n6 = node_ids[i+1, j, k+1]
                    n7 = node_ids[i+1, j+1, k+1]
                    n8 = node_ids[i, j+1, k+1]
                    
                    f.write(f"{elem_id}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")
                    
                    # Store grain and needle IDs for this element
                    grain_id = volume[i, j, k]
                    needle_id = needle_volume[i, j, k] if needle_volume is not None else grain_id
                    
                    if grain_id > 0:  # Skip empty elements
                        elem_to_grain[elem_id] = grain_id
                        elem_to_needle[elem_id] = needle_id
                    
                    elem_id += 1
        
        # Write element sets for grains
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                     GRAIN ELEMENT SETS\n")
        f.write("** ----------------------------------------------------------------\n")
        grain_ids = sorted(set(elem_to_grain.values()))
        for grain_id in grain_ids:
            f.write(f"*Elset, elset=Grain{grain_id}\n")
            
            # Find elements with this grain ID
            grain_elems = [eid for eid, gid in elem_to_grain.items() if gid == grain_id]
            
            # Write in groups of 16 (Abaqus format)
            for i in range(0, len(grain_elems), 16):
                line_elems = grain_elems[i:i+16]
                f.write(", ".join(map(str, line_elems)) + "\n")
        
        # Write element sets for needles
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                     NEEDLE ELEMENT SETS\n")
        f.write("** ----------------------------------------------------------------\n")
        needle_ids = sorted(set(elem_to_needle.values()))
        for needle_id in needle_ids:
            f.write(f"*Elset, elset=Needle{needle_id}\n")
            
            # Find elements with this needle ID
            needle_elems = [eid for eid, nid in elem_to_needle.items() if nid == needle_id]
            
            # Write in groups of 16 (Abaqus format)
            for i in range(0, len(needle_elems), 16):
                line_elems = needle_elems[i:i+16]
                f.write(", ".join(map(str, line_elems)) + "\n")
        
        # Write boundary node sets for applying BCs
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                   BOUNDARY NODE SETS\n")
        f.write("** ----------------------------------------------------------------\n")
        
        # Identify boundary nodes
        # Bottom Z (z=0)
        bottom_z_nodes = []
        for i in range(ni+1):
            for j in range(nj+1):
                bottom_z_nodes.append(node_ids[i, j, 0])
        
        # Top Z (z=max)
        top_z_nodes = []
        for i in range(ni+1):
            for j in range(nj+1):
                top_z_nodes.append(node_ids[i, j, nk])
        
        # Bottom X (x=0)
        bottom_x_nodes = []
        for j in range(nj+1):
            for k in range(nk+1):
                bottom_x_nodes.append(node_ids[0, j, k])
        
        # Top X (x=max)
        top_x_nodes = []
        for j in range(nj+1):
            for k in range(nk+1):
                top_x_nodes.append(node_ids[ni, j, k])
        
        # Bottom Y (y=0)
        bottom_y_nodes = []
        for i in range(ni+1):
            for k in range(nk+1):
                bottom_y_nodes.append(node_ids[i, 0, k])
        
        # Top Y (y=max)
        top_y_nodes = []
        for i in range(ni+1):
            for k in range(nk+1):
                top_y_nodes.append(node_ids[i, nj, k])
        
        # Write node sets
        boundary_sets = [
            ('Bottom_Z', bottom_z_nodes, 'z = 0 (bottom face)'),
            ('Top_Z', top_z_nodes, f'z = {domain_size} (top face)'),
            ('Bottom_X', bottom_x_nodes, 'x = 0 (left face)'),
            ('Top_X', top_x_nodes, f'x = {domain_size} (right face)'),
            ('Bottom_Y', bottom_y_nodes, 'y = 0 (front face)'),
            ('Top_Y', top_y_nodes, f'y = {domain_size} (back face)')
        ]
        
        for set_name, node_list, description in boundary_sets:
            f.write(f"** {description}\n")
            f.write(f"*Nset, nset={set_name}\n")
            
            # Write in groups of 16 (Abaqus format)
            for i in range(0, len(node_list), 16):
                line_nodes = node_list[i:i+16]
                f.write(", ".join(map(str, line_nodes)) + "\n")
        
        # Write materials
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                          MATERIALS\n")
        f.write("** ----------------------------------------------------------------\n")
        
        # Write material definition for needles
        if 'needle_material' in material_properties:
            mat = material_properties['needle_material']
            f.write(f"*Material, name={mat['name']}\n")
            
            if mat['type'] == 'orthotropic':
                f.write("*Elastic, type=ENGINEERING CONSTANTS\n")
                constants = mat['constants']
                f.write(f"{constants[0]}, {constants[1]}, {constants[2]}, {constants[3]}, "
                       f"{constants[4]}, {constants[5]}, {constants[6]}, {constants[7]}\n {constants[8]}\n")
            elif mat['type'] == 'isotropic':
                f.write("*Elastic\n")
                f.write(f"{mat['E']}, {mat['nu']}\n")
        
        # Write material definition for grains if different from needles
        if 'grain_material' in material_properties:
            mat = material_properties['grain_material']
            f.write(f"*Material, name={mat['name']}\n")
            
            if mat['type'] == 'isotropic':
                f.write("*Elastic\n")
                f.write(f"{mat['E']}, {mat['nu']}\n")
        
        # Write section assignments with orientations
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                    SECTION ASSIGNMENTS\n")
        f.write("** ----------------------------------------------------------------\n")
        
        for needle_id in needle_ids:
            if needle_id in needle_directions:
                # Get the needle direction vector
                direction = needle_directions[needle_id]
                
                # Calculate material orientation
                # Primary axis (a1) is along the needle direction (c-axis for aragonite)
                a1 = direction / np.linalg.norm(direction)
                
                # Choose a secondary direction orthogonal to a1
                global_z = np.array([0, 0, 1])
                if np.abs(np.dot(a1, global_z)) > 0.99:  # If nearly parallel to Z
                    global_x = np.array([1, 0, 0])
                    a2 = np.cross(a1, global_x)
                else:
                    a2 = np.cross(a1, global_z)
                
                # Normalize a2
                a2 = a2 / np.linalg.norm(a2)
                
                # The third direction is orthogonal to both
                a3 = np.cross(a1, a2)
                a3 = a3 / np.linalg.norm(a3)
                
                # Get material name
                mat_name = material_properties.get('needle_material', {}).get('name', 'Aragonite')
                
                # Assign the material and orientation to the needle element set
                f.write(f"*Solid Section, elset=Needle{needle_id}, material={mat_name}, "
                       f"orientation=Orient{needle_id}\n")
                f.write(",\n")
                
                # Define orientation
                f.write(f"*Orientation, name=Orient{needle_id}\n")
                f.write(f"{a1[0]:.6f}, {a1[1]:.6f}, {a1[2]:.6f}, {a2[0]:.6f}, {a2[1]:.6f}, {a2[2]:.6f}\n")
                f.write("1, 0\n")  # Rectangular system, no additional rotation
            else:
                # Fallback if no direction is found
                mat_name = material_properties.get('needle_material', {}).get('name', 'Aragonite')
                f.write(f"*Solid Section, elset=Needle{needle_id}, material={mat_name}\n")
                f.write(",\n")
        
        # Write step with boundary conditions
        f.write("**\n")
        f.write("** ----------------------------------------------------------------\n")
        f.write("**                            STEP\n")
        f.write("** ----------------------------------------------------------------\n")
        
        # Apply boundary conditions based on specification
        if boundary_conditions is None:
            # Default: minimal step with one fixed corner
            f.write("*Step, name=Step-1\n")
            f.write("*Static\n")
            f.write("1., 1., 1e-05, 1.\n")
            f.write("**\n")
            f.write("** Default: Fix one corner to prevent rigid body motion\n")
            f.write("*Boundary\n")
            f.write(f"Bottom_Z, 1, 3\n")
            f.write("**\n")
            f.write("*End Step\n")
        
        else:
            bc_type = boundary_conditions.get('type', 'custom')
            step_time = boundary_conditions.get('step_time', 1.0)
            
            if bc_type == 'tension_z':
                # Uniaxial tension along Z-axis
                displacement = boundary_conditions.get('displacement', 1.0)
                f.write("*Step, name=Tension_Z, nlgeom=NO\n")
                f.write("*Static\n")
                f.write(f"0.01, {step_time}, 1e-08, 0.1\n")
                f.write("**\n")
                f.write("** UNIAXIAL TENSION TEST (Z-AXIS)\n")
                f.write("** Fix bottom surface completely\n")
                f.write("*Boundary\n")
                f.write("Bottom_Z, 1, 3\n")
                f.write("**\n")
                f.write("** Apply displacement to top surface\n")
                f.write("*Boundary\n")
                f.write(f"Top_Z, 3, 3, {displacement}\n")
                f.write("**\n")
                f.write("** OUTPUT REQUESTS\n")
                f.write("*Output, field, variable=PRESELECT\n")
                f.write("*Output, history, variable=PRESELECT\n")
                f.write("**\n")
                f.write("*End Step\n")
            
            elif bc_type == 'compression_z':
                # Uniaxial compression along Z-axis
                displacement = boundary_conditions.get('displacement', -0.5)
                f.write("*Step, name=Compression_Z, nlgeom=NO\n")
                f.write("*Static\n")
                f.write(f"0.01, {step_time}, 1e-08, 0.1\n")
                f.write("**\n")
                f.write("** UNIAXIAL COMPRESSION TEST (Z-AXIS)\n")
                f.write("** Fix bottom surface completely\n")
                f.write("*Boundary\n")
                f.write("Bottom_Z, 1, 3\n")
                f.write("**\n")
                f.write("** Apply negative displacement to top surface\n")
                f.write("*Boundary\n")
                f.write(f"Top_Z, 3, 3, {displacement}\n")
                f.write("**\n")
                f.write("** OUTPUT REQUESTS\n")
                f.write("*Output, field, variable=PRESELECT\n")
                f.write("*Output, history, variable=PRESELECT\n")
                f.write("**\n")
                f.write("*End Step\n")
            
            elif bc_type == 'shear_xy':
                # Simple shear in XY plane
                displacement = boundary_conditions.get('displacement', 0.5)
                f.write("*Step, name=Shear_XY, nlgeom=NO\n")
                f.write("*Static\n")
                f.write(f"0.01, {step_time}, 1e-08, 0.1\n")
                f.write("**\n")
                f.write("** SIMPLE SHEAR TEST (XY PLANE)\n")
                f.write("** Fix bottom surface completely\n")
                f.write("*Boundary\n")
                f.write("Bottom_Z, 1, 3\n")
                f.write("**\n")
                f.write("** Apply shear displacement to top surface\n")
                f.write("*Boundary\n")
                f.write(f"Top_Z, 1, 1, {displacement}\n")
                f.write("Top_Z, 3, 3, 0.0\n")
                f.write("**\n")
                f.write("** OUTPUT REQUESTS\n")
                f.write("*Output, field, variable=PRESELECT\n")
                f.write("*Output, history, variable=PRESELECT\n")
                f.write("**\n")
                f.write("*End Step\n")
            
            elif bc_type == 'biaxial_xy':
                # Biaxial tension in XY plane
                displacement = boundary_conditions.get('displacement', 0.5)
                f.write("*Step, name=Biaxial_XY, nlgeom=NO\n")
                f.write("*Static\n")
                f.write(f"0.01, {step_time}, 1e-08, 0.1\n")
                f.write("**\n")
                f.write("** BIAXIAL TENSION TEST (XY PLANE)\n")
                f.write("** Fix bottom corners to prevent rigid body motion\n")
                f.write("*Boundary\n")
                f.write("Bottom_X, 1, 1, 0.0\n")
                f.write("Bottom_Y, 2, 2, 0.0\n")
                f.write("Bottom_Z, 3, 3, 0.0\n")
                f.write("**\n")
                f.write("** Apply biaxial displacement\n")
                f.write("*Boundary\n")
                f.write(f"Top_X, 1, 1, {displacement}\n")
                f.write(f"Top_Y, 2, 2, {displacement}\n")
                f.write("**\n")
                f.write("** OUTPUT REQUESTS\n")
                f.write("*Output, field, variable=PRESELECT\n")
                f.write("*Output, history, variable=PRESELECT\n")
                f.write("**\n")
                f.write("*End Step\n")
            
            elif bc_type == 'custom':
                # Custom boundary conditions
                f.write("*Step, name=CustomLoad, nlgeom=NO\n")
                f.write("*Static\n")
                f.write(f"0.01, {step_time}, 1e-08, 0.1\n")
                f.write("**\n")
                f.write("** CUSTOM BOUNDARY CONDITIONS\n")
                
                custom_bc = boundary_conditions.get('custom', {})
                
                # Apply fixed boundary conditions
                fixed_sets = custom_bc.get('fixed', [])
                if fixed_sets:
                    f.write("** Fixed node sets\n")
                    f.write("*Boundary\n")
                    for node_set in fixed_sets:
                        f.write(f"{node_set}, 1, 3\n")
                    f.write("**\n")
                
                # Apply prescribed displacements
                prescribed = custom_bc.get('prescribed', [])
                if prescribed:
                    f.write("** Prescribed displacements\n")
                    f.write("*Boundary\n")
                    for node_set, dof, value in prescribed:
                        f.write(f"{node_set}, {dof}, {dof}, {value}\n")
                    f.write("**\n")
                
                # Apply concentrated forces
                forces = custom_bc.get('forces', [])
                if forces:
                    f.write("** Concentrated forces\n")
                    f.write("*Cload\n")
                    for node_set, dof, value in forces:
                        f.write(f"{node_set}, {dof}, {value}\n")
                    f.write("**\n")
                
                # Apply distributed loads (pressure)
                pressures = custom_bc.get('pressures', [])
                if pressures:
                    f.write("** Distributed pressure loads\n")
                    for elem_set, value in pressures:
                        f.write(f"*Dsload\n")
                        f.write(f"{elem_set}, P, {value}\n")
                    f.write("**\n")
                
                f.write("** OUTPUT REQUESTS\n")
                f.write("*Output, field, variable=PRESELECT\n")
                f.write("*Output, history, variable=PRESELECT\n")
                f.write("**\n")
                f.write("*End Step\n")

    
    print(f"Abaqus INP file written: {filename}")
    print(f"  Total elements: {elem_id-1}")
    print(f"  Total nodes: {node_id-1}")
    print(f"  Grains: {len(grain_ids)}")
    print(f"  Needles: {len(needle_ids)}")
    print(f"  Boundary node sets: Bottom_Z, Top_Z, Bottom_X, Top_X, Bottom_Y, Top_Y")
    
    if boundary_conditions is not None:
        bc_type = boundary_conditions.get('type', 'custom')
        if bc_type == 'tension_z':
            disp = boundary_conditions.get('displacement', 1.0)
            print(f"  ✓ Applied BCs: Tension test (Bottom_Z fixed, Top_Z displacement={disp})")
        elif bc_type == 'compression_z':
            disp = boundary_conditions.get('displacement', -0.5)
            print(f"  ✓ Applied BCs: Compression test (Bottom_Z fixed, Top_Z displacement={disp})")
        elif bc_type == 'shear_xy':
            disp = boundary_conditions.get('displacement', 0.5)
            print(f"  ✓ Applied BCs: Shear test (Bottom_Z fixed, Top_Z shear={disp})")
        elif bc_type == 'biaxial_xy':
            disp = boundary_conditions.get('displacement', 0.5)
            print(f"  ✓ Applied BCs: Biaxial tension (displacement={disp})")
        elif bc_type == 'custom':
            print(f"  ✓ Applied BCs: Custom boundary conditions")
        print(f"  ✓ Ready to run in Abaqus!")
    else:
        print(f"  ✓ Minimal step (add boundary conditions manually)")

def export_to_exodus(volume, needle_volume, domain_size, filename, 
                    center_properties=None, spacing=None, origin=None):
    """
    Export to Exodus II format with Euler angles
    
    CHANGED: Direct Euler angle calculation from needle direction (c-axis)
    """
    try:
        import meshio
        import netCDF4 as nc
    except ImportError:
        print("meshio and netCDF4 required for Exodus export")
        print("Install with: pip install meshio netCDF4 --break-system-packages")
        return
    
    ni, nj, nk = volume.shape
    
    if spacing is None:
        spacing = (domain_size/ni, domain_size/nj, domain_size/nk)
    if origin is None:
        origin = (0, 0, 0)
    
    # Build lookup for needle directions (these ARE the c-axes!)
    needle_directions = {}
    if center_properties is not None:
        for center in center_properties:
            for needle in center['needles']:
                needle_directions[needle['id']] = needle['direction']
    
    # Create points
    points = []
    for k in range(nk+1):
        for j in range(nj+1):
            for i in range(ni+1):
                x = i * spacing[0] + origin[0]
                y = j * spacing[1] + origin[1]
                z = k * spacing[2] + origin[2]
                points.append([x, y, z])
    points = np.array(points)
    
    def node_index(i, j, k):
        return i + j*(ni+1) + k*(ni+1)*(nj+1)
    
    cells = []
    grain_ids_list = []
    needle_ids_list = []
    euler_phi1_list = []
    euler_Phi_list = []
    euler_phi2_list = []
    
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                elem_nodes = [
                    node_index(i, j, k),
                    node_index(i+1, j, k),
                    node_index(i+1, j+1, k),
                    node_index(i, j+1, k),
                    node_index(i, j, k+1),
                    node_index(i+1, j, k+1),
                    node_index(i+1, j+1, k+1),
                    node_index(i, j+1, k+1)
                ]
                cells.append(elem_nodes)
                grain_ids_list.append(volume[i, j, k])
                
                needle_id = needle_volume[i, j, k] if needle_volume is not None else volume[i, j, k]
                needle_ids_list.append(needle_id)
                
                # Direct Euler angle calculation from c-axis (needle direction)
                if needle_id in needle_directions:
                    c_axis = needle_directions[needle_id]
                    c = c_axis / np.linalg.norm(c_axis)
                    
                    # Spherical coordinates of c-axis
                    phi1 = np.arctan2(c[1], c[0])
                    Phi = np.arccos(np.clip(c[2], -1, 1))
                    
                    # Random phi2 per needle (twist around c-axis)
                    np.random.seed(needle_id)
                    phi2 = np.random.uniform(0, 2*np.pi)
                    
                    phi1 = np.degrees(phi1) % 360
                    Phi = np.degrees(Phi)
                    phi2 = np.degrees(phi2)
                    
                    euler_phi1_list.append(phi1)
                    euler_Phi_list.append(Phi)
                    euler_phi2_list.append(phi2)
                else:
                    euler_phi1_list.append(0.0)
                    euler_Phi_list.append(0.0)
                    euler_phi2_list.append(0.0)
    
    # Create mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("hexahedron", np.array(cells))],
        cell_data={}
    )
    
    meshio.write(filename, mesh, file_format="exodus")
    
    # Add element variables
    exo = nc.Dataset(filename, 'r+')
    
    try:
        num_vars = 5
        exo.createDimension('num_elem_var', num_vars)
        
        name_var = exo.createVariable('name_elem_var', 'S1', ('num_elem_var', 'len_string'))
        var_names = ['grain_id', 'needle_id', 'euler_phi1', 'euler_Phi', 'euler_phi2']
        
        for i, vname in enumerate(var_names):
            name_str = vname.ljust(33, '\x00')
            name_array = np.array([c.encode('utf-8') for c in name_str], dtype='S1')
            name_var[i, :] = name_array
        
        var_data = [grain_ids_list, needle_ids_list, euler_phi1_list, euler_Phi_list, euler_phi2_list]
        
        for var_idx, data in enumerate(var_data):
            var_name = f'vals_elem_var{var_idx+1}eb1'
            var = exo.createVariable(var_name, 'f8', ('time_step', 'num_el_in_blk1'))
            var[0, :] = np.array(data, dtype=np.float64)
        
    finally:
        exo.close()
    
    phi1_arr = np.array(euler_phi1_list)
    Phi_arr = np.array(euler_Phi_list)
    phi2_arr = np.array(euler_phi2_list)
    
    phi1_nonzero = phi1_arr[phi1_arr > 0]
    phi2_nonzero = phi2_arr[phi2_arr > 0]
    
    print(f"Exodus II file written: {filename}")
    print(f"  Total elements: {len(cells)}")
    print(f"  ✓ Euler angles from needle direction (c-axis):")
    print(f"    - phi1: [{phi1_nonzero.min():.1f}, {phi1_nonzero.max():.1f}]°")
    print(f"    - Phi:  [{Phi_arr.min():.1f}, {Phi_arr.max():.1f}]°")
    print(f"    - phi2: [{phi2_nonzero.min():.1f}, {phi2_nonzero.max():.1f}]°")
    
    phi1_unique = len(np.unique(phi1_nonzero))
    phi2_unique = len(np.unique(phi2_nonzero))
    print(f"  ✓ Unique values: phi1={phi1_unique}, phi2={phi2_unique}")
    
    if phi1_unique > 1:
        print(f"  ✓ SUCCESS: phi1 varies across grains!")
    if phi2_unique > 10:
        print(f"  ✓ SUCCESS: phi2 varies across needles!")

def compute_local_coordinate_system(direction):
    """
    Compute complete orthonormal coordinate system from primary direction
    Same logic as Abaqus orientation system
    
    Parameters:
    -----------
    direction : numpy.ndarray
        Primary direction vector (c-axis for aragonite)
    
    Returns:
    --------
    a1, a2, a3 : numpy.ndarray
        Three orthonormal vectors forming the local coordinate system
        a1 = primary direction (c-axis, local z for aragonite)
        a2 = secondary direction (perpendicular to a1)
        a3 = tertiary direction (completes right-handed system)"""
        
    # Primary axis (a1) is the needle direction
    a1 = direction / np.linalg.norm(direction)
    
    # Secondary axis perpendicular to a1
    global_z = np.array([0, 0, 1])
    if np.abs(np.dot(a1, global_z)) > 0.99:
        global_x = np.array([1, 0, 0])
        a2 = np.cross(a1, global_x)
    else:
        a2 = np.cross(a1, global_z)
    
    a2 = a2 / np.linalg.norm(a2)
    
    # Tertiary axis completes system
    a3 = np.cross(a1, a2)
    a3 = a3 / np.linalg.norm(a3)
    
    return a1, a2, a3

def export_vtk_unstructured(volume, needle_volume, domain_size, filename, 
                            center_properties=None, spacing=None, origin=None):
    """
    Export to VTK Unstructured Grid format
    
    NOW INCLUDES ALL THREE EULER ANGLES:
    - phi1 (azimuthal angle)
    - euler_Phi (polar angle) - keeping old name for compatibility
    - phi2 (twist angle) - NEW!
    """
    
    # Get mesh dimensions
    ni, nj, nk = volume.shape
    
    # Set default spacing and origin if not provided
    if spacing is None:
        spacing = (domain_size/ni, domain_size/nj, domain_size/nk)
    if origin is None:
        origin = (0, 0, 0)
    
    # Build lookup for needle directions and compute Euler angles
    needle_properties = {}
    if center_properties is not None:
        for center in center_properties:
            for needle in center['needles']:
                needle_id = needle['id']
                c_axis = needle['direction']
                
                # Normalize
                c = c_axis / np.linalg.norm(c_axis)
                
                # Compute ALL THREE Euler angles (same as Exodus)
                phi1 = np.arctan2(c[1], c[0])
                Phi = np.arccos(np.clip(c[2], -1, 1))
                
                # Random phi2 per needle (deterministic)
                np.random.seed(needle_id)
                phi2 = np.random.uniform(0, 2*np.pi)
                
                # Convert to degrees
                phi1_deg = np.degrees(phi1) % 360
                Phi_deg = np.degrees(Phi)
                phi2_deg = np.degrees(phi2)
                
                needle_properties[needle_id] = {
                    'c_axis': c,
                    'phi1': phi1_deg,
                    'Phi': Phi_deg,
                    'phi2': phi2_deg
                }
    
    with open(filename, 'w') as f:
        # Write header
        f.write("# vtk DataFile Version 4.2\n")
        f.write("Radial Needle Microstructure with Complete Euler Angles\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Write points (nodes)
        num_points = (ni+1) * (nj+1) * (nk+1)
        f.write(f"\nPOINTS {num_points} float\n")
        
        for k in range(nk+1):
            for j in range(nj+1):
                for i in range(ni+1):
                    x = i * spacing[0] + origin[0]
                    y = j * spacing[1] + origin[1]
                    z = k * spacing[2] + origin[2]
                    f.write(f"{x} {y} {z}\n")
        
        # Write cells (elements)
        num_cells = ni * nj * nk
        f.write(f"\nCELLS {num_cells} {num_cells * 9}\n")
        
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    # Node indices (VTK ordering)
                    n0 = i + j*(ni+1) + k*(ni+1)*(nj+1)
                    n1 = n0 + 1
                    n2 = n0 + (ni+1) + 1
                    n3 = n0 + (ni+1)
                    n4 = n0 + (ni+1)*(nj+1)
                    n5 = n4 + 1
                    n6 = n4 + (ni+1) + 1
                    n7 = n4 + (ni+1)
                    f.write(f"8 {n0} {n1} {n2} {n3} {n4} {n5} {n6} {n7}\n")
        
        # Cell types (12 = hexahedron)
        f.write(f"\nCELL_TYPES {num_cells}\n")
        for _ in range(num_cells):
            f.write("12\n")
        
        # ====================================================================
        # CELL DATA
        # ====================================================================
        f.write(f"\nCELL_DATA {num_cells}\n")
        
        # Grain IDs
        f.write("\nSCALARS grain_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    f.write(f"{volume[i,j,k]}\n")
        
        # Needle IDs
        f.write("\nSCALARS needle_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nk):
            for j in range(nj):
                for i in range(ni):
                    needle_id = needle_volume[i,j,k] if needle_volume is not None else 0
                    f.write(f"{needle_id}\n")
        
        # C-axis (needle direction) components
        for comp_idx, comp_name in enumerate(['x', 'y', 'z']):
            f.write(f"\nSCALARS c_axis_{comp_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        needle_id = needle_volume[i,j,k] if needle_volume is not None else 0
                        if needle_id in needle_properties:
                            value = needle_properties[needle_id]['c_axis'][comp_idx]
                            f.write(f"{value:.6f}\n")
                        else:
                            f.write("0.0\n")
        
        # ====================================================================
        # EULER ANGLES - ALL THREE NOW!
        # ====================================================================
        
        # phi1 (azimuthal angle) - NEW!
        if needle_properties:
            f.write("\nSCALARS euler_phi1 float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        needle_id = needle_volume[i,j,k] if needle_volume is not None else 0
                        if needle_id in needle_properties:
                            f.write(f"{needle_properties[needle_id]['phi1']:.6f}\n")
                        else:
                            f.write("0.0\n")
        
        # euler_Phi (polar angle) - EXISTING
        if needle_properties:
            f.write("\nSCALARS euler_Phi float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        needle_id = needle_volume[i,j,k] if needle_volume is not None else 0
                        if needle_id in needle_properties:
                            f.write(f"{needle_properties[needle_id]['Phi']:.6f}\n")
                        else:
                            f.write("0.0\n")
        
        # phi2 (twist angle) - NEW!
        if needle_properties:
            f.write("\nSCALARS euler_phi2 float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        needle_id = needle_volume[i,j,k] if needle_volume is not None else 0
                        if needle_id in needle_properties:
                            f.write(f"{needle_properties[needle_id]['phi2']:.6f}\n")
                        else:
                            f.write("0.0\n")
        
        # angle_from_z (backward compatibility)
        if needle_properties:
            f.write("\nSCALARS angle_from_z float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        needle_id = needle_volume[i,j,k] if needle_volume is not None else 0
                        if needle_id in needle_properties:
                            f.write(f"{needle_properties[needle_id]['Phi']:.6f}\n")
                        else:
                            f.write("0.0\n")
        
        # Point data for boundaries (unchanged)
        f.write(f"\nPOINT_DATA {num_points}\n")
        f.write("SCALARS boundary_flag int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nk+1):
            for j in range(nj+1):
                for i in range(ni+1):
                    flag = 0
                    if k == 0: flag = 1
                    elif k == nk: flag = 2
                    elif i == 0: flag = 3
                    elif i == ni: flag = 4
                    elif j == 0: flag = 5
                    elif j == nj: flag = 6
                    f.write(f"{flag}\n")
    
    if needle_properties:
        # Get statistics
        phi1_vals = [p['phi1'] for p in needle_properties.values()]
        Phi_vals = [p['Phi'] for p in needle_properties.values()]
        phi2_vals = [p['phi2'] for p in needle_properties.values()]
        
        print(f"VTK file written: {filename}")
        print(f"  ✓ Complete Euler angles exported:")
        print(f"    - euler_phi1: [{min(phi1_vals):.1f}, {max(phi1_vals):.1f}]°")
        print(f"    - euler_Phi:  [{min(Phi_vals):.1f}, {max(Phi_vals):.1f}]°")
        print(f"    - euler_phi2: [{min(phi2_vals):.1f}, {max(phi2_vals):.1f}]°")
        print(f"  ✓ Direction vectors: c_axis_x/y/z")
        print(f"  ✓ Same Euler angles as Exodus (compatible!)")
    else:
        print(f"VTK file written: {filename}")
        
def visualize_slices(volume, domain_size, filename_prefix="microstructure"):
    """
    Visualize and save slices through the microstructure
    """
    ni, nj, nk = volume.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY slice (middle Z)
    axes[0].imshow(volume[:, :, nk//2].T, origin='lower', cmap='tab20')
    axes[0].set_title('XY Slice (mid-Z)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # XZ slice (middle Y)
    axes[1].imshow(volume[:, nj//2, :].T, origin='lower', cmap='tab20')
    axes[1].set_title('XZ Slice (mid-Y)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    # YZ slice (middle X)
    axes[2].imshow(volume[ni//2, :, :].T, origin='lower', cmap='tab20')
    axes[2].set_title('YZ Slice (mid-X)')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_slices.png', dpi=150)
    plt.close()
    print(f"Visualization saved: {filename_prefix}_slices.png")


# Main execution example
if __name__ == "__main__":
    
    # ========================================================================
    # PARAMETERS - Adjust these for your needs
    # ========================================================================
    
    domain_size = 200  # Physical size (e.g., microns)
    num_centers = 45   # Number of nucleation centers (grains)
    needle_length_range = (20, 40)  # Length range for needles
    needles_per_center_range = (10, 35)  # Number of needles per grain
    resolution = 100  # Voxel resolution
    quasi_2d = True   # True for quasi-2D structure, False for full 3D
    
    output_prefix = "enhanced_microstructure"
    
    # Material properties for Abaqus
    material_props = {
        'needle_material': {
            'name': 'Aragonite',
            'type': 'orthotropic',
            # E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
            'constants': [38.617e3, 52.212e3, 118.645e3, 0.498, 0.232, 0.106, 
                         34.069e3, 37.36e3, 23.996e3]
        }
    }
    
    # ========================================================================
    # GENERATE MICROSTRUCTURE
    # ========================================================================
    
    print("="*70)
    print("GENERATING RADIAL NEEDLE MICROSTRUCTURE")
    print("="*70)
    print(f"Domain size: {domain_size}")
    print(f"Resolution: {resolution}x{resolution}x{resolution}")
    print(f"Number of grains: {num_centers}")
    print(f"Quasi-2D: {quasi_2d}")
    print()
    
    volume, needle_volume, center_properties, mesh = radial_needles_more_2d(
        num_centers, domain_size, needle_length_range, needles_per_center_range, 
        resolution, z_constraint_factor=0.1, quasi_2d=quasi_2d
    )
    
    # Check for empty spaces
    empty_count = np.sum(volume == 0)
    print(f"\nEmpty voxels: {empty_count} out of {np.prod(volume.shape)}")
    print(f"Unique grains: {len(np.unique(volume)) - 1}")  # -1 to exclude 0
    print(f"Unique needles: {len(np.unique(needle_volume)) - 1}")
    
    # ========================================================================
    # EXPORT TO DIFFERENT FORMATS
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPORTING TO FE FORMATS")
    print("="*70)
    
    # Export to Abaqus INP
    print("\n1. Abaqus INP format...")
    export_to_abaqus_enhanced(
        volume, needle_volume, center_properties, domain_size,
        f"{output_prefix}.inp",
        material_properties=material_props
    )
    
    # Export to VTK
    print("\n2. VTK unstructured grid format...")
    export_vtk_unstructured(
        volume, needle_volume, domain_size,
        f"{output_prefix}.vtk",
        center_properties=center_properties  # Add orientation data
    )
    
    # Export to Exodus II (if meshio available)
    print("\n3. Exodus II format...")
    try:
        export_to_exodus(
            volume, needle_volume, domain_size,
            f"{output_prefix}.e",
            center_properties=center_properties  # Add orientation data
        )
    except Exception as e:
        print(f"Could not export to Exodus format: {e}")
    
    # Save numpy arrays for later use
    print("\n4. Saving numpy arrays...")
    np.save(f'{output_prefix}_grain_id.npy', volume)
    np.save(f'{output_prefix}_needle_id.npy', needle_volume)
    print(f"  - {output_prefix}_grain_id.npy")
    print(f"  - {output_prefix}_needle_id.npy")
    
    # Create visualization
    print("\n5. Creating visualization...")
    visualize_slices(volume, domain_size, output_prefix)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)