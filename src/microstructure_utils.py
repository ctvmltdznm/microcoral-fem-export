# -*- coding: utf-8 -*-
"""
Utility Functions for Microstructure Export
Helper functions for common modifications and analysis

@author: nk03
"""

import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(phi1, Phi, phi2, convention='bunge'):
    """
    Convert Euler angles to rotation matrix
    
    Parameters:
    -----------
    phi1, Phi, phi2 : float
        Euler angles in radians
    convention : str
        'bunge' for Bunge convention (default for materials science)
        'zxz' for ZXZ convention
    
    Returns:
    --------
    R : np.ndarray
        3x3 rotation matrix
    """
    if convention == 'bunge':
        # Bunge convention: Z-X-Z
        R = Rotation.from_euler('ZXZ', [phi1, Phi, phi2]).as_matrix()
    else:
        R = Rotation.from_euler(convention, [phi1, Phi, phi2]).as_matrix()
    
    return R


def generate_random_texture(num_grains, texture_type='random'):
    """
    Generate texture (orientations) for polycrystalline structure
    
    Parameters:
    -----------
    num_grains : int
        Number of grains
    texture_type : str
        'random': Random orientations (uniform in orientation space)
        'cube': Cube texture ({001}<100>)
        'goss': Goss texture ({110}<001>)
    
    Returns:
    --------
    orientations : list of tuples
        List of (phi1, Phi, phi2) Euler angles in radians
    """
    orientations = []
    
    if texture_type == 'random':
        for _ in range(num_grains):
            phi1 = np.random.uniform(0, 2*np.pi)
            Phi = np.arccos(np.random.uniform(-1, 1))
            phi2 = np.random.uniform(0, 2*np.pi)
            orientations.append((phi1, Phi, phi2))
    
    elif texture_type == 'cube':
        # Cube texture with some scatter
        for _ in range(num_grains):
            scatter = np.deg2rad(15)  # 15 degree scatter
            phi1 = np.random.normal(0, scatter)
            Phi = np.random.normal(0, scatter)
            phi2 = np.random.normal(0, scatter)
            orientations.append((phi1, Phi, phi2))
    
    elif texture_type == 'goss':
        # Goss texture {110}<001>
        for _ in range(num_grains):
            scatter = np.deg2rad(10)
            phi1 = np.random.normal(0, scatter)
            Phi = np.random.normal(np.pi/2, scatter)
            phi2 = np.random.normal(np.pi/4, scatter)
            orientations.append((phi1, Phi, phi2))
    
    return orientations


def find_grain_boundary_faces(volume):
    """
    Find faces between different grains
    
    Parameters:
    -----------
    volume : np.ndarray
        3D array of grain IDs
    
    Returns:
    --------
    boundary_faces : list
        List of tuples: (element1_ijk, element2_ijk, face_normal, grain1_id, grain2_id)
    """
    ni, nj, nk = volume.shape
    boundary_faces = []
    
    # Check all internal faces
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                grain1 = volume[i, j, k]
                
                # Check +X face
                if i < ni-1:
                    grain2 = volume[i+1, j, k]
                    if grain1 != grain2 and grain1 > 0 and grain2 > 0:
                        boundary_faces.append(((i,j,k), (i+1,j,k), 'X', grain1, grain2))
                
                # Check +Y face
                if j < nj-1:
                    grain2 = volume[i, j+1, k]
                    if grain1 != grain2 and grain1 > 0 and grain2 > 0:
                        boundary_faces.append(((i,j,k), (i,j+1,k), 'Y', grain1, grain2))
                
                # Check +Z face
                if k < nk-1:
                    grain2 = volume[i, j, k+1]
                    if grain1 != grain2 and grain1 > 0 and grain2 > 0:
                        boundary_faces.append(((i,j,k), (i,j,k+1), 'Z', grain1, grain2))
    
    return boundary_faces


def compute_misorientation(R1, R2):
    """
    Compute misorientation angle between two orientations
    
    Parameters:
    -----------
    R1, R2 : np.ndarray
        3x3 rotation matrices
    
    Returns:
    --------
    angle : float
        Misorientation angle in radians
    """
    # Compute relative rotation
    delta_R = R2 @ R1.T
    
    # Misorientation angle from trace
    trace = np.trace(delta_R)
    angle = np.arccos((trace - 1) / 2)
    
    return angle


def assign_cohesive_properties(misorientation_angle):
    """
    Assign cohesive zone properties based on grain boundary character
    
    Parameters:
    -----------
    misorientation_angle : float
        Misorientation angle in radians
    
    Returns:
    --------
    properties : dict
        Cohesive zone material properties
    """
    # Example: Weaker GB for high-angle boundaries
    angle_deg = np.rad2deg(misorientation_angle)
    
    if angle_deg < 15:
        # Low-angle GB (stronger)
        return {
            'stiffness': 1e6,
            'strength': 1000,
            'toughness': 100
        }
    else:
        # High-angle GB (weaker)
        return {
            'stiffness': 5e5,
            'strength': 500,
            'toughness': 50
        }


def create_abaqus_cohesive_section(filename, boundary_faces, grain_orientations):
    """
    Create Abaqus input section for cohesive zones at grain boundaries
    
    Parameters:
    -----------
    filename : str
        Output filename
    boundary_faces : list
        List of boundary faces from find_grain_boundary_faces
    grain_orientations : dict
        Dictionary mapping grain_id to rotation matrix
    """
    
    with open(filename, 'w') as f:
        f.write("** ======================================================\n")
        f.write("** COHESIVE ZONE MODEL FOR GRAIN BOUNDARIES\n")
        f.write("** ======================================================\n")
        f.write("**\n")
        
        # Group faces by misorientation
        low_angle_faces = []
        high_angle_faces = []
        
        for face_data in boundary_faces:
            grain1_id, grain2_id = face_data[3], face_data[4]
            
            if grain1_id in grain_orientations and grain2_id in grain_orientations:
                R1 = grain_orientations[grain1_id]
                R2 = grain_orientations[grain2_id]
                
                misori = compute_misorientation(R1, R2)
                
                if np.rad2deg(misori) < 15:
                    low_angle_faces.append(face_data)
                else:
                    high_angle_faces.append(face_data)
        
        # Write cohesive element definitions
        f.write("** Low-angle grain boundaries\n")
        f.write("*Material, name=GB_LowAngle\n")
        f.write("*Elastic, type=TRACTION\n")
        f.write("1e6, 1e6, 1e6\n")
        f.write("*Damage Initiation, criterion=MAXS\n")
        f.write("1000, 1000, 1000\n")
        f.write("*Damage Evolution, type=ENERGY\n")
        f.write("100\n")
        f.write("**\n")
        
        f.write("** High-angle grain boundaries\n")
        f.write("*Material, name=GB_HighAngle\n")
        f.write("*Elastic, type=TRACTION\n")
        f.write("5e5, 5e5, 5e5\n")
        f.write("*Damage Initiation, criterion=MAXS\n")
        f.write("500, 500, 500\n")
        f.write("*Damage Evolution, type=ENERGY\n")
        f.write("50\n")
        f.write("**\n")
        
        print(f"Created cohesive zone section: {filename}")
        print(f"  Low-angle GBs: {len(low_angle_faces)}")
        print(f"  High-angle GBs: {len(high_angle_faces)}")


def add_orientations_to_abaqus(inp_filename, grain_ids, texture_type='random'):
    """
    Add orientation definitions to existing Abaqus INP file
    
    Parameters:
    -----------
    inp_filename : str
        Existing Abaqus INP file
    grain_ids : list
        List of grain IDs
    texture_type : str
        Type of texture to generate
    """
    # Generate orientations
    orientations = generate_random_texture(len(grain_ids), texture_type)
    
    # Read existing file
    with open(inp_filename, 'r') as f:
        lines = f.readlines()
    
    # Find insertion point (after material definitions)
    insert_idx = 0
    for i, line in enumerate(lines):
        if '*Material' in line:
            insert_idx = i + 1
            while insert_idx < len(lines) and not lines[insert_idx].startswith('*'):
                insert_idx += 1
    
    # Create orientation cards
    orientation_cards = []
    for grain_id, (phi1, Phi, phi2) in zip(grain_ids, orientations):
        R = euler_to_rotation_matrix(phi1, Phi, phi2)
        
        orientation_cards.append(f"*Orientation, name=OrientGrain{grain_id}\n")
        orientation_cards.append(f"{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}, "
                                f"{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}\n")
        orientation_cards.append("1, 0\n")
    
    # Insert orientations
    lines = lines[:insert_idx] + orientation_cards + lines[insert_idx:]
    
    # Update solid sections to use orientations
    for i, line in enumerate(lines):
        if '*Solid Section, elset=Grain' in line and 'orientation=' not in line:
            # Extract grain number
            grain_num = line.split('Grain')[1].split(',')[0]
            # Add orientation reference
            lines[i] = line.rstrip() + f", orientation=OrientGrain{grain_num}\n"
    
    # Write modified file
    output_filename = inp_filename.replace('.inp', '_with_orientations.inp')
    with open(output_filename, 'w') as f:
        f.writelines(lines)
    
    print(f"Added orientations to: {output_filename}")
    return output_filename


def analyze_microstructure_stats(volume, needle_volume):
    """
    Compute statistics about the microstructure
    
    Parameters:
    -----------
    volume : np.ndarray
        Grain ID array
    needle_volume : np.ndarray
        Needle ID array
    
    Returns:
    --------
    stats : dict
        Dictionary of microstructure statistics
    """
    stats = {}
    
    # Number of grains and needles
    grain_ids = np.unique(volume)
    grain_ids = grain_ids[grain_ids > 0]  # Remove background
    needle_ids = np.unique(needle_volume)
    needle_ids = needle_ids[needle_ids > 0]
    
    stats['num_grains'] = len(grain_ids)
    stats['num_needles'] = len(needle_ids)
    
    # Grain sizes
    grain_sizes = []
    for gid in grain_ids:
        size = np.sum(volume == gid)
        grain_sizes.append(size)
    
    stats['mean_grain_size'] = np.mean(grain_sizes)
    stats['std_grain_size'] = np.std(grain_sizes)
    stats['min_grain_size'] = np.min(grain_sizes)
    stats['max_grain_size'] = np.max(grain_sizes)
    
    # Needle sizes
    needle_sizes = []
    for nid in needle_ids:
        size = np.sum(needle_volume == nid)
        needle_sizes.append(size)
    
    stats['mean_needle_size'] = np.mean(needle_sizes)
    stats['std_needle_size'] = np.std(needle_sizes)
    
    # Grain boundary area (approximate)
    boundary_faces = find_grain_boundary_faces(volume)
    stats['num_gb_faces'] = len(boundary_faces)
    
    return stats


def export_orientation_data(center_properties, filename):
    """
    Export needle orientation data for analysis
    
    Parameters:
    -----------
    center_properties : list
        List of center properties from microstructure generation
    filename : str
        Output CSV filename
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['grain_id', 'needle_id', 'dir_x', 'dir_y', 'dir_z', 
                        'length', 'width', 'aspect_ratio'])
        
        for center in center_properties:
            grain_id = center['id']
            for needle in center['needles']:
                writer.writerow([
                    grain_id,
                    needle['id'],
                    needle['direction'][0],
                    needle['direction'][1],
                    needle['direction'][2],
                    needle['length'],
                    needle['width'],
                    needle['aspect_ratio']
                ])
    
    print(f"Orientation data exported: {filename}")


# Example usage
if __name__ == "__main__":
    
    print("="*70)
    print("UTILITY FUNCTIONS DEMO")
    print("="*70)
    
    # Example 1: Generate random texture
    print("\n1. Generating random texture for 10 grains...")
    orientations = generate_random_texture(10, texture_type='random')
    print(f"   Generated {len(orientations)} orientations")
    
    # Example 2: Convert to rotation matrices
    print("\n2. Converting Euler angles to rotation matrices...")
    for i, (phi1, Phi, phi2) in enumerate(orientations[:3]):
        R = euler_to_rotation_matrix(phi1, Phi, phi2)
        print(f"   Grain {i+1}:")
        print(f"   Euler angles: φ1={np.rad2deg(phi1):.1f}°, Φ={np.rad2deg(Phi):.1f}°, φ2={np.rad2deg(phi2):.1f}°")
        print(f"   Rotation matrix determinant: {np.linalg.det(R):.6f} (should be 1)")
    
    # Example 3: Compute misorientation
    print("\n3. Computing misorientation between first two grains...")
    R1 = euler_to_rotation_matrix(*orientations[0])
    R2 = euler_to_rotation_matrix(*orientations[1])
    misori = compute_misorientation(R1, R2)
    print(f"   Misorientation angle: {np.rad2deg(misori):.2f}°")
    
    # Example 4: Assign cohesive properties
    print("\n4. Assigning cohesive properties based on misorientation...")
    props = assign_cohesive_properties(misori)
    print(f"   Stiffness: {props['stiffness']}")
    print(f"   Strength: {props['strength']}")
    print(f"   Toughness: {props['toughness']}")
    
    print("\n" + "="*70)
    print("Use these functions in your workflow!")
    print("="*70)
