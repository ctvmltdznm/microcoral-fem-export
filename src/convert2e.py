#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Mesh Converter to Exodus Format - PROPERLY ORGANIZED VERSION
Reorganizes elements into separate blocks by grain_id/needle_id for Paraview coloring

Author: Nikolai
Date: 2025
"""

import numpy as np
import meshio
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


class MeshConverter:
    """Convert various mesh formats to Exodus with proper element block organization"""
    
    def __init__(self, input_file, output_file=None, verbose=True, organize_by=None):
        self.input_file = Path(input_file)
        self.verbose = verbose
        self.organize_by = organize_by  # 'grain_id', 'needle_id', or None
        
        if output_file is None:
            self.output_file = self.input_file.with_suffix('.e')
        else:
            self.output_file = Path(output_file)
            if not self.output_file.suffix in ['.e', '.exo']:
                self.output_file = self.output_file.with_suffix('.e')
        
        self.mesh = None
        
    def log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    def detect_format(self):
        """Detect input file format"""
        suffix = self.input_file.suffix.lower()
        
        if suffix in ['.vtk', '.vtu', '.vti']:
            return 'vtk'
        elif suffix == '.feb':
            return 'febio'
        elif suffix == '.inp':
            return 'abaqus'
        else:
            raise ValueError(f"Unknown file format: {suffix}")
    
    def reorganize_by_field(self, mesh, field_name):
        """
        Reorganize mesh so each unique value of field_name gets its own element block
        This is the KEY to making Paraview coloring work properly
        """
        self.log(f"\nReorganizing elements by '{field_name}'...")
        
        # Collect all cells and their field values
        all_cells = []
        all_field_values = []
        
        # Flatten cell blocks
        for block_idx, cell_block in enumerate(mesh.cells):
            cell_type = cell_block.type
            cell_data = cell_block.data
            
            # Get field values for this block
            if field_name in mesh.cell_data:
                field_values = mesh.cell_data[field_name][block_idx]
            else:
                self.log(f"  Warning: Field '{field_name}' not found in cell_data")
                return mesh
            
            for i, cell_conn in enumerate(cell_data):
                all_cells.append({
                    'type': cell_type,
                    'connectivity': cell_conn,
                    field_name: field_values[i]
                })
                all_field_values.append(field_values[i])
        
        self.log(f"  Total cells before reorganization: {len(all_cells)}")
        
        # Get unique field values
        unique_values = np.unique(all_field_values)
        self.log(f"  Unique {field_name} values: {len(unique_values)}")
        self.log(f"  Range: [{unique_values.min()}, {unique_values.max()}]")
        
        # Reorganize cells into blocks by field value
        new_cells = []
        new_cell_data = {field_name: []}
        
        for value in unique_values:
            # Get all cells with this field value
            cells_with_value = [c for c in all_cells if c[field_name] == value]
            
            if not cells_with_value:
                continue
            
            # Assume all cells are same type (typical for structured grids)
            cell_type = cells_with_value[0]['type']
            connectivity_list = [c['connectivity'] for c in cells_with_value]
            
            # Create new block
            new_cells.append((cell_type, np.array(connectivity_list)))
            new_cell_data[field_name].append(np.full(len(connectivity_list), value, dtype=np.int32))
            
            self.log(f"  Block {field_name}={value}: {len(connectivity_list)} elements")
        
        # Preserve other cell data fields
        for other_field in mesh.cell_data.keys():
            if other_field != field_name:
                new_cell_data[other_field] = []
                
                # Reconstruct other field data organized by new blocks
                for value in unique_values:
                    cells_with_value = [c for c in all_cells if c[field_name] == value]
                    if cells_with_value:
                        # Get the other field values for these cells
                        other_values = []
                        cell_idx = 0
                        for block_idx, cell_block in enumerate(mesh.cells):
                            block_size = len(cell_block.data)
                            if other_field in mesh.cell_data:
                                other_field_data = mesh.cell_data[other_field][block_idx]
                                for i in range(block_size):
                                    if all_cells[cell_idx][field_name] == value:
                                        other_values.append(other_field_data[i])
                                    cell_idx += 1
                        
                        if other_values:
                            new_cell_data[other_field].append(np.array(other_values, dtype=np.int32))
        
        # Create new mesh with reorganized blocks
        new_mesh = meshio.Mesh(
            points=mesh.points,
            cells=new_cells,
            cell_data=new_cell_data,
            point_data=mesh.point_data if hasattr(mesh, 'point_data') else None
        )
        
        self.log(f"  ✓ Reorganized into {len(new_cells)} element blocks")
        
        return new_mesh
    
    def convert_vtk_to_exodus(self):
        """Convert VTK file to Exodus using meshio"""
        self.log(f"Reading VTK file: {self.input_file}")
        
        try:
            # Read VTK file
            mesh = meshio.read(self.input_file)
            
            self.log(f"  Points: {len(mesh.points)}")
            
            # Log cell information
            total_cells = 0
            for cell_block in mesh.cells:
                block_size = len(cell_block.data)
                total_cells += block_size
                self.log(f"  Cell block ({cell_block.type}): {block_size} elements")
            
            # Check for cell data
            if mesh.cell_data:
                self.log(f"  Cell data fields found: {list(mesh.cell_data.keys())}")
                
                # Show info about each field
                for field_name in mesh.cell_data.keys():
                    field_data = mesh.cell_data[field_name]
                    if isinstance(field_data, list):
                        for block_idx, block_data in enumerate(field_data):
                            unique_vals = np.unique(block_data)
                            self.log(f"    {field_name} (block {block_idx}): {len(unique_vals)} unique values, range [{unique_vals.min()}, {unique_vals.max()}]")
            else:
                self.log(f"  Warning: No cell data found in VTK file")
                
            # Auto-detect which field to organize by
            if self.organize_by is None and mesh.cell_data:
                # Prefer needle_id over grain_id if both exist
                if 'needle_id' in mesh.cell_data:
                    self.organize_by = 'needle_id'
                elif 'grain_id' in mesh.cell_data:
                    self.organize_by = 'grain_id'
                else:
                    # Use first available field
                    self.organize_by = list(mesh.cell_data.keys())[0]
            
            # Reorganize mesh by the chosen field
            if self.organize_by and self.organize_by in mesh.cell_data:
                mesh = self.reorganize_by_field(mesh, self.organize_by)
            elif self.organize_by:
                self.log(f"  Warning: Field '{self.organize_by}' not found in mesh")
            
            # Write to Exodus
            self.log(f"\nWriting Exodus file: {self.output_file}")
            meshio.write(self.output_file, mesh, file_format="exodus")
            
            self.log("✓ Conversion successful!")
            self.log(f"\nIn Paraview:")
            self.log(f"  1. Open {self.output_file}")
            self.log(f"  2. Click 'Apply'")
            self.log(f"  3. In the toolbar, change 'Solid Color' to '{self.organize_by}'")
            self.log(f"  4. You should see {len(mesh.cells)} different colors/blocks")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Error converting VTK: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def parse_febio_xml(self):
        """Parse FEBio XML file and extract mesh data"""
        self.log(f"Reading FEBio file: {self.input_file}")
        
        try:
            tree = ET.parse(self.input_file)
            root = tree.getroot()
            
            geometry = root.find('Geometry')
            if geometry is None:
                raise ValueError("No Geometry section found in FEBio file")
            
            # Parse nodes
            nodes_elem = geometry.find('Nodes')
            if nodes_elem is None:
                raise ValueError("No Nodes found in Geometry section")
            
            points = []
            node_ids = {}
            
            for node in nodes_elem.findall('node'):
                node_id = int(node.get('id'))
                coords = [float(x) for x in node.text.split(',')]
                node_ids[node_id] = len(points)
                points.append(coords)
            
            points = np.array(points)
            self.log(f"  Parsed {len(points)} nodes")
            
            # Parse elements
            elements_list = geometry.findall('Elements')
            
            cells = []
            cell_data_arrays = []
            
            for elem_block in elements_list:
                elem_type = elem_block.get('type', 'hex8')
                
                type_map = {
                    'hex8': 'hexahedron',
                    'hex20': 'hexahedron20',
                    'hex27': 'hexahedron27',
                    'tet4': 'tetra',
                    'tet10': 'tetra10',
                    'penta6': 'wedge',
                }
                
                meshio_type = type_map.get(elem_type, 'hexahedron')
                
                block_elements = []
                block_mat_ids = []
                
                for elem in elem_block.findall('elem'):
                    mat_id = int(elem.get('mat', 1))
                    
                    conn_text = elem.text.strip()
                    conn = [int(x) for x in conn_text.split(',')]
                    
                    # Convert to 0-based indexing
                    conn_0based = [node_ids[c] for c in conn]
                    
                    block_elements.append(conn_0based)
                    block_mat_ids.append(mat_id)
                
                if block_elements:
                    cells.append((meshio_type, np.array(block_elements)))
                    cell_data_arrays.append(np.array(block_mat_ids, dtype=np.int32))
                    self.log(f"  Block: {len(block_elements)} {elem_type} elements")
            
            # Format cell data properly
            cell_data = {
                'material_id': cell_data_arrays
            }
            
            self.mesh = meshio.Mesh(
                points=points,
                cells=cells,
                cell_data=cell_data
            )
            
            return True
            
        except Exception as e:
            self.log(f"✗ Error parsing FEBio file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def parse_abaqus_inp(self):
        """Parse Abaqus INP file"""
        self.log(f"Reading Abaqus file: {self.input_file}")
        
        try:
            with open(self.input_file, 'r') as f:
                lines = f.readlines()
            
            points = []
            node_map = {}
            cells_dict = {}
            elem_ids_dict = {}
            
            mode = None
            elem_type = None
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if not line or line.startswith('**'):
                    i += 1
                    continue
                
                if line.startswith('*'):
                    line_upper = line.upper()
                    
                    if line_upper.startswith('*NODE'):
                        mode = 'nodes'
                        
                    elif line_upper.startswith('*ELEMENT'):
                        mode = 'elements'
                        
                        if 'TYPE=' in line_upper:
                            elem_type = line_upper.split('TYPE=')[1].split(',')[0].strip()
                            if elem_type not in cells_dict:
                                cells_dict[elem_type] = []
                                elem_ids_dict[elem_type] = []
                    else:
                        mode = None
                    
                    i += 1
                    continue
                
                if mode == 'nodes':
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 4:
                        node_id = int(parts[0])
                        coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                        node_map[node_id] = len(points)
                        points.append(coords)
                
                elif mode == 'elements':
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 2:
                        elem_id = int(parts[0])
                        connectivity = [int(x) for x in parts[1:] if x]
                        
                        # Handle multi-line elements
                        while i + 1 < len(lines) and not lines[i + 1].strip().startswith('*'):
                            i += 1
                            cont_line = lines[i].strip()
                            if cont_line:
                                cont_parts = [int(x.strip()) for x in cont_line.split(',') if x.strip()]
                                connectivity.extend(cont_parts)
                            else:
                                break
                        
                        conn_0based = [node_map[nid] for nid in connectivity]
                        
                        cells_dict[elem_type].append(conn_0based)
                        elem_ids_dict[elem_type].append(elem_id)
                
                i += 1
            
            points = np.array(points)
            self.log(f"  Parsed {len(points)} nodes")
            
            # Convert to meshio format
            abaqus_to_meshio = {
                'C3D8': 'hexahedron',
                'C3D8R': 'hexahedron',
                'C3D20': 'hexahedron20',
                'C3D4': 'tetra',
                'C3D10': 'tetra10',
            }
            
            cells = []
            elem_id_arrays = []
            
            for abq_type, elements in cells_dict.items():
                meshio_type = abaqus_to_meshio.get(abq_type.upper(), 'hexahedron')
                if elements:
                    cells.append((meshio_type, np.array(elements)))
                    elem_id_arrays.append(np.array(elem_ids_dict[abq_type], dtype=np.int32))
                    self.log(f"  Found {len(elements)} {abq_type} elements")
            
            # Create cell data
            cell_data = {
                'element_id': elem_id_arrays
            }
            
            self.mesh = meshio.Mesh(
                points=points,
                cells=cells,
                cell_data=cell_data
            )
            
            return True
            
        except Exception as e:
            self.log(f"✗ Error parsing Abaqus file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def write_exodus(self):
        """Write the parsed mesh to Exodus format"""
        if self.mesh is None:
            self.log("✗ No mesh data to write")
            return False
        
        try:
            self.log(f"Writing Exodus file: {self.output_file}")
            meshio.write(self.output_file, self.mesh, file_format="exodus")
            self.log("✓ Conversion successful!")
            
            if self.mesh.cell_data:
                self.log(f"\nAvailable fields in Paraview:")
                for field_name in self.mesh.cell_data.keys():
                    self.log(f"  - {field_name}")
            
            return True
        except Exception as e:
            self.log(f"✗ Error writing Exodus file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def convert(self):
        """Main conversion function"""
        format_type = self.detect_format()
        self.log(f"Detected format: {format_type}\n")
        
        success = False
        
        if format_type == 'vtk':
            success = self.convert_vtk_to_exodus()
        elif format_type == 'febio':
            if self.parse_febio_xml():
                success = self.write_exodus()
        elif format_type == 'abaqus':
            if self.parse_abaqus_inp():
                success = self.write_exodus()
        
        return success


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Convert VTK, FEBio, or Abaqus mesh files to Exodus II format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect field to organize by
  python mesh_to_exodus_v2.py microstructure.vtk
  
  # Organize by specific field
  python mesh_to_exodus_v2.py microstructure.vtk --organize-by needle_id
  
  # With custom output name
  python mesh_to_exodus_v2.py model.vtk -o output.e --organize-by grain_id
        """
    )
    
    parser.add_argument('input', help='Input mesh file')
    parser.add_argument('-o', '--output', help='Output Exodus file')
    parser.add_argument('--organize-by', help='Field name to organize blocks by (e.g., grain_id, needle_id)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"✗ Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    converter = MeshConverter(args.input, args.output, 
                             verbose=not args.quiet,
                             organize_by=args.organize_by)
    
    try:
        success = converter.convert()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()