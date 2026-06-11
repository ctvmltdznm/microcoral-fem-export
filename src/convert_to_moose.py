#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_moose.py
===================
Convert a single-block Exodus II file (output of the microstructure generator)
into a MOOSE-ready multi-block Exodus file, and produce all the supporting
files needed to run a CZM simulation automatically.

What this script does
---------------------
1.  Reads the single-block Exodus file with element variables
    (grain_id, needle_id, euler_phi1, euler_Phi, euler_phi2).

2.  Reorganises elements into one block per needle_id (default) or grain_id.
    Element variables are preserved in every block.

3.  Detects all internal interfaces: pairs of adjacent blocks that share at
    least one face.  For each interface, determines the type:
        - intra_grain : both blocks belong to the same sclerodermite (grain_id)
        - inter_grain : blocks from different sclerodermites

4.  Writes mesh_info.json  — mesh bounds + block→grain mapping.
    Bounds are used to generate ParsedGenerateSideset blocks in MOOSE.

5.  Writes interface_map.json  — every block-pair with type, grain IDs,
    shared face count.

6.  Writes a MOOSE input snippet  (.i file fragment) containing:
        [Mesh] block with FileMeshGenerator, ParsedGenerateSideset
               (6 external faces), and BreakMeshByBlockGenerator
        [InterfaceKernels] stubs — one CZM kernel per interface type
        [BCs] stubs — 6 named external sidesets

Usage
-----
    python convert_to_moose.py input.e output.e [options]

    --organize-by   needle_id | grain_id          (default: needle_id)
    --force                                         overwrite output files
    --no-moose-snippet                              skip .i file generation
    --czm-intra     intra-grain CZM material name  (default: CZM_IntraGrain)
    --czm-inter     inter-grain CZM material name  (default: CZM_InterGrain)

Examples
--------
    # Needle-level interfaces (default)
    python convert_to_moose.py tiny_aragonite.e tiny_aragonite_moose.e

    # Grain-level interfaces only
    python convert_to_moose.py tiny_aragonite.e tiny_aragonite_moose.e \\
        --organize-by grain_id

    # Custom CZM material names
    python convert_to_moose.py input.e output.e \\
        --czm-intra CZM_Protein --czm-inter CZM_Water

Author: nk03 / pipeline
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import meshio
import netCDF4 as nc
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Hex8 face definitions  (local node indices, MOOSE/Exodus convention)
# ──────────────────────────────────────────────────────────────────────────────
HEX8_FACES = [
    (0, 1, 2, 3),   # bottom  (z-)
    (4, 5, 6, 7),   # top     (z+)
    (0, 1, 5, 4),   # front   (y-)
    (2, 3, 7, 6),   # back    (y+)
    (0, 3, 7, 4),   # left    (x-)
    (1, 2, 6, 5),   # right   (x+)
]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Read mesh and element variables
# ──────────────────────────────────────────────────────────────────────────────

def read_exodus(input_file: str) -> dict:
    """
    Read single-block Exodus file via meshio.
    Returns a dict with: points, connectivity, cell_type, elem_vars.
    elem_vars keys: whatever is in the file (grain_id, needle_id, euler_*)
    """
    print(f"Reading {input_file}...")
    mesh = meshio.read(input_file)

    if len(mesh.cells) == 0:
        raise ValueError("No cells found in mesh.")

    # Flatten all cell blocks into one (generator writes single block)
    all_cells = np.vstack([b.data for b in mesh.cells])
    cell_type = mesh.cells[0].type
    print(f"  Nodes:    {len(mesh.points):,}")
    print(f"  Elements: {len(all_cells):,}  ({cell_type})")

    # Flatten element variables
    elem_vars = {}
    for key, arrays in mesh.cell_data.items():
        flat = np.concatenate(arrays)
        elem_vars[key] = flat
        print(f"  Var '{key}': range [{flat.min():.2f}, {flat.max():.2f}]")

    return {
        'points':       mesh.points,
        'connectivity': all_cells,
        'cell_type':    cell_type,
        'elem_vars':    elem_vars,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Organise into blocks
# ──────────────────────────────────────────────────────────────────────────────

def organise_blocks(mesh_data: dict, organize_by: str) -> dict:
    """
    Split elements into blocks by the chosen field (needle_id or grain_id).
    Returns a dict: block_id → {connectivity, elem_vars, grain_id}
    """
    elem_vars    = mesh_data['elem_vars']
    connectivity = mesh_data['connectivity']

    if organize_by not in elem_vars:
        raise KeyError(f"Field '{organize_by}' not found. "
                       f"Available: {list(elem_vars.keys())}")

    id_data    = elem_vars[organize_by].astype(int)
    unique_ids = np.unique(id_data)
    print(f"\nOrganising {len(unique_ids)} blocks by '{organize_by}'...")

    blocks = {}
    for bid in unique_ids:
        mask = (id_data == bid)
        block_vars = {k: v[mask] for k, v in elem_vars.items()}

        # Determine which grain this block belongs to
        # (for needle-level: grain_id is constant per block)
        if 'grain_id' in elem_vars:
            grain_ids_in_block = np.unique(elem_vars['grain_id'][mask].astype(int))
            grain_id = int(grain_ids_in_block[0])
            if len(grain_ids_in_block) > 1:
                print(f"  WARNING: block {bid} spans grains {grain_ids_in_block}")
        else:
            grain_id = int(bid) if organize_by == 'grain_id' else -1

        blocks[int(bid)] = {
            'connectivity': connectivity[mask],
            'elem_vars':    block_vars,
            'grain_id':     grain_id,
            'n_elements':   int(mask.sum()),
        }

    if len(unique_ids) <= 30:
        for bid, b in blocks.items():
            print(f"  Block {bid:4d}: {b['n_elements']:6,} elems  grain={b['grain_id']}")

    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Detect interfaces
# ──────────────────────────────────────────────────────────────────────────────

def detect_interfaces(mesh_data: dict, blocks: dict) -> dict:
    """
    Find all internal interfaces between adjacent blocks.

    Strategy: build a face→element lookup over ALL elements.
    Any face shared by exactly 2 elements that belong to different blocks
    is an interface face.

    Returns interface_map:
        (block_A, block_B) → {
            'type':        'intra_grain' | 'inter_grain',
            'grain_A':     int,
            'grain_B':     int,
            'n_faces':     int,
        }
    where block_A < block_B always (canonical ordering).
    """
    connectivity = mesh_data['connectivity']
    n_elements   = len(connectivity)

    print(f"\nDetecting interfaces ({n_elements:,} elements)...")

    # Build element → block_id mapping
    elem_to_block = np.zeros(n_elements, dtype=int)
    for bid, b in blocks.items():
        # Reconstruct which global element indices belong to this block
        # We rely on the fact that blocks partition all elements
        pass  # filled below

    # Rebuild in one pass
    id_field = list(blocks.values())[0]['elem_vars']
    # Use any elem_var that was the organize_by key
    # Actually we need to reconstruct from connectivity matching
    # Simpler: re-derive from original id_data order
    # We stored connectivity subsets — recover original indices via matching
    # For large meshes this matters; use a hash on first node of each element
    #
    # Efficient approach: assign block IDs in order we know they were created.
    # Since we split by unique_ids in sorted order, rebuild elem_to_block directly
    # from the original field (which is still in mesh_data['elem_vars']).

    # Determine which element variable field was used to organise blocks.
    # blocks.keys() == unique values of the organize_by field, so find
    # whichever elem_var matches those unique values exactly.
    block_ids_set = set(blocks.keys())
    elem_to_block = None

    for field_name, field_data in mesh_data['elem_vars'].items():
        vals = set(np.unique(field_data.astype(int)).tolist())
        if vals == block_ids_set:
            elem_to_block = field_data.astype(int)
            print(f"  Using '{field_name}' as block-ID field "
                  f"({len(block_ids_set)} unique values)")
            break

    if elem_to_block is None:
        # Fallback: use grain_id if available, otherwise first field
        fallback = ('grain_id' if 'grain_id' in mesh_data['elem_vars']
                    else list(mesh_data['elem_vars'].keys())[0])
        elem_to_block = mesh_data['elem_vars'][fallback].astype(int)
        print(f"  WARNING: no elem_var matched block IDs exactly. "
              f"Falling back to '{fallback}'.")
        print(f"  Interface classification may be incorrect.") 

    # Build face → [elem_idx, ...] lookup
    # Each hex has 6 faces; each face represented as sorted tuple of 4 global node IDs
    face_to_elems: dict[tuple, list] = defaultdict(list)

    print("  Building face lookup table...")
    for elem_idx in range(n_elements):
        nodes = connectivity[elem_idx]
        for local_face in HEX8_FACES:
            global_nodes = tuple(sorted(nodes[i] for i in local_face))
            face_to_elems[global_nodes].append(elem_idx)

    print(f"  Total faces: {len(face_to_elems):,}")

    # Find interface faces (shared by 2 elements of different blocks)
    print("  Scanning for interface faces...")
    interface_faces: dict[tuple[int,int], int] = defaultdict(int)  # (blkA,blkB)→count

    for face_nodes, elems in face_to_elems.items():
        if len(elems) != 2:
            continue   # boundary face (1) or degenerate (>2)
        bA = int(elem_to_block[elems[0]])
        bB = int(elem_to_block[elems[1]])
        if bA == bB:
            continue
        key = (min(bA, bB), max(bA, bB))
        interface_faces[key] += 1

    print(f"  Found {len(interface_faces)} unique block-pair interfaces")

    # Classify each interface
    interface_map = {}
    for (bA, bB), n_faces in sorted(interface_faces.items()):
        gA = blocks[bA]['grain_id'] if bA in blocks else -1
        gB = blocks[bB]['grain_id'] if bB in blocks else -1
        itype = 'intra_grain' if gA == gB else 'inter_grain'
        interface_map[(bA, bB)] = {
            'type':    itype,
            'grain_A': gA,
            'grain_B': gB,
            'n_faces': n_faces,
        }

    n_intra = sum(1 for v in interface_map.values() if v['type'] == 'intra_grain')
    n_inter = sum(1 for v in interface_map.values() if v['type'] == 'inter_grain')
    print(f"  Intra-grain interfaces: {n_intra}")
    print(f"  Inter-grain interfaces: {n_inter}")

    return interface_map


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Compute mesh bounds
# ──────────────────────────────────────────────────────────────────────────────

def compute_bounds(points: np.ndarray, tol_fraction: float = 0.001) -> dict:
    """
    Compute mesh bounding box and boundary detection tolerances.
    tol_fraction: fraction of domain size used for ParsedGenerateSideset thresholds.
    """
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    tol_x = dx * tol_fraction
    tol_y = dy * tol_fraction
    tol_z = dz * tol_fraction

    return {
        'xmin': float(xmin), 'xmax': float(xmax),
        'ymin': float(ymin), 'ymax': float(ymax),
        'zmin': float(zmin), 'zmax': float(zmax),
        'tol_x': float(tol_x),
        'tol_y': float(tol_y),
        'tol_z': float(tol_z),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Write multi-block Exodus
# ──────────────────────────────────────────────────────────────────────────────

def write_exodus(output_file: str, mesh_data: dict, blocks: dict) -> None:
    """
    Write multi-block Exodus II with element variables preserved in every block.
    Block numbering in the output is 1-indexed and matches sorted block IDs.
    """
    print(f"\nWriting multi-block Exodus: {output_file}")

    points    = mesh_data['points']
    cell_type = mesh_data['cell_type']
    sorted_bids = sorted(blocks.keys())

    cells     = []
    cell_data = defaultdict(list)

    for bid in sorted_bids:
        b = blocks[bid]
        cells.append((cell_type, b['connectivity']))
        for key, vals in b['elem_vars'].items():
            cell_data[key].append(vals)

    new_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=dict(cell_data),
    )
    new_mesh.write(output_file, file_format='exodus')
    print(f"  Base exodus written ({len(cells)} blocks)")

    # Re-open with netCDF4 to ensure element variables are properly stored
    # (meshio sometimes drops them for multi-block exodus)
    print("  Patching element variables via netCDF4...")
    var_names = list(cell_data.keys())
    n_vars    = len(var_names)
    n_blocks  = len(sorted_bids)

    with nc.Dataset(output_file, 'r+') as exo:
        if 'num_elem_var' not in exo.dimensions:
            exo.createDimension('num_elem_var', n_vars)

            name_var = exo.createVariable('name_elem_var', 'S1',
                                          ('num_elem_var', 'len_string'))
            for i, vname in enumerate(var_names):
                name_str   = vname.ljust(33, '\x00')
                name_array = np.array([c.encode('utf-8') for c in name_str], dtype='S1')
                name_var[i, :] = name_array

        # Set explicit block names to their numeric IDs.
        # This ensures BreakMeshByBlockGenerator creates sidesets named
        # 'interface_{A}_{B}' (e.g. 'interface_1_2') rather than the
        # meshio-generated names like 'Block-1_Block-2'.
        if 'eb_names' in exo.variables:
            for blk_idx, bid in enumerate(sorted_bids):
                name_str   = str(bid).ljust(33, '\x00')
                name_array = np.array([c.encode('utf-8') for c in name_str], dtype='S1')
                exo.variables['eb_names'][blk_idx, :] = name_array
            print(f"  Block names set to numeric IDs: "
                  f"{sorted_bids[0]}..{sorted_bids[-1]}")
            print(f"  BreakMeshByBlock will create sidesets: "
                  f"interface_{sorted_bids[0]}_{sorted_bids[1]}, ...")

        for var_idx, var_name in enumerate(var_names):
            for block_idx in range(n_blocks):
                var_key  = f'vals_elem_var{var_idx+1}eb{block_idx+1}'
                dim_name = f'num_el_in_blk{block_idx+1}'

                if var_key in exo.variables:
                    continue
                if dim_name not in exo.dimensions:
                    continue

                var = exo.createVariable(var_key, 'f8', ('time_step', dim_name))
                var[0, :] = cell_data[var_name][block_idx].astype(np.float64)

    print(f"  Done — {n_blocks} blocks, {n_vars} element variables")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Write JSON outputs
# ──────────────────────────────────────────────────────────────────────────────

def write_mesh_info(output_stem: str, bounds: dict, blocks: dict,
                    organize_by: str) -> str:
    """Write mesh_info.json."""
    block_map = {
        str(bid): {'grain_id': b['grain_id'], 'n_elements': b['n_elements']}
        for bid, b in blocks.items()
    }
    info = {
        'bounds':       bounds,
        'organize_by':  organize_by,
        'n_blocks':     len(blocks),
        'block_map':    block_map,
    }
    path = output_stem + '_mesh_info.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"  mesh_info.json written: {path}")
    return path


def write_interface_map(output_stem: str, interface_map: dict) -> str:
    """Write interface_map.json."""
    # Convert tuple keys to strings for JSON
    serialisable = {
        f"block{bA}_block{bB}": v
        for (bA, bB), v in sorted(interface_map.items())
    }
    path = output_stem + '_interface_map.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2)
    print(f"  interface_map.json written: {path}  ({len(serialisable)} interfaces)")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Generate MOOSE input snippet
# ──────────────────────────────────────────────────────────────────────────────

def write_moose_snippet(
    output_stem: str,
    exodus_file: str,
    bounds: dict,
    interface_map: dict,
    czm_intra: str = 'CZM_IntraGrain',
    czm_inter: str = 'CZM_InterGrain',
) -> str:
    """
    Generate a ready-to-use MOOSE .i snippet.

    HomogenizedExponentialCZM is a MATERIAL (not an InterfaceKernel).
    The kernel machinery comes from [Physics][SolidMechanics][CohesiveZone].
    See the working example in examples/grain_interfaces.i for reference.

    Sections generated:
      [Mesh]      FileMeshGenerator + 6 ParsedGenerateSideset + BreakMeshByBlock
      [Physics]   CohesiveZone sub-block (to add inside your QuasiStatic action)
      [Materials] HomogenizedExponentialCZM for intra- and inter-grain boundaries
      [BCs]       External face stubs (uniaxial tension along X by default)

    IMPORTANT — boundary naming:
      BreakMeshByBlockGenerator creates sidesets named 'interface_{A}_{B}'
      where A and B are the block IDs in the exodus file.
      Verify the exact names with a --mesh-only MOOSE run before committing
      to a full simulation.
    """
    b = bounds
    x_lo = f"x < {b['xmin'] + b['tol_x']:.6f}"
    x_hi = f"x > {b['xmax'] - b['tol_x']:.6f}"
    y_lo = f"y < {b['ymin'] + b['tol_y']:.6f}"
    y_hi = f"y > {b['ymax'] - b['tol_y']:.6f}"
    z_lo = f"z < {b['zmin'] + b['tol_z']:.6f}"
    z_hi = f"z > {b['zmax'] - b['tol_z']:.6f}"

    has_intra = any(v['type'] == 'intra_grain' for v in interface_map.values())
    has_inter = any(v['type'] == 'inter_grain'  for v in interface_map.values())

    intra_pairs = [(bA, bB) for (bA, bB), v in sorted(interface_map.items())
                   if v['type'] == 'intra_grain']
    inter_pairs = [(bA, bB) for (bA, bB), v in sorted(interface_map.items())
                   if v['type'] == 'inter_grain']

    # Build boundary string lists for each type
    # BreakMeshByBlockGenerator naming: 'interface_{blockA}_{blockB}'
    intra_boundaries = ' '.join(f'interface_{bA}_{bB}' for bA, bB in intra_pairs)
    inter_boundaries = ' '.join(f'interface_{bA}_{bB}' for bA, bB in inter_pairs)

    lines = []
    lines.append("# ============================================================")
    lines.append("# MOOSE input snippet — generated by convert_to_moose.py")
    lines.append("#")
    lines.append("# USAGE:")
    lines.append("#   1. Copy [Mesh] into your input file")
    lines.append("#   2. Add the [Physics][SolidMechanics][CohesiveZone] block")
    lines.append("#      INSIDE your existing QuasiStatic action")
    lines.append("#   3. Add the [Materials] CZM blocks and fill in parameters")
    lines.append("#   4. Verify boundary names with a --mesh-only run first")
    lines.append("# ============================================================")
    lines.append("")

    # ── [Mesh] ────────────────────────────────────────────────────────────────
    lines.append("[Mesh]")
    lines.append("  [file]")
    lines.append("    type = FileMeshGenerator")
    lines.append(f"    file = '{Path(exodus_file).name}'")
    lines.append("    use_for_exodus_restart = true   # needed to read Euler angle element vars")
    lines.append("  []")
    lines.append("")
    lines.append("  # External face sidesets — coordinates from actual mesh bounds")
    lines.append("  # Adjust tolerance values if mesh has non-zero origin")
    for name, expr, inp in [
        ('left',   x_lo, 'file'),
        ('right',  x_hi, 'left'),
        ('front',  y_lo, 'right'),
        ('back',   y_hi, 'front'),
        ('bottom', z_lo, 'back'),
        ('top',    z_hi, 'bottom'),
    ]:
        lines.append(f"  [{name}]")
        lines.append(f"    type = ParsedGenerateSideset")
        lines.append(f"    input = {inp}")
        lines.append(f"    combinatorial_geometry = '{expr}'")
        lines.append(f"    new_sideset_name = '{name}'")
        lines.append(f"  []")

    lines.append("")
    lines.append("  # Break mesh at block boundaries.")
    lines.append("  # Creates sidesets named 'interface_{blockA}_{blockB}'.")
    lines.append("  # Verify exact names with: mpirun -n 1 ./moose-opt -i this.i --mesh-only")
    lines.append("  [break]")
    lines.append("    type = BreakMeshByBlockGenerator")
    lines.append("    input = top")
    lines.append("    split_interface = true")
    lines.append("  []")
    lines.append("[]")
    lines.append("")

    # ── [Physics] — CohesiveZone (add inside existing QuasiStatic action) ────
    lines.append("# Add the CohesiveZone sub-block inside your existing")
    lines.append("# [Physics][SolidMechanics][QuasiStatic] action:")
    lines.append("#")
    lines.append("# [Physics]")
    lines.append("#   [SolidMechanics]")
    lines.append("#     [QuasiStatic]")
    lines.append("#       [bulk]")
    lines.append("#         strain = FINITE")
    lines.append("#         incremental = true")
    lines.append("#         add_variables = true")
    lines.append("#         generate_output = 'stress_xx stress_yy stress_zz vonmises_stress'")
    lines.append("#       []")
    lines.append("#     []")
    lines.append("#")

    if has_intra:
        lines.append("#     [CohesiveZone]")
        lines.append("#       [intra_grain_czm]")
        lines.append(f"#         boundary = '{intra_boundaries}'")
        lines.append("#         strain = FINITE")
        lines.append("#         generate_output = 'traction_x traction_y traction_z")
        lines.append("#                            normal_traction tangent_traction")
        lines.append("#                            jump_x jump_y jump_z")
        lines.append("#                            normal_jump tangent_jump'")
        lines.append("#       []")
        if has_inter:
            lines.append("#       [inter_grain_czm]")
            lines.append(f"#         boundary = '{inter_boundaries}'")
            lines.append("#         strain = FINITE")
            lines.append("#         generate_output = 'traction_x traction_y traction_z")
            lines.append("#                            normal_traction tangent_traction")
            lines.append("#                            normal_jump tangent_jump'")
            lines.append("#       []")
        lines.append("#     []")
    lines.append("#   []")
    lines.append("# []")
    lines.append("")

    # ── [Materials] — CZM ─────────────────────────────────────────────────────
    lines.append("# HomogenizedExponentialCZM is a MATERIAL, not an InterfaceKernel.")
    lines.append("# Apply it to the same boundary list as the CohesiveZone block above.")
    lines.append("# Fill in parameters from MD / Kvashin et al. 2026.")
    lines.append("")
    lines.append("[Materials]")

    if has_intra:
        lines.append(f"  # Intra-grain: needle-needle within one sclerodermite")
        lines.append(f"  # (organic matrix / protein bonding)")
        lines.append(f"  [czm_intra]")
        lines.append(f"    type = HomogenizedExponentialCZM")
        lines.append(f"    boundary = '{intra_boundaries}'")
        lines.append(f"    # --- Fill in from MD ---")
        lines.append(f"    normal_strength     = ...   # MPa")
        lines.append(f"    shear_strength_s    = ...   # MPa")
        lines.append(f"    shear_strength_t    = ...   # MPa")
        lines.append(f"    delta_0             = ...   # um — characteristic separation")
        lines.append(f"    md_contact_area     = ...   # um^2 — from Kvashin et al.")
        lines.append(f"    max_contacts        = ...   # integer")
        lines.append(f"    mu                  = 0.95  # loading exponent")
        lines.append(f"    eta                 = 0.1   # softening exponent")
        lines.append(f"    damage_viscosity    = 2.5")
        lines.append(f"    strength_std_dev    = 0.0")
        lines.append(f"    initial_damage_max  = 0.0")
        lines.append(f"    delta0_std_dev      = 0.0")
        lines.append(f"  []")

    if has_inter:
        lines.append(f"")
        lines.append(f"  # Inter-grain: across sclerodermite boundaries")
        lines.append(f"  # (water layer / protein — weaker than intra-grain)")
        lines.append(f"  [czm_inter]")
        lines.append(f"    type = HomogenizedExponentialCZM")
        lines.append(f"    boundary = '{inter_boundaries}'")
        lines.append(f"    # --- Fill in from MD (different values from czm_intra) ---")
        lines.append(f"    normal_strength     = ...   # MPa")
        lines.append(f"    shear_strength_s    = ...   # MPa")
        lines.append(f"    shear_strength_t    = ...   # MPa")
        lines.append(f"    delta_0             = ...   # um")
        lines.append(f"    md_contact_area     = ...")
        lines.append(f"    max_contacts        = ...")
        lines.append(f"    mu                  = 0.95")
        lines.append(f"    eta                 = 0.1")
        lines.append(f"    damage_viscosity    = 2.5")
        lines.append(f"    strength_std_dev    = 0.0")
        lines.append(f"    initial_damage_max  = 0.0")
        lines.append(f"    delta0_std_dev      = 0.0")
        lines.append(f"  []")

    lines.append("  # Add elasticity, plasticity, and other material blocks here")
    lines.append("[]")
    lines.append("")

    # ── [BCs] ─────────────────────────────────────────────────────────────────
    lines.append("# Default: uniaxial tension along X (matches working example).")
    lines.append("# Adjust for your load case.")
    lines.append("[BCs]")
    lines.append("  [fix_left_x]")
    lines.append("    type = DirichletBC")
    lines.append("    boundary = left")
    lines.append("    variable = disp_x")
    lines.append("    value = 0")
    lines.append("  []")
    lines.append(f"  [load_right_x]")
    lines.append(f"    type = FunctionDirichletBC")
    lines.append(f"    boundary = right")
    lines.append(f"    variable = disp_x")
    dx = b['xmax'] - b['xmin']
    lines.append(f"    function = '0.01*t'  # strain rate 1%/time_unit; domain X = {dx:.4f}")
    lines.append(f"  []")
    lines.append("  [fix_bottom_y]")
    lines.append("    type = DirichletBC")
    lines.append("    boundary = bottom")
    lines.append("    variable = disp_y")
    lines.append("    value = 0")
    lines.append("  []")
    lines.append("  [fix_back_z]")
    lines.append("    type = DirichletBC")
    lines.append("    boundary = back")
    lines.append("    variable = disp_z")
    lines.append("    value = 0")
    lines.append("  []")
    lines.append("[]")
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines.append("# ── Mesh summary ─────────────────────────────────────────────")
    lines.append(f"# Bounds:  X [{b['xmin']:.4f}, {b['xmax']:.4f}]")
    lines.append(f"#          Y [{b['ymin']:.4f}, {b['ymax']:.4f}]")
    lines.append(f"#          Z [{b['zmin']:.4f}, {b['zmax']:.4f}]")
    lines.append(f"# Interfaces: {len(interface_map)} total")
    if has_intra:
        lines.append(f"#   {len(intra_pairs)} intra-grain (needle-needle, same sclerodermite)")
    if has_inter:
        lines.append(f"#   {len(inter_pairs)} inter-grain (across sclerodermite boundaries)")
    lines.append("#")
    lines.append("# Interface boundary names (verify with --mesh-only run):")
    if has_intra and intra_pairs:
        sample = intra_pairs[:3]
        names = ' '.join(f'interface_{a}_{b_}' for a, b_ in sample)
        suffix = f' ... ({len(intra_pairs)} total)' if len(intra_pairs) > 3 else ''
        lines.append(f"#   intra: {names}{suffix}")
    if has_inter and inter_pairs:
        sample = inter_pairs[:3]
        names = ' '.join(f'interface_{a}_{b_}' for a, b_ in sample)
        suffix = f' ... ({len(inter_pairs)} total)' if len(inter_pairs) > 3 else ''
        lines.append(f"#   inter: {names}{suffix}")

    path = output_stem + '_moose.i'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  MOOSE snippet written: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Optional — Mackenzie plot verification from converted exodus
# ──────────────────────────────────────────────────────────────────────────────

def verify_misorientation(
    exodus_file: str,
    blocks: dict,
    n_pairs: int = 30000,
    seed: int = 0,
    output_file: Optional[str] = None,
) -> None:
    """
    Read Euler angles from the converted multi-block exodus and produce a
    three-panel Mackenzie-style misorientation plot.

    Why adjacent-element pairs (not block pairs)
    --------------------------------------------
    Each block (needle) has a unique orientation. If we sample *random block
    pairs within a grain*, the c-axes of the two needles typically point in
    different directions (they fan radially from the grain centre), so the full
    3D misorientation is dominated by the c-axis angular difference — the phi2
    twin signal is buried.

    The EBSD Mackenzie plot samples *adjacent pixels*: neighbouring pixels have
    nearly identical c-axes, so their misorientation is almost entirely a phi2
    difference — and the twin peaks appear cleanly.

    This function mimics that by reading element connectivity directly from the
    exodus and finding face-sharing element pairs.  For the three XY / XZ / YZ
    mid-plane slices it further restricts to pairs whose element centres both
    lie close to the slice plane, replicating a planar EBSD scan.

    Panels
    ------
    Left   — All adjacent cross-needle same-grain element pairs (full volume)
    Centre — XY mid-plane slice only  (mimics in-plane EBSD section)
    Right  — XZ mid-plane slice  (orthogonal cut)

    Bins: 1 degree (matching experimental EBSD resolution).
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("  matplotlib and scipy required for verification plot. Skipping.")
        return

    print("\nVerification: reading mesh + Euler angles from exodus...")

    # ── Read per-element Euler angles, grain_id, needle_id, and connectivity ─
    elem_phi1  = []  # flat arrays, global element order
    elem_Phi   = []
    elem_phi2  = []
    elem_grain = []
    elem_needle= []
    all_connectivity = []   # list of 8-node arrays
    all_points = None

    with nc.Dataset(exodus_file, 'r') as exo:
        if 'name_elem_var' not in exo.variables:
            print("  No element variables found — skipping verification.")
            return

        # Decode variable names
        raw_names = exo.variables['name_elem_var'][:]
        var_names = []
        for row in raw_names:
            try:
                name = row.tobytes().decode('utf-8').rstrip('\x00').strip()
            except Exception:
                name = ''
            var_names.append(name)

        def find_idx(keywords):
            for kw in keywords:
                idx = next((i for i,n in enumerate(var_names)
                            if kw in n.lower()), None)
                if idx is not None:
                    return idx
            return None

        phi1_idx  = find_idx(['phi1'])
        Phi_idx   = find_idx(['euler_phi', 'euler_p'])
        phi2_idx  = find_idx(['phi2'])
        grain_idx = find_idx(['grain'])
        needle_idx= find_idx(['needle'])

        # Fallback: positional (euler vars are typically indices 2,3,4)
        euler_idxs = [i for i,n in enumerate(var_names)
                      if 'euler' in n.lower() or 'phi' in n.lower()]
        if phi1_idx is None and len(euler_idxs) > 0:
            phi1_idx = euler_idxs[0]
        if Phi_idx is None and len(euler_idxs) > 1:
            Phi_idx = euler_idxs[1]
        if phi2_idx is None and len(euler_idxs) > 2:
            phi2_idx = euler_idxs[2]

        if None in (phi1_idx, Phi_idx, phi2_idx):
            print(f"  Cannot find Euler angle variables in: {var_names}")
            print("  Skipping verification.")
            return

        # Node coordinates
        coords = exo.variables['coord'][:]   # shape (3, n_nodes) or (n_nodes, 3)
        if coords.shape[0] == 3:
            all_points = coords.T            # → (n_nodes, 3)
        else:
            all_points = coords

        # Read all blocks
        n_blocks = sum(1 for k in exo.dimensions if k.startswith('num_el_in_blk'))
        global_elem = 0

        for blk in range(n_blocks):
            blk1 = blk + 1
            conn_key = f'connect{blk1}'
            if conn_key not in exo.variables:
                continue
            conn = exo.variables[conn_key][:] - 1  # 0-indexed; shape (n_el, 8)
            n_el = conn.shape[0]

            phi1_key = f'vals_elem_var{phi1_idx+1}eb{blk1}'
            Phi_key  = f'vals_elem_var{Phi_idx+1}eb{blk1}'
            phi2_key = f'vals_elem_var{phi2_idx+1}eb{blk1}'

            if phi1_key not in exo.variables:
                continue

            phi1_v  = exo.variables[phi1_key][0, :]
            Phi_v   = exo.variables[Phi_key][0, :]
            phi2_v  = exo.variables[phi2_key][0, :]

            elem_phi1.extend(phi1_v.tolist())
            elem_Phi.extend(Phi_v.tolist())
            elem_phi2.extend(phi2_v.tolist())

            if grain_idx is not None:
                gk = f'vals_elem_var{grain_idx+1}eb{blk1}'
                elem_grain.extend(exo.variables[gk][0, :].tolist()
                                  if gk in exo.variables
                                  else [blk1]*n_el)
            else:
                elem_grain.extend([blk1]*n_el)

            if needle_idx is not None:
                nk = f'vals_elem_var{needle_idx+1}eb{blk1}'
                elem_needle.extend(exo.variables[nk][0, :].tolist()
                                   if nk in exo.variables
                                   else [blk1]*n_el)
            else:
                elem_needle.extend([blk1]*n_el)

            all_connectivity.extend(conn.tolist())
            global_elem += n_el

    n_elem = len(elem_phi1)
    print(f"  Read {n_elem:,} elements from {n_blocks} blocks")

    elem_phi1  = np.array(elem_phi1,  dtype=np.float32)
    elem_Phi   = np.array(elem_Phi,   dtype=np.float32)
    elem_phi2  = np.array(elem_phi2,  dtype=np.float32)
    elem_grain = np.array(elem_grain, dtype=np.int32)
    elem_needle= np.array(elem_needle,dtype=np.int32)
    connectivity = np.array(all_connectivity, dtype=np.int32)  # (n_elem, 8)

    # ── Build rotation objects per unique needle (not per element — far fewer) ─
    unique_needles = np.unique(elem_needle)
    print(f"  Building rotations for {len(unique_needles)} unique needles...")

    from scipy.spatial.transform import Rotation
    needle_rot = {}
    for nid in unique_needles:
        mask = (elem_needle == nid)
        p1 = float(elem_phi1[mask].mean())
        P  = float(elem_Phi[mask].mean())
        p2 = float(elem_phi2[mask].mean())
        needle_rot[nid] = Rotation.from_euler('ZXZ', [p1, P, p2], degrees=True)

    sym_ops = Rotation.from_euler('ZXZ', [
        [0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]
    ], degrees=True)

    def misori(nid_a, nid_b):
        ra = needle_rot[nid_a]
        rb = needle_rot[nid_b]
        delta = ra.inv() * rb
        return min(np.degrees((s * delta).magnitude()) for s in sym_ops)

    # ── Element centres (mean of 8 corner nodes) ─────────────────────────────
    print("  Computing element centres...")
    elem_centres = all_points[connectivity].mean(axis=1)  # (n_elem, 3)
    cx_mid = (elem_centres[:, 0].min() + elem_centres[:, 0].max()) / 2
    cy_mid = (elem_centres[:, 1].min() + elem_centres[:, 1].max()) / 2
    cz_mid = (elem_centres[:, 2].min() + elem_centres[:, 2].max()) / 2

    # ── Face adjacency — find cross-needle same-grain pairs ───────────────────
    print("  Building face adjacency table...")
    face_to_elem: dict[tuple, int] = {}
    cross_pairs: list[tuple[int,int]] = []   # (elem_i, elem_j) cross-needle same-grain

    for ei in range(n_elem):
        nodes = connectivity[ei]
        for lf in HEX8_FACES:
            key = tuple(sorted(nodes[k] for k in lf))
            if key in face_to_elem:
                ej = face_to_elem[key]
                # Same grain, different needle
                if (elem_grain[ei] == elem_grain[ej] and
                        elem_needle[ei] != elem_needle[ej]):
                    cross_pairs.append((ei, ej))
            else:
                face_to_elem[key] = ei

    print(f"  Cross-needle same-grain adjacent pairs: {len(cross_pairs):,}")
    if len(cross_pairs) == 0:
        print("  No adjacent cross-needle pairs found — mesh may be too small.")
        print("  Skipping verification.")
        return

    # Deduplicate by needle pair
    pair_cache: dict[tuple, float] = {}
    all_angles: list[float] = []

    # Also collect slice-restricted pairs
    xy_angles: list[float] = []   # pairs near z = cz_mid
    xz_angles: list[float] = []   # pairs near y = cy_mid

    slice_tol = (elem_centres[:, 0].max() - elem_centres[:, 0].min()) * 0.03

    print("  Computing misorientation angles...")
    for ei, ej in cross_pairs:
        na = int(elem_needle[ei])
        nb = int(elem_needle[ej])
        key = (min(na, nb), max(na, nb))
        if key not in pair_cache:
            pair_cache[key] = misori(na, nb)
        angle = pair_cache[key]
        all_angles.append(angle)

        # XY slice: both element centres near cz_mid
        if (abs(elem_centres[ei, 2] - cz_mid) < slice_tol and
                abs(elem_centres[ej, 2] - cz_mid) < slice_tol):
            xy_angles.append(angle)

        # XZ slice: both near cy_mid
        if (abs(elem_centres[ei, 1] - cy_mid) < slice_tol and
                abs(elem_centres[ej, 1] - cy_mid) < slice_tol):
            xz_angles.append(angle)

    all_angles = np.array(all_angles)
    xy_angles  = np.array(xy_angles)
    xz_angles  = np.array(xz_angles)

    # Random inter-grain pairs for reference
    rng = np.random.default_rng(seed)
    all_nids = list(unique_needles)
    grain_of_needle = {}
    for nid in unique_needles:
        grain_of_needle[nid] = int(elem_grain[elem_needle == nid][0])

    rand_angles = []
    for _ in range(min(n_pairs, len(all_angles))):
        na, nb = rng.choice(all_nids, 2, replace=False)
        if grain_of_needle[na] != grain_of_needle[nb]:  # inter-grain
            rand_angles.append(misori(na, nb))
    rand_angles = np.array(rand_angles)

    print(f"  Full volume pairs:  {len(all_angles):,}")
    print(f"  XY-slice pairs:     {len(xy_angles):,}")
    print(f"  XZ-slice pairs:     {len(xz_angles):,}")
    print(f"  Random inter-grain: {len(rand_angles):,}")

    # ── Plot — 3 panels, 1-degree bins ────────────────────────────────────────
    twin_peaks = [
        (11.0,  'Low-angle\n11°'),
        (52.4,  'Mirror\n52.4°'),
        (57.2,  '(310)\n57.2°'),
        (63.8,  '(110)\n63.8°'),
    ]
    bins = np.arange(0, 121, 1)   # 1-degree bins, matching EBSD resolution

    datasets = [
        (all_angles, rand_angles,
         f'All adjacent (n={len(all_angles):,})',
         f'Random inter-grain (n={len(rand_angles):,})',
         'Full volume'),
        (xy_angles, None,
         f'XY mid-plane (n={len(xy_angles):,})', None,
         f'XY slice  z={cz_mid:.2f}  (mimics horizontal EBSD section)'),
        (xz_angles, None,
         f'XZ mid-plane (n={len(xz_angles):,})', None,
         f'XZ slice  y={cy_mid:.2f}  (mimics vertical EBSD section)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (data, ref, lbl, ref_lbl, title) in zip(axes, datasets):
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No pairs in this slice',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        if ref is not None and len(ref) > 0:
            ax.hist(ref, bins=bins, density=True, alpha=0.40,
                    color='orange', label=ref_lbl)
        ax.hist(data, bins=bins, density=True, alpha=0.70,
                color='steelblue', label=lbl)

        ymax = ax.get_ylim()[1]
        for ang, lbl_peak in twin_peaks:
            ax.axvline(ang, color='firebrick', lw=1.0, ls='--')
            ax.text(ang+0.3, ymax*0.94, lbl_peak, rotation=90,
                    va='top', fontsize=7, color='firebrick')

        ax.set_xlabel('Misorientation angle (deg)')
        ax.set_ylabel('Density')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 120)

    fig.suptitle(
        f'Orientation verification — {Path(exodus_file).name}\n'
        f'Adjacent-element misorientation, 1-degree bins  '
        f'(compare with EBSD Mackenzie plot)',
        fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    out = output_file or str(Path(exodus_file).with_suffix('')) + '_mackenzie.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Mackenzie plot saved: {out}")
    plt.close()




def convert(
    input_file:   str,
    output_file:  str,
    organize_by:  str  = 'needle_id',
    force:        bool = False,
    moose_snippet: bool = True,
    czm_intra:    str  = 'CZM_IntraGrain',
    czm_inter:    str  = 'CZM_InterGrain',
    verify:       bool = False,
) -> dict:
    """
    Full conversion pipeline.  Returns dict with paths of all output files.
    """
    # ── Guard: output file ───────────────────────────────────────────────────
    if Path(output_file).exists() and not force:
        ans = input(f"\n'{output_file}' exists. Overwrite? (y/n): ")
        if ans.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    output_stem = str(Path(output_file).with_suffix(''))

    print("\n" + "=" * 70)
    print("CONVERT TO MOOSE — ARAGONITE PIPELINE")
    print("=" * 70)
    print(f"  Input:       {input_file}")
    print(f"  Output:      {output_file}")
    print(f"  Organise by: {organize_by}")

    # Step 1: Read
    mesh_data = read_exodus(input_file)

    # Step 2: Organise
    blocks = organise_blocks(mesh_data, organize_by)

    # Step 3: Interfaces
    interface_map = detect_interfaces(mesh_data, blocks)

    # Step 4: Bounds
    bounds = compute_bounds(mesh_data['points'])
    print(f"\nMesh bounds:")
    print(f"  X: [{bounds['xmin']:.4f}, {bounds['xmax']:.4f}]")
    print(f"  Y: [{bounds['ymin']:.4f}, {bounds['ymax']:.4f}]")
    print(f"  Z: [{bounds['zmin']:.4f}, {bounds['zmax']:.4f}]")

    # Step 5: Write exodus
    write_exodus(output_file, mesh_data, blocks)

    # Step 6: Write JSON
    print("\nWriting JSON outputs...")
    info_path  = write_mesh_info(output_stem, bounds, blocks, organize_by)
    imap_path  = write_interface_map(output_stem, interface_map)

    # Step 7: MOOSE snippet
    snippet_path: Optional[str] = None
    if moose_snippet:
        snippet_path = write_moose_snippet(
            output_stem, output_file, bounds, interface_map,
            czm_intra, czm_inter,
        )

    if verify:
        verify_misorientation(output_file, blocks, seed=0)

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"  Exodus:          {output_file}")
    print(f"  Mesh info:       {info_path}")
    print(f"  Interface map:   {imap_path}")
    if snippet_path:
        print(f"  MOOSE snippet:   {snippet_path}")
    print(f"\n  Interfaces: {len(interface_map)} total  "
          f"({sum(1 for v in interface_map.values() if v['type']=='intra_grain')} intra, "
          f"{sum(1 for v in interface_map.values() if v['type']=='inter_grain')} inter)")
    print()

    return {
        'exodus':        output_file,
        'mesh_info':     info_path,
        'interface_map': imap_path,
        'moose_snippet': snippet_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('input',  help='Input Exodus file (single-block from generator)')
    p.add_argument('output', help='Output Exodus file (multi-block for MOOSE)')
    p.add_argument('--organize-by', default='needle_id',
                   choices=['needle_id', 'grain_id'],
                   help='Field to use for block assignment (default: needle_id)')
    p.add_argument('--force', action='store_true',
                   help='Overwrite output files without prompting')
    p.add_argument('--no-moose-snippet', action='store_true',
                   help='Skip generation of the MOOSE .i snippet')
    p.add_argument('--czm-intra', default='CZM_IntraGrain',
                   help='Material base_name for intra-grain CZM (default: CZM_IntraGrain)')
    p.add_argument('--czm-inter', default='CZM_InterGrain',
                   help='Material base_name for inter-grain CZM (default: CZM_InterGrain)')
    p.add_argument('--verify', action='store_true',
                   help='Generate Mackenzie misorientation plot from converted exodus '
                        '(requires matplotlib + scipy)')
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if not Path(args.input).exists():
        print(f"ERROR: '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    convert(
        input_file    = args.input,
        output_file   = args.output,
        organize_by   = args.organize_by,
        force         = args.force,
        moose_snippet = not args.no_moose_snippet,
        czm_intra     = args.czm_intra,
        czm_inter     = args.czm_inter,
        verify        = args.verify,
    )


if __name__ == '__main__':
    main()
    
