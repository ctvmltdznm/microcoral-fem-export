import numpy as np, sys, matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from assign_orientations import (OrientationParams, assign_orientations,
    _build_sym_ops, _misori_mmm)
from scipy.spatial.transform import Rotation

# ── Microstructure parameters ─────────────────────────────────────────────────
RESOLUTION, N_GRAINS, NEEDLES_GRAIN, SEED = 100, 100, 120, 42

def make_filled_microstructure(n_grains, needles_per_grain, resolution, seed):
    """
    Build test microstructure WITH distance-transform gap fill,
    so the entire volume is assigned to a grain/needle (like the real generator).
    """
    rng = np.random.default_rng(seed)
    volume     = np.zeros((resolution,)*3, dtype=int)
    needle_vol = np.zeros((resolution,)*3, dtype=int)
    center_props = []
    needle_id = 1
    dom = float(resolution)

    for g in range(n_grains):
        grain_id = g + 1
        cx, cy, cz = rng.uniform(8, dom-8, 3)

        phi   = rng.uniform(0, 2*np.pi, needles_per_grain)
        cos_t = rng.uniform(-1, 1, needles_per_grain)
        theta = np.arccos(cos_t)
        dirs  = np.column_stack([np.sin(theta)*np.cos(phi),
                                  np.sin(theta)*np.sin(phi),
                                  np.cos(theta)])
        needles = []
        for d in dirs:
            length = rng.uniform(dom*0.12, dom*0.28)
            width  = length / rng.uniform(8, 15)
            w_vox  = max(1, int(width))

            t_vals = np.linspace(0, 1, max(5, int(length/width*4)))
            for t in t_vals:
                pos = np.array([cx, cy, cz]) + d * t * length
                vi, vj, vk = np.round(pos).astype(int)
                for di in range(-w_vox, w_vox+1):
                    for dj in range(-w_vox, w_vox+1):
                        for dk in range(-w_vox, w_vox+1):
                            if di*di+dj*dj+dk*dk <= w_vox*w_vox:
                                ii,jj,kk = vi+di, vj+dj, vk+dk
                                if 0<=ii<resolution and 0<=jj<resolution and 0<=kk<resolution:
                                    volume[ii,jj,kk]     = grain_id
                                    needle_vol[ii,jj,kk] = needle_id

            needles.append({'id': needle_id, 'grain_id': grain_id, 'direction': d,
                            'length': length, 'width': width, 'aspect_ratio': length/width})
            needle_id += 1
        center_props.append({'id': grain_id, 'position': np.array([cx,cy,cz]), 'needles': needles})

    # Fill empty voxels via distance transform (same as real generator)
    print(f"  Filling empty voxels ({(volume==0).sum():,} / {volume.size:,})...")
    empty = (volume == 0)
    if empty.any():
        _, idx = distance_transform_edt(empty, return_indices=True)
        for arr, src in [(volume, volume.copy()), (needle_vol, needle_vol.copy())]:
            filled = arr.copy()
            filled[empty] = src[idx[0][empty], idx[1][empty], idx[2][empty]]
            arr[:] = filled

    fill_pct = 100 * (1 - empty.sum()/volume.size)
    print(f"  Volume fill before DT: {fill_pct:.0f}%  →  after: 100%")
    return center_props, volume, needle_vol


print(f"Generating {N_GRAINS}-grain {RESOLUTION}^3 filled microstructure...")
center_props, volume, needle_vol = make_filled_microstructure(
    N_GRAINS, NEEDLES_GRAIN, RESOLUTION, SEED)
print(f"  Unique grains: {len(np.unique(volume))-1}  "
      f"Unique needles: {len(np.unique(needle_vol))-1}")

print("Assigning orientations...")
params = OrientationParams()
assign_orientations(center_props, params, seed=SEED)

# ── Build rotation lookup ─────────────────────────────────────────────────────
needle_rot = {}
for center in center_props:
    for needle in center['needles']:
        needle_rot[needle['id']] = Rotation.from_euler(
            'ZXZ', [needle['phi1'], needle['Phi'], needle['phi2']], degrees=True)

sym_ops = _build_sym_ops()

# ── Exhaustive adjacent-voxel scan ────────────────────────────────────────────
def scan_adjacent(volume, needle_vol, needle_rot, sym_ops):
    ni, nj, nk = volume.shape
    pair_cache = {}
    angles = []

    for ax_idx, (si,sj,sk) in enumerate([(1,0,0),(0,1,0),(0,0,1)]):
        if si:
            v0=volume[:ni-1,:,:]; v1=volume[1:,:,:]
            n0=needle_vol[:ni-1,:,:]; n1=needle_vol[1:,:,:]
        elif sj:
            v0=volume[:,:nj-1,:]; v1=volume[:,1:,:]
            n0=needle_vol[:,:nj-1,:]; n1=needle_vol[:,1:,:]
        else:
            v0=volume[:,:,:nk-1]; v1=volume[:,:,1:]
            n0=needle_vol[:,:,:nk-1]; n1=needle_vol[:,:,1:]

        # Same grain, different needle, both >0
        mask = (v0==v1) & (v0>0) & (n0!=n1) & (n0>0) & (n1>0)
        pa = n0[mask]; pb = n1[mask]
        print(f"  Axis {ax_idx}: {mask.sum():,} cross-needle same-grain pairs")

        for a,b in zip(pa, pb):
            key = (min(int(a),int(b)), max(int(a),int(b)))
            if key not in pair_cache:
                pair_cache[key] = (_misori_mmm(needle_rot[int(a)], needle_rot[int(b)], sym_ops)
                                   if int(a) in needle_rot and int(b) in needle_rot else None)
            if pair_cache[key] is not None:
                angles.append(pair_cache[key])
    return np.array(angles)

print("Scanning all adjacent voxels (exhaustive)...")
angles = scan_adjacent(volume, needle_vol, needle_rot, sym_ops)
print(f"Total misorientation values: {len(angles):,}")

# ── Random reference ──────────────────────────────────────────────────────────
all_ids = list(needle_rot.keys())
rng = np.random.default_rng(SEED)
rand_angles = []
for _ in range(min(len(angles), 80000)):
    a,b = rng.choice(all_ids, 2, replace=False)
    rand_angles.append(_misori_mmm(needle_rot[int(a)], needle_rot[int(b)], sym_ops))
rand_angles = np.array(rand_angles)

# ── Plot ──────────────────────────────────────────────────────────────────────
twin_peaks = [
    (params.half_angle_lowangle*2, 'Low-angle\n11°',  'C0'),
    (params.half_angle_mirror*2,   'Mirror\n52.4°',   'C1'),
    (params.half_angle_twin310*2,  '(310)\n57.2°',    'C2'),
    (params.half_angle_twin110*2,  '(110)\n63.8°',    'C3'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A — full 0-120° range
ax = axes[0]
bins = np.linspace(0, 120, 121)
ax.hist(rand_angles, bins=bins, density=True, alpha=0.45, color='orange',
        label=f'Random pairs  (n={len(rand_angles):,})')
ax.hist(angles,      bins=bins, density=True, alpha=0.70, color='steelblue',
        label=f'Adjacent voxels, same grain  (n={len(angles):,})')
ymax = ax.get_ylim()[1]
for ang, lbl, col in twin_peaks:
    ax.axvline(ang, color='firebrick', lw=1.2, ls='--')
    ax.text(ang+0.3, ymax*0.95, lbl, rotation=90, va='top', fontsize=8, color='firebrick')
ax.set_xlabel('Misorientation angle (°)'); ax.set_ylabel('Density')
ax.set_title(f'Exhaustive adjacent-voxel misorientation\n'
             f'{RESOLUTION}³ filled volume — {N_GRAINS} grains × {NEEDLES_GRAIN} needles')
ax.legend(fontsize=9); ax.set_xlim(0,120)

# Panel B — zoomed 0-90°, finer bins
ax2 = axes[1]
bins2 = np.linspace(0, 90, 181)  # 0.5° bins
ax2.hist(angles, bins=bins2, density=True, alpha=0.75, color='steelblue',
         label=f'n = {len(angles):,}')
ymax2 = ax2.get_ylim()[1]
for ang, lbl, col in twin_peaks:
    if ang <= 90:
        ax2.axvline(ang, color='firebrick', lw=1.5, ls='--')
        ax2.text(ang+0.3, ymax2*0.95, lbl, rotation=90, va='top',
                 fontsize=9, color='firebrick')
ax2.set_xlabel('Misorientation angle (°)'); ax2.set_ylabel('Density')
ax2.set_title('Zoomed: 0–90°, 0.5° bins\n(compare with EBSD Mackenzie plot)')
ax2.legend(fontsize=9); ax2.set_xlim(0,90)

fig.suptitle('Synthetic Aragonite — Adjacent Voxel Misorientation\n'
             '(mimics EBSD pixel-pair analysis, filled volume)',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('adjacent_misori.png', dpi=150, bbox_inches='tight')
print("Saved: adjacent_misori.png")
plt.close()
