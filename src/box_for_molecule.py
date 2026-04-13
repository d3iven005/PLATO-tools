import numpy as np
BOHR = 0.529177
def build_molecule_box(atom_xyz, padding_angstrom=3.0):
    coords_ang = np.array(
        [[float(a[1]), float(a[2]), float(a[3])] for a in atom_xyz],
        dtype=float
    )

    xyz_min = coords_ang.min(axis=0) - padding_angstrom
    xyz_max = coords_ang.max(axis=0) + padding_angstrom
    lengths_ang = xyz_max - xyz_min

    origin_ang = xyz_min

    cell_a = np.array([lengths_ang[0] / BOHR, 0.0, 0.0], dtype=float)
    cell_b = np.array([0.0, lengths_ang[1] / BOHR, 0.0], dtype=float)
    cell_c = np.array([0.0, 0.0, lengths_ang[2] / BOHR], dtype=float)

    return origin_ang, cell_a, cell_b, cell_c
