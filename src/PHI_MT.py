from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.PHI_cal import PHInk_c


BOHR = 0.529177

ELEMENT_LIST = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca'
]


def _write_cube(filename, data, atom_xyz, cell_a, cell_b, cell_c, origin_ang=None):
    """
    Write one scalar field to cube file.
    """
    data_real = np.asarray(np.real(data), dtype=float)
    Nx, Ny, Nz = data_real.shape
    Natom = len(atom_xyz)

    if origin_ang is None:
        origin_bohr = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        origin_bohr = np.asarray(origin_ang, dtype=float) / BOHR

    with open(filename, 'w') as f:
        f.write('cube file for molecular orbital\n')
        f.write('\n')
        f.write(f'{Natom} {origin_bohr[0]} {origin_bohr[1]} {origin_bohr[2]}\n')
        f.write(f'{Nx} {cell_a[0]/Nx} {cell_a[1]/Nx} {cell_a[2]/Nx}\n')
        f.write(f'{Ny} {cell_b[0]/Ny} {cell_b[1]/Ny} {cell_b[2]/Ny}\n')
        f.write(f'{Nz} {cell_c[0]/Nz} {cell_c[1]/Nz} {cell_c[2]/Nz}\n')

        for atom in atom_xyz:
            elem = atom[0]
            atomic_number = ELEMENT_LIST.index(elem) + 1
            x = float(atom[1]) / BOHR
            y = float(atom[2]) / BOHR
            z = float(atom[3]) / BOHR
            f.write(f'{atomic_number} 0.0 {x} {y} {z}\n')

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    f.write(f'{data_real[i, j, k]} ')
                    if k % 6 == 5:
                        f.write('\n')
                f.write('\n')


def PHImt(ifcrystal, cell_a, cell_b, cell_c, obtdictionary,
          atom_xyz, kpoint_coe_list, kpoint_list, phi_coe_list,
          xyzgrid, energylevel_list, ncpu, origin_ang=None):
    """
    Molecular orbital calculation on the real-space grid.

    Parameters
    ----------
    Same style as existing main.py call.

    Returns
    -------
    list[np.ndarray]
        List of normalized PHI fields for requested energy levels.
    """
    print('>>>MOLECULAR ORBITAL CALCULATION STARTING')

    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)

    if len(kpoint_coe_list) != len(kpoint_list) or len(kpoint_list) != len(phi_coe_list):
        raise ValueError("kpoint_coe_list, kpoint_list, and phi_coe_list must have the same length.")

    nk = len(kpoint_coe_list)
    V = abs(np.dot(cell_a, np.cross(cell_b, cell_c)))
    N = int(xyzgrid.size / 3)
    dV = V / N

    tasks = []
    for level in energylevel_list:
        for ik in range(nk):
            tasks.append((
                ifcrystal,
                cell_a,
                cell_b,
                cell_c,
                obtdictionary,
                atom_xyz,
                kpoint_coe_list[ik],
                kpoint_list[ik],
                phi_coe_list[ik],
                xyzgrid,
                level,
            ))

    with ThreadPoolExecutor(max_workers=ncpu) as executor:
        results = list(executor.map(lambda p: PHInk_c(*p), tasks))

    phi_fields = []
    idx = 0

    for level in energylevel_list:
        PHI = np.zeros(xyzgrid.shape[:3], dtype=complex)

        for _ in range(nk):
            PHI += results[idx]
            idx += 1

        norm = np.sqrt(1.0 / np.sum(PHI * np.conj(PHI) * dV))
        PHI *= norm

        phi_fields.append(PHI)

        filename = f'./01_results/MO_{level + 1}.cube'
        _write_cube(
            filename,
            PHI,
            atom_xyz,
            cell_a,
            cell_b,
            cell_c,
            origin_ang=origin_ang
        )

    return phi_fields
