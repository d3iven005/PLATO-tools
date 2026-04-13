from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.PHI_grad import PHInk_grad_c
from src.V_cell import Vcell


BOHR = 0.529177
WKED_CORRECTION = 0.99
RHO_CUTOFF = 1e-6

ELEMENT_LIST = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca'
]


def _occupied_levels_from_occ(energylevel_occ):
    levels = []
    for i, occ in enumerate(energylevel_occ):
        if occ == 1:
            levels.append(i)
        else:
            break
    return levels


def _write_cube(filename, data, atom_xyz, cell_a, cell_b, cell_c,origin_ang=None):
    data_real = np.asarray(np.real(data), dtype=float)
    Nx, Ny, Nz = data_real.shape
    Natom = len(atom_xyz)

    with open(filename, 'w') as f:
        f.write('cube file for electron localisation function\n')
        f.write('\n')     
        if origin_ang is None:
            origin_bohr = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            origin_bohr = np.asarray(origin_ang, dtype=float) / BOHR
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


def ELF(ifcrystal, cell_a, cell_b, cell_c, obtdictionary, atom_xyz,
        kpoint_coe_list, kpoint_list, phi_coe_list, xyzgrid, energylevel_occ, ncpu,origin_ang=None):

    print('>>>ELECTRON LOCALISATION FUNCTION CALCULATION STARTING')

    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)

    V = Vcell(cell_a, cell_b, cell_c)
    N = int(xyzgrid.size / 3)
    dV = V / N

    occupied_levels = _occupied_levels_from_occ(energylevel_occ)

    if len(kpoint_coe_list) != len(kpoint_list) or len(kpoint_list) != len(phi_coe_list):
        raise ValueError("kpoint_coe_list, kpoint_list, and phi_coe_list must have the same length.")

    nk = len(kpoint_coe_list)

    tasks = []
    for level in occupied_levels:
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
        results = list(executor.map(lambda p: PHInk_grad_c(*p), tasks))

    rho_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_x_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_y_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_z_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    KE_total = np.zeros(xyzgrid.shape[:3], dtype=float)

    idx = 0
    for level in occupied_levels:
        PHI = np.zeros(xyzgrid.shape[:3], dtype=complex)
        dPHI_dx = np.zeros(xyzgrid.shape[:3], dtype=complex)
        dPHI_dy = np.zeros(xyzgrid.shape[:3], dtype=complex)
        dPHI_dz = np.zeros(xyzgrid.shape[:3], dtype=complex)

        for _ in range(nk):
            phi, gx, gy, gz = results[idx]
            PHI += phi
            dPHI_dx += gx
            dPHI_dy += gy
            dPHI_dz += gz
            idx += 1

        norm = np.sqrt(1.0 / np.sum(PHI * np.conj(PHI) * dV))
        PHI *= norm
        dPHI_dx *= norm
        dPHI_dy *= norm
        dPHI_dz *= norm

        rho = np.real(np.conj(PHI) * PHI)

        grad_rho_x = 2.0 * np.real(np.conj(PHI) * dPHI_dx)
        grad_rho_y = 2.0 * np.real(np.conj(PHI) * dPHI_dy)
        grad_rho_z = 2.0 * np.real(np.conj(PHI) * dPHI_dz)

        KE = np.real(
            np.conj(dPHI_dx) * dPHI_dx
            + np.conj(dPHI_dy) * dPHI_dy
            + np.conj(dPHI_dz) * dPHI_dz
        )

        rho_total += rho
        grad_rho_x_total += grad_rho_x
        grad_rho_y_total += grad_rho_y
        grad_rho_z_total += grad_rho_z
        KE_total += KE

    WKED = 0.25 * WKED_CORRECTION * (
        grad_rho_x_total**2 + grad_rho_y_total**2 + grad_rho_z_total**2
    )

    rho_safe = np.where(rho_total > 1e-14, rho_total, 1e-14)
    UEG = 9.115599744691192 * (rho_safe ** (5.0 / 3.0))
    UEG_safe = np.where(UEG > 1e-14, UEG, 1e-14)

    chi = (KE_total - WKED / rho_safe) / UEG_safe
    elf = 1.0 / (1.0 + chi**2)
    elf = np.nan_to_num(elf)

    # mask low-density region
    elf = np.where(rho_total > RHO_CUTOFF, elf, 0.0)

    _write_cube('./01_results/CD.cube', rho_total, atom_xyz, cell_a, cell_b, cell_c,origin_ang=origin_ang)
    _write_cube('./01_results/ELF.cube', elf, atom_xyz, cell_a, cell_b, cell_c,origin_ang=origin_ang)

    return elf
