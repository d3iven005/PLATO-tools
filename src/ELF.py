from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.PHI_grad import accumulate_level_chunk_all_k_from_local_basis
from src.V_cell import Vcell


BOHR = 0.529177
WKED_CORRECTION = 0.99
RHO_CUTOFF = 1e-6
LEVEL_CHUNK_SIZE = 2

ELEMENT_LIST = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca'
]


def _occupation_matrix(energylevel_occ, nk):
    occ = np.asarray(energylevel_occ, dtype=float)
    if occ.ndim == 1:
        return np.tile(occ[np.newaxis, :], (nk, 1))
    if occ.ndim == 2:
        if occ.shape[0] != nk:
            raise ValueError(
                f"Occupation array has {occ.shape[0]} k blocks, expected {nk}."
            )
        return occ
    raise ValueError("energylevel_occ must be a 1D or 2D occupation array.")


def _occupied_levels_from_occ(occ_matrix):
    occupied = np.where(np.any(occ_matrix > 0.0, axis=0))[0]
    return occupied.tolist()


def _atom_index_chunks(n_atoms, ncpu):
    n_workers = max(1, min(int(ncpu), n_atoms))
    indices = np.arange(n_atoms)
    return [chunk.tolist() for chunk in np.array_split(indices, n_workers) if chunk.size > 0]


def _sum_chunk_results(target_results, partial_results):
    for ilevel, partial in enumerate(partial_results):
        for ifield in range(4):
            target_results[ilevel][ifield] += partial[ifield]


def _sum_k_chunk_results(target_results, partial_results):
    for ik_local, partial_k in enumerate(partial_results):
        for ilevel, partial_level in enumerate(partial_k):
            for ifield in range(4):
                target_results[ik_local][ilevel][ifield] += partial_level[ifield]


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
        kpoint_coe_list, kpoint_list, phi_coe_list, xyzgrid, energylevel_occ,
        ncpu, origin_ang=None, level_chunk_size=LEVEL_CHUNK_SIZE,
        grid_cell_a=None, grid_cell_b=None, grid_cell_c=None,
        normalize_orbitals=True):

    print('>>>ELECTRON LOCALISATION FUNCTION CALCULATION STARTING')

    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)
    if grid_cell_a is None:
        grid_cell_a = cell_a
    if grid_cell_b is None:
        grid_cell_b = cell_b
    if grid_cell_c is None:
        grid_cell_c = cell_c
    grid_cell_a = np.asarray(grid_cell_a, dtype=float)
    grid_cell_b = np.asarray(grid_cell_b, dtype=float)
    grid_cell_c = np.asarray(grid_cell_c, dtype=float)

    V = Vcell(grid_cell_a, grid_cell_b, grid_cell_c)
    N = int(xyzgrid.size / 3)
    dV = V / N

    if len(kpoint_coe_list) != len(kpoint_list) or len(kpoint_list) != len(phi_coe_list):
        raise ValueError("kpoint_coe_list, kpoint_list, and phi_coe_list must have the same length.")

    nk = len(kpoint_coe_list)
    occ_matrix = _occupation_matrix(energylevel_occ, nk)
    occupied_levels = _occupied_levels_from_occ(occ_matrix)
    nonzero_k_indices = [ik for ik in range(nk) if abs(kpoint_coe_list[ik]) > 0.0]
    if not nonzero_k_indices:
        raise ValueError("No non-zero k-point weights are available for ELF calculation.")

    rho_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_x_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_y_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_z_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    KE_total = np.zeros(xyzgrid.shape[:3], dtype=float)

    level_chunk_size = max(1, int(level_chunk_size))
    atom_chunks = _atom_index_chunks(len(atom_xyz), ncpu)
    print('>>>ELF ATOM-PARALLEL WORKERS:', len(atom_chunks))
    print('>>>ELF LEVEL CHUNK SIZE:', level_chunk_size)
    print('>>>ELF ORBITAL NORMALIZATION:', 'ON' if normalize_orbitals else 'OFF')

    for chunk_start in range(0, len(occupied_levels), level_chunk_size):
        level_chunk = occupied_levels[chunk_start:chunk_start + level_chunk_size]
        print("CALCULATING ENERGYLEVELS:", [level + 1 for level in level_chunk])

        active_k_indices = [
            ik for ik in nonzero_k_indices
            if np.any(occ_matrix[ik, level_chunk] > 0.0)
        ]
        if not active_k_indices:
            continue

        if len(atom_chunks) == 1:
            chunk_results = accumulate_level_chunk_all_k_from_local_basis(
                ifcrystal,
                cell_a,
                cell_b,
                cell_c,
                obtdictionary,
                atom_xyz,
                kpoint_list,
                phi_coe_list,
                xyzgrid,
                level_chunk,
                active_k_indices,
                atom_indices=atom_chunks[0],
            )
        else:
            chunk_results = [
                [
                    [
                        np.zeros(xyzgrid.shape[:3], dtype=complex),
                        np.zeros(xyzgrid.shape[:3], dtype=complex),
                        np.zeros(xyzgrid.shape[:3], dtype=complex),
                        np.zeros(xyzgrid.shape[:3], dtype=complex),
                    ]
                    for _ in level_chunk
                ]
                for _ in active_k_indices
            ]

            with ThreadPoolExecutor(max_workers=len(atom_chunks)) as executor:
                futures = [
                    executor.submit(
                        accumulate_level_chunk_all_k_from_local_basis,
                        ifcrystal,
                        cell_a,
                        cell_b,
                        cell_c,
                        obtdictionary,
                        atom_xyz,
                        kpoint_list,
                        phi_coe_list,
                        xyzgrid,
                        level_chunk,
                        active_k_indices,
                        atom_indices=atom_chunk,
                    )
                    for atom_chunk in atom_chunks
                ]

                for future in futures:
                    _sum_k_chunk_results(chunk_results, future.result())

        for ik_local, ik in enumerate(active_k_indices):
            for ilevel, (PHI, dPHI_dx, dPHI_dy, dPHI_dz) in enumerate(chunk_results[ik_local]):
                level = level_chunk[ilevel]
                contribution_weight = occ_matrix[ik, level] * kpoint_coe_list[ik]
                if contribution_weight <= 0.0:
                    continue

                if normalize_orbitals:
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

                rho_total += contribution_weight * rho
                grad_rho_x_total += contribution_weight * grad_rho_x
                grad_rho_y_total += contribution_weight * grad_rho_y
                grad_rho_z_total += contribution_weight * grad_rho_z
                KE_total += contribution_weight * KE

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

    _write_cube(
        './01_results/CD.cube',
        rho_total,
        atom_xyz,
        grid_cell_a,
        grid_cell_b,
        grid_cell_c,
        origin_ang=origin_ang,
    )
    _write_cube(
        './01_results/ELF.cube',
        elf,
        atom_xyz,
        grid_cell_a,
        grid_cell_b,
        grid_cell_c,
        origin_ang=origin_ang,
    )

    return elf
