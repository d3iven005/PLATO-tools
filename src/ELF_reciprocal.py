from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.ELF import _occupation_matrix, _occupied_levels_from_occ, _write_cube
from src.PHI_grad import accumulate_level_chunk_all_k_values_from_local_basis
from src.V_cell import Vcell


RHO_CUTOFF = 1e-6
WKED_CORRECTION = 0.99


def _reciprocal_vectors(cell_a, cell_b, cell_c):
    volume_factor = np.dot(cell_a, np.cross(cell_b, cell_c))
    if abs(volume_factor) < 1e-14:
        raise ValueError("Cell volume is too small or zero; cannot build reciprocal lattice vectors.")

    reci_a = 2.0 * np.pi * np.cross(cell_b, cell_c) / volume_factor
    reci_b = 2.0 * np.pi * np.cross(cell_c, cell_a) / volume_factor
    reci_c = 2.0 * np.pi * np.cross(cell_a, cell_b) / volume_factor
    return reci_a, reci_b, reci_c


def _atom_index_chunks(n_atoms, ncpu):
    n_workers = max(1, min(int(ncpu), n_atoms))
    indices = np.arange(n_atoms)
    return [chunk.tolist() for chunk in np.array_split(indices, n_workers) if chunk.size > 0]


def _sum_phi_results(target_results, partial_results):
    for ik_local, partial_k in enumerate(partial_results):
        for ilevel, partial_phi in enumerate(partial_k):
            target_results[ik_local][ilevel] += partial_phi


def _k_vectors_from_reduced(kpoint_list, k_indices, cell_a, cell_b, cell_c):
    reci_a, reci_b, reci_c = _reciprocal_vectors(cell_a, cell_b, cell_c)
    k_vectors = []
    for ik in k_indices:
        kpoint = kpoint_list[ik]
        k_vectors.append(kpoint[0] * reci_a + kpoint[1] * reci_b + kpoint[2] * reci_c)
    return k_vectors


def _g_grids(shape, cell_a, cell_b, cell_c):
    nx, ny, nz = shape
    reci_a, reci_b, reci_c = _reciprocal_vectors(cell_a, cell_b, cell_c)
    gx_i = np.fft.fftfreq(nx, d=1.0 / nx)
    gy_i = np.fft.fftfreq(ny, d=1.0 / ny)
    gz_i = np.fft.fftfreq(nz, d=1.0 / nz)
    ii, jj, kk = np.meshgrid(gx_i, gy_i, gz_i, indexing='ij')
    gx = ii * reci_a[0] + jj * reci_b[0] + kk * reci_c[0]
    gy = ii * reci_a[1] + jj * reci_b[1] + kk * reci_c[1]
    gz = ii * reci_a[2] + jj * reci_b[2] + kk * reci_c[2]
    return gx, gy, gz


def _phase_grid(xyzgrid, k_vector):
    return np.exp(1j * np.dot(xyzgrid, np.asarray(k_vector, dtype=float)))


def _fft_bloch_gradient(phi, xyzgrid, k_vector, g_x, g_y, g_z):
    """
    Compute grad psi for a Bloch state psi = exp(i k.r) u(r).

    u is periodic, so grad u is evaluated with FFT:
        grad u(G) = i G u(G)

    Then:
        grad psi = exp(i k.r) [grad u + i k u]
    """
    phase = _phase_grid(xyzgrid, k_vector)
    u = phi / phase
    u_g = np.fft.fftn(u)

    du_dx = np.fft.ifftn(1j * g_x * u_g)
    du_dy = np.fft.ifftn(1j * g_y * u_g)
    du_dz = np.fft.ifftn(1j * g_z * u_g)

    dphi_dx = phase * (du_dx + 1j * k_vector[0] * u)
    dphi_dy = phase * (du_dy + 1j * k_vector[1] * u)
    dphi_dz = phase * (du_dz + 1j * k_vector[2] * u)
    return dphi_dx, dphi_dy, dphi_dz


def ELF_reciprocal(ifcrystal, cell_a, cell_b, cell_c, obtdictionary, atom_xyz,
                   kpoint_coe_list, kpoint_list, phi_coe_list, xyzgrid,
                   energylevel_occ, ncpu, origin_ang=None, level_chunk_size=2):
    if ifcrystal != 1:
        raise NotImplementedError("ELF_reciprocal currently supports periodic systems only.")

    print('>>>RECIPROCAL/FFT ELF CALCULATION STARTING')

    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)

    if len(kpoint_coe_list) != len(kpoint_list) or len(kpoint_list) != len(phi_coe_list):
        raise ValueError("kpoint_coe_list, kpoint_list, and phi_coe_list must have the same length.")

    nk = len(kpoint_coe_list)
    occ_matrix = _occupation_matrix(energylevel_occ, nk)
    occupied_levels = _occupied_levels_from_occ(occ_matrix)
    nonzero_k_indices = [ik for ik in range(nk) if abs(kpoint_coe_list[ik]) > 0.0]
    if not nonzero_k_indices:
        raise ValueError("No non-zero k-point weights are available for ELF calculation.")

    V = Vcell(cell_a, cell_b, cell_c)
    dV = V / int(xyzgrid.size / 3)
    g_x, g_y, g_z = _g_grids(xyzgrid.shape[:3], cell_a, cell_b, cell_c)

    rho_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_x_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_y_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    grad_rho_z_total = np.zeros(xyzgrid.shape[:3], dtype=float)
    KE_total = np.zeros(xyzgrid.shape[:3], dtype=float)

    atom_chunks = _atom_index_chunks(len(atom_xyz), ncpu)
    level_chunk_size = max(1, int(level_chunk_size))
    print('>>>RECIPROCAL ELF ATOM-PARALLEL WORKERS:', len(atom_chunks))
    print('>>>RECIPROCAL ELF LEVEL CHUNK SIZE:', level_chunk_size)

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
            phi_results = accumulate_level_chunk_all_k_values_from_local_basis(
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
            phi_results = [
                [np.zeros(xyzgrid.shape[:3], dtype=complex) for _ in level_chunk]
                for _ in active_k_indices
            ]
            with ThreadPoolExecutor(max_workers=len(atom_chunks)) as executor:
                futures = [
                    executor.submit(
                        accumulate_level_chunk_all_k_values_from_local_basis,
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
                    _sum_phi_results(phi_results, future.result())

        k_vectors = _k_vectors_from_reduced(kpoint_list, active_k_indices, cell_a, cell_b, cell_c)

        for ik_local, ik in enumerate(active_k_indices):
            k_vector = np.asarray(k_vectors[ik_local], dtype=float)
            for ilevel, phi in enumerate(phi_results[ik_local]):
                level = level_chunk[ilevel]
                contribution_weight = occ_matrix[ik, level] * kpoint_coe_list[ik]
                if contribution_weight <= 0.0:
                    continue

                norm = np.sqrt(1.0 / np.sum(phi * np.conj(phi) * dV))
                phi *= norm

                dphi_dx, dphi_dy, dphi_dz = _fft_bloch_gradient(
                    phi, xyzgrid, k_vector, g_x, g_y, g_z
                )

                rho = np.real(np.conj(phi) * phi)
                grad_rho_x = 2.0 * np.real(np.conj(phi) * dphi_dx)
                grad_rho_y = 2.0 * np.real(np.conj(phi) * dphi_dy)
                grad_rho_z = 2.0 * np.real(np.conj(phi) * dphi_dz)
                KE = np.real(
                    np.conj(dphi_dx) * dphi_dx
                    + np.conj(dphi_dy) * dphi_dy
                    + np.conj(dphi_dz) * dphi_dz
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
    elf = np.where(rho_total > RHO_CUTOFF, elf, 0.0)

    _write_cube('./01_results/CD_reciprocal.cube', rho_total, atom_xyz, cell_a, cell_b, cell_c, origin_ang=origin_ang)
    _write_cube('./01_results/ELF_reciprocal.cube', elf, atom_xyz, cell_a, cell_b, cell_c, origin_ang=origin_ang)

    return elf
