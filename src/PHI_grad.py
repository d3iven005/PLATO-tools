import numpy as np

from src.count_obt import count_obt


BOHR = 0.529177
EPS_R = 1.0e-14
DEFAULT_TAPER_WIDTH = 0.3  # in the same unit as the radial table, usually Bohr
BLOCH_CACHE_MAX_BYTES = 512 * 1024 * 1024

NEIGHBOR_SHIFTS = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]


def _reciprocal_vectors(cell_a, cell_b, cell_c):
    volume_factor = np.dot(cell_a, np.cross(cell_b, cell_c))
    if abs(volume_factor) < 1e-14:
        raise ValueError("Cell volume is too small or zero; cannot build reciprocal lattice vectors.")

    reci_a = 2.0 * np.pi * np.cross(cell_b, cell_c) / volume_factor
    reci_b = 2.0 * np.pi * np.cross(cell_c, cell_a) / volume_factor
    reci_c = 2.0 * np.pi * np.cross(cell_a, cell_b) / volume_factor
    return reci_a, reci_b, reci_c


def _atom_position_bohr(atom_row):
    return np.array(
        [float(atom_row[1]), float(atom_row[2]), float(atom_row[3])],
        dtype=float
    ) / BOHR


def _orbital_offsets(obtinfo):
    return np.cumsum(np.r_[0, obtinfo[:-1]])


def _zero_field(xyzgrid):
    return np.zeros(xyzgrid.shape[:3], dtype=complex)


def _is_axis_aligned_cell(cell_a, cell_b, cell_c):
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)
    return (
        abs(cell_a[1]) < 1e-14 and abs(cell_a[2]) < 1e-14
        and abs(cell_b[0]) < 1e-14 and abs(cell_b[2]) < 1e-14
        and abs(cell_c[0]) < 1e-14 and abs(cell_c[1]) < 1e-14
    )


def _local_slice_for_cutoff(xyzgrid, atom_pos, cutoff, cell_a, cell_b, cell_c):
    """
    Return a rectangular grid slice around atom_pos for axis-aligned cells.

    If the cell is not axis-aligned, fall back to the full grid. The radial
    interpolation still zeros values outside cutoff; this fallback preserves
    correctness while keeping the optimized path simple for the current MGO use.
    """
    if not _is_axis_aligned_cell(cell_a, cell_b, cell_c):
        return (slice(None), slice(None), slice(None))

    axes = (
        xyzgrid[:, 0, 0, 0],
        xyzgrid[0, :, 0, 1],
        xyzgrid[0, 0, :, 2],
    )

    slices = []
    for dim, coords in enumerate(axes):
        lo = atom_pos[dim] - cutoff
        hi = atom_pos[dim] + cutoff
        idx = np.where((coords >= lo) & (coords <= hi))[0]
        if idx.size == 0:
            return None
        slices.append(slice(int(idx[0]), int(idx[-1]) + 1))

    return tuple(slices)


def _axis_phase_cache(xyzgrid, k_vectors, cell_a, cell_b, cell_c):
    """
    Precompute separable Bloch phases for axis-aligned grids.

    exp(i k.r) = exp(i kx x) exp(i ky y) exp(i kz z)
    """
    if not _is_axis_aligned_cell(cell_a, cell_b, cell_c):
        return None

    x_axis = xyzgrid[:, 0, 0, 0]
    y_axis = xyzgrid[0, :, 0, 1]
    z_axis = xyzgrid[0, 0, :, 2]

    cache = []
    for k_vector in k_vectors:
        k_vector = np.asarray(k_vector, dtype=float)
        cache.append((
            np.exp(1j * k_vector[0] * x_axis),
            np.exp(1j * k_vector[1] * y_axis),
            np.exp(1j * k_vector[2] * z_axis),
        ))
    return cache


def _full_bloch_phase_cache(xyzgrid, k_vectors, max_bytes=BLOCH_CACHE_MAX_BYTES):
    """
    Precompute exp(i k.r) on the full grid for each k-vector.

    This is faster than rebuilding local phases for every atom/image when the
    cache fits comfortably in memory. If it would be too large, return None.
    """
    n_k = len(k_vectors)
    n_grid = int(np.prod(xyzgrid.shape[:3]))
    estimated_bytes = n_k * n_grid * np.dtype(np.complex128).itemsize
    if estimated_bytes > max_bytes:
        return None

    return [
        np.exp(1j * np.dot(xyzgrid, np.asarray(k_vector, dtype=float)))
        for k_vector in k_vectors
    ]


def _bloch_from_axis_phase(axis_phase, local_slice):
    sx, sy, sz = local_slice
    phase_x, phase_y, phase_z = axis_phase
    return (
        phase_x[sx][:, np.newaxis, np.newaxis]
        * phase_y[sy][np.newaxis, :, np.newaxis]
        * phase_z[sz][np.newaxis, np.newaxis, :]
    )


def _orbital_cutoff(OBT):
    OBT = np.asarray(OBT, dtype=float)
    return float(OBT[-1, 0])


def _basis_table_keys(obt_count, elementtype):
    if obt_count == 1:
        return ['Hs']
    if obt_count == 4:
        return [elementtype + 's', elementtype + 'p']
    if obt_count == 9:
        return []
    raise ValueError(f"Unsupported orbital count: {obt_count}")


def _basis_cutoff(obt_count, elementtype, obtdictionary):
    keys = _basis_table_keys(obt_count, elementtype)
    if not keys:
        return 0.0
    return max(_orbital_cutoff(obtdictionary[key]) for key in keys)


def _smooth_switch(r, r_start, r_cut):
    """
    Quintic smooth switch:
    - s(r) = 1, for r <= r_start
    - s(r) = 0, for r >= r_cut
    - smooth in between, with zero first and second derivatives at both ends

    Returns
    -------
    s : np.ndarray
        switch function
    dsdr : np.ndarray
        derivative ds/dr
    """
    s = np.ones_like(r, dtype=float)
    dsdr = np.zeros_like(r, dtype=float)

    mask_mid = (r > r_start) & (r < r_cut)
    mask_hi = r >= r_cut

    t = np.zeros_like(r, dtype=float)
    t[mask_mid] = (r[mask_mid] - r_start) / (r_cut - r_start)

    # s(t) = 1 - 10 t^3 + 15 t^4 - 6 t^5
    s[mask_mid] = (
        1.0
        - 10.0 * t[mask_mid]**3
        + 15.0 * t[mask_mid]**4
        - 6.0 * t[mask_mid]**5
    )
    s[mask_hi] = 0.0

    # ds/dt = -30 t^2 + 60 t^3 - 30 t^4
    dsdt = np.zeros_like(r, dtype=float)
    dsdt[mask_mid] = (
        -30.0 * t[mask_mid]**2
        + 60.0 * t[mask_mid]**3
        - 30.0 * t[mask_mid]**4
    )
    dsdr[mask_mid] = dsdt[mask_mid] / (r_cut - r_start)

    return s, dsdr


def _interp_radial_smooth(OBT, r, taper_width=DEFAULT_TAPER_WIDTH):
    """
    Interpolate radial orbital and its radial derivative, with smooth taper near cutoff.

    Parameters
    ----------
    OBT : np.ndarray
        shape (N, 2), first column radius, second column radial value
    r : np.ndarray
        distance array
    taper_width : float
        width of the smoothing region before cutoff

    Returns
    -------
    Rval : np.ndarray
    dRdr : np.ndarray
    """
    OBT = np.asarray(OBT, dtype=float)

    if OBT.ndim != 2 or OBT.shape[1] != 2:
        raise ValueError(f"OBT must have shape (N, 2), got {OBT.shape}")

    x = OBT[:, 0]
    y = OBT[:, 1]

    if np.any(np.diff(x) < 0):
        raise ValueError("Radial grid in OBT must be monotonically increasing.")

    r_cut = x[-1]

    # Raw interpolation
    R_raw = np.where(r > r_cut, 0.0, np.interp(r, x, y))
    dRdr_raw_table = np.gradient(y, x)
    dRdr_raw = np.where(r > r_cut, 0.0, np.interp(r, x, dRdr_raw_table))

    # Smooth taper region
    r_start = max(x[0], r_cut - taper_width)

    # If taper region is too small, fall back to raw interpolation
    if r_start >= r_cut - 1e-12:
        return R_raw, dRdr_raw

    s, dsdr = _smooth_switch(r, r_start, r_cut)

    # Product rule:
    # d/dr [R_raw * s] = dR_raw/dr * s + R_raw * ds/dr
    Rval = R_raw * s
    dRdr = dRdr_raw * s + R_raw * dsdr

    return Rval, dRdr


def _interp_radial_smooth_value(OBT, r, taper_width=DEFAULT_TAPER_WIDTH):
    """
    Interpolate radial orbital values with the same smooth cutoff used by the
    analytical-gradient path, but do not compute radial derivatives.
    """
    OBT = np.asarray(OBT, dtype=float)

    if OBT.ndim != 2 or OBT.shape[1] != 2:
        raise ValueError(f"OBT must have shape (N, 2), got {OBT.shape}")

    x = OBT[:, 0]
    y = OBT[:, 1]

    if np.any(np.diff(x) < 0):
        raise ValueError("Radial grid in OBT must be monotonically increasing.")

    r_cut = x[-1]
    R_raw = np.where(r > r_cut, 0.0, np.interp(r, x, y))
    r_start = max(x[0], r_cut - taper_width)

    if r_start >= r_cut - 1e-12:
        return R_raw

    s, _ = _smooth_switch(r, r_start, r_cut)
    return R_raw * s


def _angular_value_and_gradient(grid, atomic_position, m, l):
    """
    Return Y and its Cartesian derivatives:
        Y, dYdx, dYdy, dYdz

    Current conventions follow your Y_cal.py:
    - l=0,m=0 : s
    - l=1,m=0 : pz
    - l=1,m=1 : px
    - l=1,m=-1: py
    """
    vector_M = grid - atomic_position
    x = vector_M[..., 0]
    y = vector_M[..., 1]
    z = vector_M[..., 2]

    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r_safe = np.where(r > EPS_R, r, 1.0)
    r3_safe = r_safe**3

    if l == 0:
        Y = np.full(r.shape, 1.0 / (2.0 * np.sqrt(np.pi)), dtype=float)
        dYdx = np.zeros_like(r, dtype=float)
        dYdy = np.zeros_like(r, dtype=float)
        dYdz = np.zeros_like(r, dtype=float)
        return Y, dYdx, dYdy, dYdz

    if l == 1:
        pref = 0.5 * np.sqrt(3.0 / np.pi)

        if m == 1:  # px ~ x/r
            Y = pref * (x / r_safe)
            dYdx = pref * ((r2 - x**2) / r3_safe)
            dYdy = pref * (-(x * y) / r3_safe)
            dYdz = pref * (-(x * z) / r3_safe)

        elif m == -1:  # py ~ y/r
            Y = pref * (y / r_safe)
            dYdx = pref * (-(x * y) / r3_safe)
            dYdy = pref * ((r2 - y**2) / r3_safe)
            dYdz = pref * (-(y * z) / r3_safe)

        elif m == 0:  # pz ~ z/r
            Y = pref * (z / r_safe)
            dYdx = pref * (-(x * z) / r3_safe)
            dYdy = pref * (-(y * z) / r3_safe)
            dYdz = pref * ((r2 - z**2) / r3_safe)

        else:
            raise ValueError(f"Unsupported m={m} for l=1")

        # Safe value at r=0
        Y = np.where(r > EPS_R, Y, 0.0)
        dYdx = np.where(r > EPS_R, dYdx, 0.0)
        dYdy = np.where(r > EPS_R, dYdy, 0.0)
        dYdz = np.where(r > EPS_R, dYdz, 0.0)

        return Y, dYdx, dYdy, dYdz

    raise ValueError(f"Unsupported l={l}; currently only l=0 and l=1 are implemented")


def _angular_value(grid, atomic_position, m, l):
    vector_M = grid - atomic_position
    x = vector_M[..., 0]
    y = vector_M[..., 1]
    z = vector_M[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    if l == 0:
        return np.full(r.shape, 1.0 / (2.0 * np.sqrt(np.pi)), dtype=float)

    if l == 1:
        r_safe = np.where(r > EPS_R, r, 1.0)
        pref = 0.5 * np.sqrt(3.0 / np.pi)
        if m == 1:
            Y = pref * (x / r_safe)
        elif m == -1:
            Y = pref * (y / r_safe)
        elif m == 0:
            Y = pref * (z / r_safe)
        else:
            raise ValueError(f"Unsupported m={m} for l=1")
        return np.where(r > EPS_R, Y, 0.0)

    raise ValueError(f"Unsupported l={l}; currently only l=0 and l=1 are implemented")


def _radial_value_and_gradient(OBT, grid, atomic_position, taper_width=DEFAULT_TAPER_WIDTH):
    """
    Return:
        Rval, dRdx, dRdy, dRdz

    where grad R = (dR/dr) * rhat
    """
    vector_M = grid - atomic_position
    x = vector_M[..., 0]
    y = vector_M[..., 1]
    z = vector_M[..., 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > EPS_R, r, 1.0)

    Rval, dRdr = _interp_radial_smooth(OBT, r, taper_width=taper_width)

    dRdx = dRdr * (x / r_safe)
    dRdy = dRdr * (y / r_safe)
    dRdz = dRdr * (z / r_safe)

    dRdx = np.where(r > EPS_R, dRdx, 0.0)
    dRdy = np.where(r > EPS_R, dRdy, 0.0)
    dRdz = np.where(r > EPS_R, dRdz, 0.0)

    return Rval, dRdx, dRdy, dRdz


def _basis_value_and_gradient(obt_count, elementtype, atom_pos, xyzgrid, obtdictionary):
    """
    Build basis functions and their Cartesian gradients centered at atom_pos.

    Return format:
    [
      (chi, dchi_dx, dchi_dy, dchi_dz),
      ...
    ]

    Ordering:
    - obt_count == 1: [s]
    - obt_count == 4: [s, pz, px, py]
    """
    if obt_count == 1:
        Rv, dRdx, dRdy, dRdz = _radial_value_and_gradient(
            obtdictionary['Hs'], xyzgrid, atom_pos
        )
        Yv, dYdx, dYdy, dYdz = _angular_value_and_gradient(
            xyzgrid, atom_pos, 0, 0
        )

        chi = Rv * Yv
        dchi_dx = dRdx * Yv + Rv * dYdx
        dchi_dy = dRdy * Yv + Rv * dYdy
        dchi_dz = dRdz * Yv + Rv * dYdz

        return [(chi, dchi_dx, dchi_dy, dchi_dz)]

    if obt_count == 4:
        basis_list = []

        # s
        Rv_s, dRdx_s, dRdy_s, dRdz_s = _radial_value_and_gradient(
            obtdictionary[elementtype + 's'], xyzgrid, atom_pos
        )
        Yv_s, dYdx_s, dYdy_s, dYdz_s = _angular_value_and_gradient(
            xyzgrid, atom_pos, 0, 0
        )

        chi_s = Rv_s * Yv_s
        dchi_s_dx = dRdx_s * Yv_s + Rv_s * dYdx_s
        dchi_s_dy = dRdy_s * Yv_s + Rv_s * dYdy_s
        dchi_s_dz = dRdz_s * Yv_s + Rv_s * dYdz_s
        basis_list.append((chi_s, dchi_s_dx, dchi_s_dy, dchi_s_dz))

        # p radial part is shared
        Rv_p, dRdx_p, dRdy_p, dRdz_p = _radial_value_and_gradient(
            obtdictionary[elementtype + 'p'], xyzgrid, atom_pos
        )

        # pz
        Yv, dYdx, dYdy, dYdz = _angular_value_and_gradient(
            xyzgrid, atom_pos, 0, 1
        )
        chi = Rv_p * Yv
        dchi_dx = dRdx_p * Yv + Rv_p * dYdx
        dchi_dy = dRdy_p * Yv + Rv_p * dYdy
        dchi_dz = dRdz_p * Yv + Rv_p * dYdz
        basis_list.append((chi, dchi_dx, dchi_dy, dchi_dz))

        # px
        Yv, dYdx, dYdy, dYdz = _angular_value_and_gradient(
            xyzgrid, atom_pos, 1, 1
        )
        chi = Rv_p * Yv
        dchi_dx = dRdx_p * Yv + Rv_p * dYdx
        dchi_dy = dRdy_p * Yv + Rv_p * dYdy
        dchi_dz = dRdz_p * Yv + Rv_p * dYdz
        basis_list.append((chi, dchi_dx, dchi_dy, dchi_dz))

        # py
        Yv, dYdx, dYdy, dYdz = _angular_value_and_gradient(
            xyzgrid, atom_pos, -1, 1
        )
        chi = Rv_p * Yv
        dchi_dx = dRdx_p * Yv + Rv_p * dYdx
        dchi_dy = dRdy_p * Yv + Rv_p * dYdy
        dchi_dz = dRdz_p * Yv + Rv_p * dYdz
        basis_list.append((chi, dchi_dx, dchi_dy, dchi_dz))

        return basis_list

    if obt_count == 9:
        # d orbitals not implemented yet
        return []

    raise ValueError(f"Unsupported orbital count: {obt_count}")


def _basis_value_only(obt_count, elementtype, atom_pos, xyzgrid, obtdictionary):
    """
    Build basis function values without Cartesian gradients.

    Ordering matches _basis_value_and_gradient:
    - obt_count == 1: [s]
    - obt_count == 4: [s, pz, px, py]
    """
    vector_M = xyzgrid - atom_pos
    r = np.sqrt(
        vector_M[..., 0]**2
        + vector_M[..., 1]**2
        + vector_M[..., 2]**2
    )

    if obt_count == 1:
        Rv = _interp_radial_smooth_value(obtdictionary['Hs'], r)
        return [Rv * _angular_value(xyzgrid, atom_pos, 0, 0)]

    if obt_count == 4:
        Rs = _interp_radial_smooth_value(obtdictionary[elementtype + 's'], r)
        Rp = _interp_radial_smooth_value(obtdictionary[elementtype + 'p'], r)
        chi_s = Rs * _angular_value(xyzgrid, atom_pos, 0, 0)
        chi_pz = Rp * _angular_value(xyzgrid, atom_pos, 0, 1)
        chi_px = Rp * _angular_value(xyzgrid, atom_pos, 1, 1)
        chi_py = Rp * _angular_value(xyzgrid, atom_pos, -1, 1)
        return [chi_s, chi_pz, chi_px, chi_py]

    if obt_count == 9:
        return []

    raise ValueError(f"Unsupported orbital count: {obt_count}")


def accumulate_level_chunk_from_local_basis(
    ifcrystal,
    cell_a,
    cell_b,
    cell_c,
    obtdictionary,
    atom_xyz,
    kpoint_coe_list,
    kpoint_list,
    phi_coe_list,
    xyzgrid,
    levels,
    k_indices,
    atom_indices=None,
):
    """
    Accumulate PHI and gradients for a small chunk of energy levels.

    Basis fields are computed once per atom/image on a cutoff-local grid slice,
    then reused across all requested k-points and levels in this chunk. This is
    the memory-bounded optimized path used by ELF.
    """
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)
    levels = list(levels)

    PHI = [_zero_field(xyzgrid) for _ in levels]
    dPHI_dx = [_zero_field(xyzgrid) for _ in levels]
    dPHI_dy = [_zero_field(xyzgrid) for _ in levels]
    dPHI_dz = [_zero_field(xyzgrid) for _ in levels]

    obtinfo = count_obt(atom_xyz)
    offsets = _orbital_offsets(obtinfo)
    if atom_indices is None:
        atom_indices = range(len(atom_xyz))

    if ifcrystal == 1:
        shifts = NEIGHBOR_SHIFTS
        reci_a, reci_b, reci_c = _reciprocal_vectors(cell_a, cell_b, cell_c)
        k_vectors = []
        for ik in k_indices:
            kpoint = kpoint_list[ik]
            k_vectors.append(kpoint[0] * reci_a + kpoint[1] * reci_b + kpoint[2] * reci_c)
    elif ifcrystal == 0:
        shifts = [(0, 0, 0)]
        k_vectors = []
        k_indices = [0]
    else:
        raise ValueError(f"ifcrystal must be 0 or 1, got {ifcrystal}")

    for i in atom_indices:
        atom = atom_xyz[i]
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos0 = _atom_position_bohr(atom)
        cutoff = _basis_cutoff(obt_count, elementtype, obtdictionary)

        if cutoff <= 0.0:
            continue

        for ii, jj, kk in shifts:
            atom_pos = atom_pos0 + ii * cell_a + jj * cell_b + kk * cell_c
            local_slice = _local_slice_for_cutoff(
                xyzgrid, atom_pos, cutoff, cell_a, cell_b, cell_c
            )
            if local_slice is None:
                continue

            local_grid = xyzgrid[local_slice]
            basis_list = _basis_value_and_gradient(
                obt_count=obt_count,
                elementtype=elementtype,
                atom_pos=atom_pos,
                xyzgrid=local_grid,
                obtdictionary=obtdictionary,
            )

            if ifcrystal == 0:
                phi_coe = phi_coe_list[0]
                for ib, (chi, dchi_x, dchi_y, dchi_z) in enumerate(basis_list):
                    basis_index = offset + ib
                    for ilevel, level in enumerate(levels):
                        coeff = phi_coe[level][basis_index]
                        PHI[ilevel][local_slice] += coeff * chi
                        dPHI_dx[ilevel][local_slice] += coeff * dchi_x
                        dPHI_dy[ilevel][local_slice] += coeff * dchi_y
                        dPHI_dz[ilevel][local_slice] += coeff * dchi_z
                continue

            for ik, k_vector in zip(k_indices, k_vectors):
                k_vector = np.asarray(k_vector, dtype=float)
                phase_atom = np.exp(-1j * np.dot(k_vector, atom_pos))
                bloch = np.exp(1j * np.dot(local_grid, k_vector)) * phase_atom

                ikx = 1j * k_vector[0]
                iky = 1j * k_vector[1]
                ikz = 1j * k_vector[2]
                phi_coe = phi_coe_list[ik]

                for ib, (chi, dchi_x, dchi_y, dchi_z) in enumerate(basis_list):
                    basis_index = offset + ib
                    chi_bloch = chi * bloch
                    grad_x = dchi_x * bloch + chi * ikx * bloch
                    grad_y = dchi_y * bloch + chi * iky * bloch
                    grad_z = dchi_z * bloch + chi * ikz * bloch

                    for ilevel, level in enumerate(levels):
                        coeff = phi_coe[level][basis_index]
                        PHI[ilevel][local_slice] += coeff * chi_bloch
                        dPHI_dx[ilevel][local_slice] += coeff * grad_x
                        dPHI_dy[ilevel][local_slice] += coeff * grad_y
                        dPHI_dz[ilevel][local_slice] += coeff * grad_z

    return list(zip(PHI, dPHI_dx, dPHI_dy, dPHI_dz))


def accumulate_level_chunk_all_k_from_local_basis(
    ifcrystal,
    cell_a,
    cell_b,
    cell_c,
    obtdictionary,
    atom_xyz,
    kpoint_list,
    phi_coe_list,
    xyzgrid,
    levels,
    k_indices,
    atom_indices=None,
):
    """
    Accumulate PHI and gradients for all requested k-points in one pass.

    For each atom/image, basis values and gradients are built once on the local
    cutoff slice, then reused across every requested k-point and band in
    ``levels``. Return shape is:

        results[ik_local][ilevel] = (PHI, dPHI_dx, dPHI_dy, dPHI_dz)
    """
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)
    levels = list(levels)
    k_indices = list(k_indices)

    if ifcrystal == 0:
        k_indices = [0]

    results = [
        [
            [_zero_field(xyzgrid), _zero_field(xyzgrid), _zero_field(xyzgrid), _zero_field(xyzgrid)]
            for _ in levels
        ]
        for _ in k_indices
    ]

    obtinfo = count_obt(atom_xyz)
    offsets = _orbital_offsets(obtinfo)
    if atom_indices is None:
        atom_indices = range(len(atom_xyz))

    if ifcrystal == 1:
        shifts = NEIGHBOR_SHIFTS
        reci_a, reci_b, reci_c = _reciprocal_vectors(cell_a, cell_b, cell_c)
        k_vectors = []
        for ik in k_indices:
            kpoint = kpoint_list[ik]
            k_vectors.append(kpoint[0] * reci_a + kpoint[1] * reci_b + kpoint[2] * reci_c)
        full_phase_cache = _full_bloch_phase_cache(xyzgrid, k_vectors)
        axis_phase_cache = None
        if full_phase_cache is None:
            axis_phase_cache = _axis_phase_cache(xyzgrid, k_vectors, cell_a, cell_b, cell_c)
    elif ifcrystal == 0:
        shifts = [(0, 0, 0)]
        k_vectors = [None]
        full_phase_cache = None
        axis_phase_cache = None
    else:
        raise ValueError(f"ifcrystal must be 0 or 1, got {ifcrystal}")

    for i in atom_indices:
        atom = atom_xyz[i]
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos0 = _atom_position_bohr(atom)
        cutoff = _basis_cutoff(obt_count, elementtype, obtdictionary)

        if cutoff <= 0.0:
            continue

        for ii, jj, kk in shifts:
            atom_pos = atom_pos0 + ii * cell_a + jj * cell_b + kk * cell_c
            local_slice = _local_slice_for_cutoff(
                xyzgrid, atom_pos, cutoff, cell_a, cell_b, cell_c
            )
            if local_slice is None:
                continue

            local_grid = xyzgrid[local_slice]
            basis_list = _basis_value_and_gradient(
                obt_count=obt_count,
                elementtype=elementtype,
                atom_pos=atom_pos,
                xyzgrid=local_grid,
                obtdictionary=obtdictionary,
            )

            if ifcrystal == 0:
                phi_coe = phi_coe_list[0]
                for ib, (chi, dchi_x, dchi_y, dchi_z) in enumerate(basis_list):
                    basis_index = offset + ib
                    for ilevel, level in enumerate(levels):
                        coeff = phi_coe[level][basis_index]
                        fields = results[0][ilevel]
                        fields[0][local_slice] += coeff * chi
                        fields[1][local_slice] += coeff * dchi_x
                        fields[2][local_slice] += coeff * dchi_y
                        fields[3][local_slice] += coeff * dchi_z
                continue

            for ik_local, (ik, k_vector) in enumerate(zip(k_indices, k_vectors)):
                k_vector = np.asarray(k_vector, dtype=float)
                phase_atom = np.exp(-1j * np.dot(k_vector, atom_pos))
                if full_phase_cache is not None:
                    bloch_grid = full_phase_cache[ik_local][local_slice]
                elif axis_phase_cache is not None:
                    bloch_grid = _bloch_from_axis_phase(axis_phase_cache[ik_local], local_slice)
                else:
                    bloch_grid = np.exp(1j * np.dot(local_grid, k_vector))
                bloch = bloch_grid * phase_atom

                ikx = 1j * k_vector[0]
                iky = 1j * k_vector[1]
                ikz = 1j * k_vector[2]
                phi_coe = phi_coe_list[ik]

                for ib, (chi, dchi_x, dchi_y, dchi_z) in enumerate(basis_list):
                    basis_index = offset + ib
                    chi_bloch = chi * bloch
                    grad_x = dchi_x * bloch + chi * ikx * bloch
                    grad_y = dchi_y * bloch + chi * iky * bloch
                    grad_z = dchi_z * bloch + chi * ikz * bloch

                    for ilevel, level in enumerate(levels):
                        coeff = phi_coe[level][basis_index]
                        fields = results[ik_local][ilevel]
                        fields[0][local_slice] += coeff * chi_bloch
                        fields[1][local_slice] += coeff * grad_x
                        fields[2][local_slice] += coeff * grad_y
                        fields[3][local_slice] += coeff * grad_z

    return results


def accumulate_level_chunk_all_k_values_from_local_basis(
    ifcrystal,
    cell_a,
    cell_b,
    cell_c,
    obtdictionary,
    atom_xyz,
    kpoint_list,
    phi_coe_list,
    xyzgrid,
    levels,
    k_indices,
    atom_indices=None,
):
    """
    Accumulate PHI values for all requested k-points in one pass.

    Return shape:
        results[ik_local][ilevel] = PHI
    """
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)
    levels = list(levels)
    k_indices = list(k_indices)

    if ifcrystal == 0:
        k_indices = [0]

    results = [
        [_zero_field(xyzgrid) for _ in levels]
        for _ in k_indices
    ]

    obtinfo = count_obt(atom_xyz)
    offsets = _orbital_offsets(obtinfo)
    if atom_indices is None:
        atom_indices = range(len(atom_xyz))

    if ifcrystal == 1:
        shifts = NEIGHBOR_SHIFTS
        reci_a, reci_b, reci_c = _reciprocal_vectors(cell_a, cell_b, cell_c)
        k_vectors = []
        for ik in k_indices:
            kpoint = kpoint_list[ik]
            k_vectors.append(kpoint[0] * reci_a + kpoint[1] * reci_b + kpoint[2] * reci_c)
        full_phase_cache = _full_bloch_phase_cache(xyzgrid, k_vectors)
        axis_phase_cache = None
        if full_phase_cache is None:
            axis_phase_cache = _axis_phase_cache(xyzgrid, k_vectors, cell_a, cell_b, cell_c)
    elif ifcrystal == 0:
        shifts = [(0, 0, 0)]
        k_vectors = [None]
        full_phase_cache = None
        axis_phase_cache = None
    else:
        raise ValueError(f"ifcrystal must be 0 or 1, got {ifcrystal}")

    for i in atom_indices:
        atom = atom_xyz[i]
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos0 = _atom_position_bohr(atom)
        cutoff = _basis_cutoff(obt_count, elementtype, obtdictionary)

        if cutoff <= 0.0:
            continue

        for ii, jj, kk in shifts:
            atom_pos = atom_pos0 + ii * cell_a + jj * cell_b + kk * cell_c
            local_slice = _local_slice_for_cutoff(
                xyzgrid, atom_pos, cutoff, cell_a, cell_b, cell_c
            )
            if local_slice is None:
                continue

            local_grid = xyzgrid[local_slice]
            basis_list = _basis_value_only(
                obt_count=obt_count,
                elementtype=elementtype,
                atom_pos=atom_pos,
                xyzgrid=local_grid,
                obtdictionary=obtdictionary,
            )

            if ifcrystal == 0:
                phi_coe = phi_coe_list[0]
                for ib, chi in enumerate(basis_list):
                    basis_index = offset + ib
                    for ilevel, level in enumerate(levels):
                        coeff = phi_coe[level][basis_index]
                        results[0][ilevel][local_slice] += coeff * chi
                continue

            for ik_local, (ik, k_vector) in enumerate(zip(k_indices, k_vectors)):
                k_vector = np.asarray(k_vector, dtype=float)
                phase_atom = np.exp(-1j * np.dot(k_vector, atom_pos))
                if full_phase_cache is not None:
                    bloch_grid = full_phase_cache[ik_local][local_slice]
                elif axis_phase_cache is not None:
                    bloch_grid = _bloch_from_axis_phase(axis_phase_cache[ik_local], local_slice)
                else:
                    bloch_grid = np.exp(1j * np.dot(local_grid, k_vector))
                bloch = bloch_grid * phase_atom
                phi_coe = phi_coe_list[ik]

                for ib, chi in enumerate(basis_list):
                    basis_index = offset + ib
                    chi_bloch = chi * bloch
                    for ilevel, level in enumerate(levels):
                        coeff = phi_coe[level][basis_index]
                        results[ik_local][ilevel][local_slice] += coeff * chi_bloch

    return results


def _accumulate_nonperiodic(phi_coe_level, atom_xyz, obtinfo, offsets, xyzgrid, obtdictionary):
    PHI = _zero_field(xyzgrid)
    dPHI_dx = _zero_field(xyzgrid)
    dPHI_dy = _zero_field(xyzgrid)
    dPHI_dz = _zero_field(xyzgrid)

    for i, atom in enumerate(atom_xyz):
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos = _atom_position_bohr(atom)

        basis_list = _basis_value_and_gradient(
            obt_count=obt_count,
            elementtype=elementtype,
            atom_pos=atom_pos,
            xyzgrid=xyzgrid,
            obtdictionary=obtdictionary,
        )

        for ib, (chi, dchi_dx, dchi_dy, dchi_dz) in enumerate(basis_list):
            coeff = phi_coe_level[offset + ib]
            PHI += coeff * chi
            dPHI_dx += coeff * dchi_dx
            dPHI_dy += coeff * dchi_dy
            dPHI_dz += coeff * dchi_dz

    return PHI, dPHI_dx, dPHI_dy, dPHI_dz


def _accumulate_periodic(
    phi_coe_level,
    atom_xyz,
    obtinfo,
    offsets,
    xyzgrid,
    obtdictionary,
    cell_a,
    cell_b,
    cell_c,
    k_vector,
    kpoint_coe,
):
    PHI = _zero_field(xyzgrid)
    dPHI_dx = _zero_field(xyzgrid)
    dPHI_dy = _zero_field(xyzgrid)
    dPHI_dz = _zero_field(xyzgrid)

    bloch_grid = np.exp(1j * np.dot(xyzgrid, k_vector))

    ikx = 1j * k_vector[0]
    iky = 1j * k_vector[1]
    ikz = 1j * k_vector[2]

    for i, atom in enumerate(atom_xyz):
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos0 = _atom_position_bohr(atom)

        for ii, jj, kk in NEIGHBOR_SHIFTS:
            atom_pos = atom_pos0 + ii * cell_a + jj * cell_b + kk * cell_c

            phase_atom = np.exp(-1j * np.dot(k_vector, atom_pos))
            bloch = bloch_grid * phase_atom

            basis_list = _basis_value_and_gradient(
                obt_count=obt_count,
                elementtype=elementtype,
                atom_pos=atom_pos,
                xyzgrid=xyzgrid,
                obtdictionary=obtdictionary,
            )

            for ib, (chi, dchi_dx, dchi_dy, dchi_dz) in enumerate(basis_list):
                coeff = kpoint_coe * phi_coe_level[offset + ib]

                PHI += coeff * chi * bloch
                dPHI_dx += coeff * (dchi_dx * bloch + chi * ikx * bloch)
                dPHI_dy += coeff * (dchi_dy * bloch + chi * iky * bloch)
                dPHI_dz += coeff * (dchi_dz * bloch + chi * ikz * bloch)

    return PHI, dPHI_dx, dPHI_dy, dPHI_dz


def PHInk_grad_c(ifcrystal, cell_a, cell_b, cell_c, obtdictionary,
                 atom_xyz, kpoint_coe, kpoint, phi_coe, xyzgrid, energylevel):
    """
    Construct PHI and its Cartesian gradients on xyzgrid for one energy level.

    Returns
    -------
    PHI, dPHI_dx, dPHI_dy, dPHI_dz
    """
    obtinfo = count_obt(atom_xyz)
    offsets = _orbital_offsets(obtinfo)
    phi_coe_level = phi_coe[energylevel]

    if ifcrystal == 0:
        print("NOCALCULATING ENERGYLEVEL:", energylevel + 1)
        return _accumulate_nonperiodic(
            phi_coe_level=phi_coe_level,
            atom_xyz=atom_xyz,
            obtinfo=obtinfo,
            offsets=offsets,
            xyzgrid=xyzgrid,
            obtdictionary=obtdictionary,
        )

    if ifcrystal == 1:
        reci_a, reci_b, reci_c = _reciprocal_vectors(cell_a, cell_b, cell_c)
        k_vector = kpoint[0] * reci_a + kpoint[1] * reci_b + kpoint[2] * reci_c

        print("CALCULATING KPOINT:", kpoint, "ENERGYLEVEL:", energylevel + 1)

        return _accumulate_periodic(
            phi_coe_level=phi_coe_level,
            atom_xyz=atom_xyz,
            obtinfo=obtinfo,
            offsets=offsets,
            xyzgrid=xyzgrid,
            obtdictionary=obtdictionary,
            cell_a=np.asarray(cell_a, dtype=float),
            cell_b=np.asarray(cell_b, dtype=float),
            cell_c=np.asarray(cell_c, dtype=float),
            k_vector=np.asarray(k_vector, dtype=float),
            kpoint_coe=kpoint_coe,
        )

    raise ValueError(f"ifcrystal must be 0 or 1, got {ifcrystal}")
