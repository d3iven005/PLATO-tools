import numpy as np

from src.count_obt import count_obt


BOHR = 0.529177
EPS_R = 1.0e-14
DEFAULT_TAPER_WIDTH = 0.3  # in the same unit as the radial table, usually Bohr

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
