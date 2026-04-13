import numpy as np

from src.count_obt import count_obt
from src.R_cal import R
from src.Y_cal import Y
from src.distnt import distnt


BOHR = 0.529177
NEIGHBOR_SHIFTS = [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]


def _reciprocal_vectors(cell_a, cell_b, cell_c):
    """
    Build reciprocal lattice vectors from real-space cell vectors.
    """
    volume_factor = np.dot(cell_a, np.cross(cell_b, cell_c))
    if abs(volume_factor) < 1e-14:
        raise ValueError("Cell volume is too small or zero; cannot build reciprocal lattice vectors.")

    reci_a = 2.0 * np.pi * np.cross(cell_b, cell_c) / volume_factor
    reci_b = 2.0 * np.pi * np.cross(cell_c, cell_a) / volume_factor
    reci_c = 2.0 * np.pi * np.cross(cell_a, cell_b) / volume_factor
    return reci_a, reci_b, reci_c


def _atom_position_bohr(atom_row):
    """
    Extract atomic position from one xyz row and convert from Angstrom to Bohr.
    """
    return np.array(
        [float(atom_row[1]), float(atom_row[2]), float(atom_row[3])],
        dtype=float
    ) / BOHR


def _orbital_offsets(obtinfo):
    """
    Starting basis-function index for each atom.
    Example: obtinfo=[1,4,4] -> offsets=[0,1,5]
    """
    return np.cumsum(np.r_[0, obtinfo[:-1]])


def _zero_field(xyzgrid):
    """
    Complex field with the spatial shape of xyzgrid.
    """
    return np.zeros(xyzgrid.shape[:3], dtype=complex)


def _build_local_basis_fields(obt_count, elementtype, atom_pos, xyzgrid, obtdictionary):
    """
    Build basis fields centered at a single atomic position.

    Returns a list of basis fields in the same ordering assumed by phi_coe:
    - obt_count == 1: [s]
    - obt_count == 4: [s, pz, px, py]
    - obt_count == 9: not implemented yet -> []
    """
    d_M = distnt(xyzgrid, atom_pos)

    if obt_count == 1:
        chi_s = R(obtdictionary['Hs'], d_M) * Y(xyzgrid, atom_pos, 0, 0)
        return [chi_s]

    if obt_count == 4:
        Rs = R(obtdictionary[elementtype + 's'], d_M)
        Rp = R(obtdictionary[elementtype + 'p'], d_M)

        chi_s = Rs * Y(xyzgrid, atom_pos, 0, 0)
        chi_pz = Rp * Y(xyzgrid, atom_pos, 0, 1)
        chi_px = Rp * Y(xyzgrid, atom_pos, 1, 1)
        chi_py = Rp * Y(xyzgrid, atom_pos, -1, 1)
        return [chi_s, chi_pz, chi_px, chi_py]

    if obt_count == 9:
        # d orbitals not implemented in current code
        return []

    raise ValueError(f"Unsupported orbital count: {obt_count}")


def _accumulate_nonperiodic(phi_coe_level, atom_xyz, obtinfo, offsets, xyzgrid, obtdictionary):
    """
    Build PHI for one energy level in the non-periodic case.
    """
    phi_total = _zero_field(xyzgrid)

    for i, atom in enumerate(atom_xyz):
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos = _atom_position_bohr(atom)

        basis_fields = _build_local_basis_fields(
            obt_count=obt_count,
            elementtype=elementtype,
            atom_pos=atom_pos,
            xyzgrid=xyzgrid,
            obtdictionary=obtdictionary,
        )

        if obt_count == 1:
            phi_total += phi_coe_level[offset] * basis_fields[0]

        elif obt_count == 4:
            phi_total += phi_coe_level[offset + 0] * basis_fields[0]
            phi_total += phi_coe_level[offset + 1] * basis_fields[1]
            phi_total += phi_coe_level[offset + 2] * basis_fields[2]
            phi_total += phi_coe_level[offset + 3] * basis_fields[3]

        elif obt_count == 9:
            # d orbitals not implemented yet
            pass

    return phi_total


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
    """
    Build PHI for one energy level in the periodic case.
    """
    phi_total = _zero_field(xyzgrid)

    # This term depends only on grid and k-vector, not on atom or image.
    bloch_grid = np.exp(1j * np.dot(xyzgrid, k_vector))

    for i, atom in enumerate(atom_xyz):
        obt_count = int(obtinfo[i])
        offset = int(offsets[i])
        elementtype = atom[0]
        atom_pos0 = _atom_position_bohr(atom)

        for ii, jj, kk in NEIGHBOR_SHIFTS:
            atom_pos = atom_pos0 + ii * cell_a + jj * cell_b + kk * cell_c

            phase_atom = np.exp(-1j * np.dot(k_vector, atom_pos))
            bloch = bloch_grid * phase_atom

            basis_fields = _build_local_basis_fields(
                obt_count=obt_count,
                elementtype=elementtype,
                atom_pos=atom_pos,
                xyzgrid=xyzgrid,
                obtdictionary=obtdictionary,
            )

            if obt_count == 1:
                phi_total += kpoint_coe * phi_coe_level[offset] * basis_fields[0] * bloch

            elif obt_count == 4:
                phi_total += kpoint_coe * phi_coe_level[offset + 0] * basis_fields[0] * bloch
                phi_total += kpoint_coe * phi_coe_level[offset + 1] * basis_fields[1] * bloch
                phi_total += kpoint_coe * phi_coe_level[offset + 2] * basis_fields[2] * bloch
                phi_total += kpoint_coe * phi_coe_level[offset + 3] * basis_fields[3] * bloch

            elif obt_count == 9:
                # d orbitals not implemented yet
                pass

    return phi_total


def PHInk_c(ifcrystal, cell_a, cell_b, cell_c, obtdictionary,
            atom_xyz, kpoint_coe, kpoint, phi_coe, xyzgrid, energylevel):
    """
    Construct real-space wavefunction PHI on xyzgrid for one energy level.

    Parameters
    ----------
    ifcrystal : int
        0 for non-periodic, 1 for periodic
    cell_a, cell_b, cell_c : array-like
        Cell vectors. Expected in Bohr in the current project convention.
    obtdictionary : dict
        Orbital tables, e.g. 'Hs', 'Cs', 'Cp', ...
    atom_xyz : array-like
        Atomic xyz-like array, with rows [element, x, y, z]
        Coordinates are assumed to be in Angstrom and are converted to Bohr here.
    kpoint_coe : float
        k-point weight
    kpoint : array-like
        Reduced k-point coordinates
    phi_coe : np.ndarray
        Orbital coefficients for one k-point, indexed as phi_coe[energylevel][basis_index]
    xyzgrid : np.ndarray
        Grid array with shape (..., 3)
    energylevel : int
        Orbital index

    Returns
    -------
    np.ndarray
        Complex PHI field on the grid
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
