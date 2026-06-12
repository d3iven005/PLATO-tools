import numpy as np


def _parse_int_line(line: str):
    """Try to parse all tokens in a line as ints."""
    parts = line.split()
    if not parts:
        return None
    try:
        return [int(x) for x in parts]
    except ValueError:
        return None


def _parse_float_line(line: str):
    """Try to parse all tokens in a line as floats."""
    parts = line.split()
    if not parts:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def _parse_plato_numeric_header(lines, expected_n_basis):
    """
    Parse the numeric header used by newer PLATO .wf files.

    Observed layout:
        line 1: nk n_orbitals n_basis ... ...
        line 2: n_atoms
        line 3: basis offsets, length n_atoms + 1
        line 4: basis angular/channel labels, length n_basis

    Returns
    -------
    header : dict | None
        Parsed metadata, or None if the file does not start with this header.
    data_start : int
        First line after the header. Zero if no header is detected.
    """
    if len(lines) < 4:
        return None, 0

    first = _parse_int_line(lines[0])
    natom_line = _parse_int_line(lines[1])

    if first is None or natom_line is None:
        return None, 0
    if len(first) < 3 or len(natom_line) != 1:
        return None, 0

    n_kpoints_declared = first[0]
    n_orbitals = first[1]
    n_basis = first[2]
    n_atoms = natom_line[0]

    offsets = _parse_int_line(lines[2])
    basis_labels = _parse_int_line(lines[3])

    if offsets is None or basis_labels is None:
        return None, 0
    if n_atoms < 0 or n_basis <= 0 or n_orbitals <= 0:
        return None, 0
    if len(offsets) != n_atoms + 1:
        return None, 0
    if offsets[0] != 0 or offsets[-1] != n_basis:
        return None, 0
    if len(basis_labels) != n_basis:
        return None, 0

    if expected_n_basis is not None and n_basis != expected_n_basis:
        raise ValueError(
            "Basis count mismatch between xyz-derived orbital count and .wf header: "
            f"xyz/count_obt gives {expected_n_basis}, .wf header gives {n_basis}."
        )

    header = {
        "n_kpoints_declared": n_kpoints_declared,
        "n_orbitals": n_orbitals,
        "n_basis": n_basis,
        "n_atoms": n_atoms,
        "offsets": offsets,
        "basis_labels": basis_labels,
        "raw_first_line": first,
    }
    return header, 4


def _is_orbital_header(lines, idx, n_basis):
    """
    Check whether lines[idx] looks like an orbital header:
    - current line has exactly 2 floats: [energy, occupation]
    - followed by at least n_basis coefficient lines
    - each coefficient line has 1 or 2 floats
    """
    vals = _parse_float_line(lines[idx])
    if vals is None or len(vals) != 2:
        return False

    if idx + n_basis >= len(lines):
        return False

    for j in range(1, n_basis + 1):
        coeff_vals = _parse_float_line(lines[idx + j])
        if coeff_vals is None or len(coeff_vals) not in (1, 2):
            return False

    return True


def _find_first_orbital_header(lines, start_idx, n_basis, stop_idx=None):
    """
    Find the first orbital-header line starting from start_idx.
    Search until stop_idx (exclusive) or end of file.
    """
    if stop_idx is None:
        stop_idx = len(lines)

    upper = min(stop_idx, len(lines))
    for idx in range(start_idx, upper):
        if _is_orbital_header(lines, idx, n_basis):
            return idx
    return None


def _parse_one_k_block(lines, start_idx, n_basis, max_orbitals):
    """
    Parse one k-point/Gamma block starting from the first orbital header.

    Returns
    -------
    energies, occs, coeffs, next_idx
    """
    energies = []
    occs = []
    coeffs = []

    idx = start_idx
    n_read = 0

    while idx < len(lines) and n_read < max_orbitals:
        if not _is_orbital_header(lines, idx, n_basis):
            break

        header_vals = _parse_float_line(lines[idx])
        energy, occ = header_vals[0], header_vals[1]

        orb_coeffs = []
        for j in range(1, n_basis + 1):
            vals = _parse_float_line(lines[idx + j])
            if len(vals) == 1:
                orb_coeffs.append(vals[0])
            elif len(vals) == 2:
                orb_coeffs.append(vals[0] + 1j * vals[1])
            else:
                raise ValueError(
                    f"Invalid coefficient line at line {idx + j + 1}: {lines[idx + j]!r}"
                )

        energies.append(energy)
        occs.append(occ)
        coeffs.append(orb_coeffs)

        idx += n_basis + 1
        n_read += 1

    return energies, occs, coeffs, idx


def _parse_kpoint_line(line):
    parts = line.split()
    # Expected: K-point 1  -0.25000  -0.25000   0.00000 0.5000000000
    if len(parts) < 6:
        raise ValueError(f"Malformed K-point line: {line!r}")

    try:
        kx = float(parts[2])
        ky = float(parts[3])
        kz = float(parts[4])
        weight = float(parts[5])
    except ValueError as exc:
        raise ValueError(f"Failed to parse K-point line: {line!r}") from exc

    return [kx, ky, kz], weight


def _validate_equal_length_blocks(E_list, Occ_list, phi_coe_list, n_orbitals):
    for iblk, (energies, occs, coeffs) in enumerate(
        zip(E_list, Occ_list, phi_coe_list), start=1
    ):
        if len(energies) != n_orbitals:
            raise ValueError(
                f"K/Gamma block {iblk} contains {len(energies)} orbitals; "
                f"expected {n_orbitals}."
            )
        if len(occs) != len(energies) or len(coeffs) != len(energies):
            raise ValueError(f"Incomplete orbital data in K/Gamma block {iblk}.")


def rdwf(X, y):
    """
    Read a PLATO .wf file.

    Supported layouts:
    - legacy files that start directly with ``K-point ...`` blocks
    - newer files with a four-line numeric PLATO header before the data
    - non-periodic/Gamma-only files without an explicit ``K-point`` line

    Parameters
    ----------
    X : str
        wf filename or filename stem
    y : list-like
        orbital-number list; sum(y) is used as basis/orbital count

    Returns
    -------
    K_coe : np.ndarray
        k-point weights, shape (nk,)
    E_list : np.ndarray
        eigenvalues, shape (nk, n_orb)
    Occ_list : np.ndarray
        occupations, shape (nk, n_orb)
    phi_coe_list : np.ndarray
        orbital coefficients, shape (nk, n_orb, n_basis)
    K_point : np.ndarray
        k-points, shape (nk, 3)
    """
    filename = X if X.endswith(".wf") else X + ".wf"

    with open(filename, "r") as wf_file:
        lines = [line.strip() for line in wf_file if line.strip()]

    n_basis = int(sum(y))
    header, data_start = _parse_plato_numeric_header(lines, n_basis)
    n_orbitals = header["n_orbitals"] if header is not None else n_basis

    kpoint_indices = [
        i
        for i, line in enumerate(lines[data_start:], start=data_start)
        if line.startswith("K-point")
    ]

    K_coe = []
    K_point = []
    E_list = []
    Occ_list = []
    phi_coe_list = []

    if kpoint_indices:
        kpoint_indices.append(len(lines))

        for iblk in range(len(kpoint_indices) - 1):
            kp_line_idx = kpoint_indices[iblk]
            block_end = kpoint_indices[iblk + 1]

            kpoint, weight = _parse_kpoint_line(lines[kp_line_idx])

            start_idx = _find_first_orbital_header(
                lines, kp_line_idx + 1, n_basis, stop_idx=block_end
            )
            if start_idx is None:
                raise ValueError(
                    "Could not find orbital data after K-point block "
                    f"starting at line {kp_line_idx + 1}"
                )

            energies, occs, coeffs, _ = _parse_one_k_block(
                lines, start_idx, n_basis, max_orbitals=n_orbitals
            )

            K_coe.append(weight)
            K_point.append(kpoint)
            E_list.append(energies)
            Occ_list.append(occs)
            phi_coe_list.append(coeffs)

    else:
        start_idx = _find_first_orbital_header(lines, data_start, n_basis)
        if start_idx is None:
            raise ValueError("Could not locate orbital data block in non-periodic .wf file.")

        energies, occs, coeffs, _ = _parse_one_k_block(
            lines, start_idx, n_basis, max_orbitals=n_orbitals
        )

        # For non-periodic case, treat as Gamma-only single block.
        K_coe.append(1.0)
        K_point.append([0.0, 0.0, 0.0])
        E_list.append(energies)
        Occ_list.append(occs)
        phi_coe_list.append(coeffs)

    _validate_equal_length_blocks(E_list, Occ_list, phi_coe_list, n_orbitals)

    if header is not None and header["n_kpoints_declared"] not in (0, len(K_coe)):
        raise ValueError(
            "K-point count mismatch in .wf file: "
            f"header declares {header['n_kpoints_declared']}, parsed {len(K_coe)}."
        )

    K_coe = np.array(K_coe, dtype=float)
    E_list = np.array(E_list, dtype=float)
    Occ_list = np.array(Occ_list, dtype=float)
    phi_coe_list = np.array(phi_coe_list, dtype=complex)
    K_point = np.array(K_point, dtype=float)

    return K_coe, E_list, Occ_list, phi_coe_list, K_point
