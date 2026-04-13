import numpy as np


def _parse_float_line(line: str):
    """Try to parse all tokens in a line as floats."""
    parts = line.split()
    if not parts:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


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


def _parse_one_k_block(lines, start_idx, n_basis, max_orbitals=None):
    """
    Parse one k-point block starting from the first orbital header.
    Returns:
        energies, occs, coeffs, next_idx
    """
    if max_orbitals is None:
        max_orbitals = n_basis

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


def rdwf(X, y):
    """
    Read PLATO .wf file (new format, compatible with both periodic and non-periodic examples).

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
        # keep non-empty lines only
        lines = [line.strip() for line in wf_file if line.strip()]

    n_basis = int(sum(y))
    I = 1j

    # Find all K-point markers
    kpoint_indices = [i for i, line in enumerate(lines) if line.startswith("K-point")]

    K_coe = []
    K_point = []
    E_list = []
    Occ_list = []
    phi_coe_list = []

    # ------------------------------------------------------------
    # Case 1: periodic file with explicit K-point blocks
    # ------------------------------------------------------------
    if kpoint_indices:
        kpoint_indices.append(len(lines))

        for iblk in range(len(kpoint_indices) - 1):
            kp_line_idx = kpoint_indices[iblk]
            block_end = kpoint_indices[iblk + 1]

            parts = lines[kp_line_idx].split()
            # Expected: K-point 1  -0.25000  -0.25000   0.00000 0.5000000000
            if len(parts) < 6:
                raise ValueError(f"Malformed K-point line: {lines[kp_line_idx]!r}")

            try:
                kx = float(parts[2])
                ky = float(parts[3])
                kz = float(parts[4])
                weight = float(parts[5])
            except ValueError as exc:
                raise ValueError(f"Failed to parse K-point line: {lines[kp_line_idx]!r}") from exc

            start_idx = _find_first_orbital_header(lines, kp_line_idx + 1, n_basis, stop_idx=block_end)
            if start_idx is None:
                raise ValueError(
                    f"Could not find orbital data after K-point block starting at line {kp_line_idx + 1}"
                )

            energies, occs, coeffs, _ = _parse_one_k_block(
                lines, start_idx, n_basis, max_orbitals=n_basis
            )

            K_coe.append(weight)
            K_point.append([kx, ky, kz])
            E_list.append(energies)
            Occ_list.append(occs)
            phi_coe_list.append(coeffs)

    # ------------------------------------------------------------
    # Case 2: non-periodic file without explicit K-point line
    # ------------------------------------------------------------
    else:
        start_idx = _find_first_orbital_header(lines, 0, n_basis)
        if start_idx is None:
            raise ValueError("Could not locate orbital data block in non-periodic .wf file.")

        energies, occs, coeffs, _ = _parse_one_k_block(
            lines, start_idx, n_basis, max_orbitals=n_basis
        )

        # For non-periodic case, treat as Gamma-only single block
        K_coe.append(1.0)
        K_point.append([0.0, 0.0, 0.0])
        E_list.append(energies)
        Occ_list.append(occs)
        phi_coe_list.append(coeffs)

    # Convert to arrays
    # use complex dtype for coefficients so both real/complex cases work
    K_coe = np.array(K_coe, dtype=float)
    E_list = np.array(E_list, dtype=float)
    Occ_list = np.array(Occ_list, dtype=float)
    phi_coe_list = np.array(phi_coe_list, dtype=complex)
    K_point = np.array(K_point, dtype=float)

    return K_coe, E_list, Occ_list, phi_coe_list, K_point
