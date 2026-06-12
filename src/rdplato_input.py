import numpy as np


def _find_keyword(lines, keyword):
    keyword_lower = keyword.lower()
    for idx, line in enumerate(lines):
        if line.strip().lower() == keyword_lower:
            return idx
    return None


def _read_vector_line(line, expected_len, keyword):
    parts = line.split()
    if len(parts) < expected_len:
        raise ValueError(f"{keyword} expects {expected_len} values, got: {line!r}")
    try:
        return [float(x) for x in parts[:expected_len]]
    except ValueError as exc:
        raise ValueError(f"Failed to parse {keyword} values from line: {line!r}") from exc


def _read_int_after_keyword(lines, keyword):
    idx = _find_keyword(lines, keyword)
    if idx is None:
        return None
    if idx + 1 >= len(lines):
        raise ValueError(f"Missing value after {keyword}.")
    try:
        return int(lines[idx + 1].split()[0])
    except ValueError as exc:
        raise ValueError(f"Failed to parse integer after {keyword}: {lines[idx + 1]!r}") from exc


def _read_cell(lines):
    cellvec_idx = _find_keyword(lines, "CellVec")
    cellsize_idx = _find_keyword(lines, "CellSize")

    if cellvec_idx is None or cellsize_idx is None:
        return None
    if cellvec_idx + 3 >= len(lines):
        raise ValueError("CellVec must be followed by three vector lines.")
    if cellsize_idx + 1 >= len(lines):
        raise ValueError("CellSize must be followed by one size line.")

    cellvec = np.array(
        [
            _read_vector_line(lines[cellvec_idx + 1], 3, "CellVec"),
            _read_vector_line(lines[cellvec_idx + 2], 3, "CellVec"),
            _read_vector_line(lines[cellvec_idx + 3], 3, "CellVec"),
        ],
        dtype=float,
    )
    cellsize = np.array(_read_vector_line(lines[cellsize_idx + 1], 3, "CellSize"), dtype=float)

    # PLATO input CellSize is in Bohr. Scale each lattice vector by its size.
    cell_a = cellvec[0] * cellsize[0]
    cell_b = cellvec[1] * cellsize[1]
    cell_c = cellvec[2] * cellsize[2]
    return cell_a, cell_b, cell_c


def _read_atoms_format3(lines, n_atoms):
    atoms_idx = _find_keyword(lines, "Atoms")
    if atoms_idx is None:
        return None
    if n_atoms is None:
        raise ValueError("NAtom is required before reading Format 3 Atoms.")
    if atoms_idx + n_atoms >= len(lines):
        raise ValueError(
            f"Atoms section does not contain enough lines: expected {n_atoms} atoms."
        )

    atoms = []
    for i in range(n_atoms):
        line = lines[atoms_idx + 1 + i]
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid Format 3 atom line: {line!r}")
        try:
            atoms.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError as exc:
            raise ValueError(f"Failed to parse Format 3 atom line: {line!r}") from exc

    return np.array(atoms, dtype=object)


def rdplato_input(filename):
    """
    Read the subset of PLATO .in files needed by this project.

    Currently supported:
    - Format 3 only
    - CellVec and CellSize, with CellSize in Bohr
    - Atoms coordinates in Angstrom
    """
    with open(filename, "r") as input_file:
        lines = [line.strip() for line in input_file if line.strip()]

    fmt = _read_int_after_keyword(lines, "Format")
    if fmt is not None and fmt != 3:
        raise NotImplementedError(
            f"PLATO input Format {fmt} is not supported yet; only Format 3 is implemented."
        )

    n_atoms = _read_int_after_keyword(lines, "NAtom")
    atom_xyz = _read_atoms_format3(lines, n_atoms)
    cell = _read_cell(lines)

    if atom_xyz is None:
        raise ValueError(f"Could not find a Format 3 Atoms section in {filename}.")

    if cell is None:
        cell = (
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
        )

    return atom_xyz, cell[0], cell[1], cell[2]
