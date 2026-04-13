import numpy as np


def rdxyz(X):
    """Read one xyz structure.

    Parameters
    ----------
    X : str
        xyz filename or filename stem

    Returns
    -------
    position_array : np.ndarray
        shape (n_atoms, 4), columns: [element, x, y, z]
    cellA, cellB, cellC : np.ndarray
        cell vectors; if no 'Cell =' is found, return zero vectors
    """
    filename = X if X.endswith(".xyz") else X + ".xyz"

    with open(filename, "r") as position_file:
        positionflines = position_file.readlines()

    if len(positionflines) < 2:
        raise ValueError(f"Invalid xyz file: {filename}")

    try:
        n_atoms = int(positionflines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line of xyz must be atom count: {positionflines[0]!r}") from exc

    # second line may be blank, comment, or contain 'Cell ='
    comment_line = positionflines[1].strip()

    pstionlistALL = []
    atom_start = 2

    if len(positionflines) < atom_start + n_atoms:
        raise ValueError(
            f"xyz file {filename} does not contain enough atom lines: "
            f"expected {n_atoms}, found {len(positionflines) - atom_start}"
        )

    for i in range(n_atoms):
        parts = positionflines[atom_start + i].split()
        if len(parts) < 4:
            raise ValueError(
                f"Invalid atom line {atom_start + i + 1} in {filename}: "
                f"{positionflines[atom_start + i]!r}"
            )
        pstionlistALL.append([parts[0], float(parts[1]), float(parts[2]), float(parts[3])])

    # Default: no cell information
    cellA = [0.0, 0.0, 0.0]
    cellB = [0.0, 0.0, 0.0]
    cellC = [0.0, 0.0, 0.0]

    if "Cell =" in comment_line:
        cellvector = comment_line.split("Cell =")[1].split()
        if len(cellvector) < 9:
            raise ValueError(f"Cell information is incomplete in {filename}: {comment_line!r}")
        cellA = [float(cellvector[0]), float(cellvector[1]), float(cellvector[2])]
        cellB = [float(cellvector[3]), float(cellvector[4]), float(cellvector[5])]
        cellC = [float(cellvector[6]), float(cellvector[7]), float(cellvector[8])]

    return np.array(pstionlistALL, dtype=object), np.array(cellA), np.array(cellB), np.array(cellC)
