import numpy as np


# Explicit mapping: element → number of orbitals
# 1 = s
# 4 = sp
# 9 = spd
ORBITAL_MAP = {
    # s only
    'H': 1, 'He': 1,

    # sp
    'Li': 4, 'Be': 4, 'B': 4, 'C': 4, 'N': 4, 'O': 4, 'F': 4, 'Ne': 4,
    'Na': 4, 'Mg': 4,

    # spd
    'Al': 9, 'Si': 9, 'P': 9, 'S': 9, 'Cl': 9, 'Ar': 9,
    'K': 9, 'Ca': 9,
}


def count_obt(X):
    """
    Determine number of orbitals per atom.

    Parameters
    ----------
    X : list-like
        xyz list: [[element, x, y, z], ...]

    Returns
    -------
    np.ndarray
        array of orbital counts per atom (1, 4, or 9)
    """
    orbital_list = []

    for i, atom in enumerate(X):
        element = atom[0]

        if element not in ORBITAL_MAP:
            raise ValueError(f"Unknown element '{element}' at atom index {i}")

        orbital_list.append(ORBITAL_MAP[element])

    return np.array(orbital_list, dtype=int)
