import numpy as np


def Y(grid, atomic_position, m, l):
    """
    Real spherical-harmonic-like angular functions used in this code.

    Supported:
    - l = 0, m = 0   -> s
    - l = 1, m = 0   -> pz
    - l = 1, m = 1   -> px
    - l = 1, m = -1  -> py
    """
    vector_M = grid - atomic_position
    x = vector_M[..., 0]
    y = vector_M[..., 1]
    z = vector_M[..., 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    if l == 0:
        return np.full(r.shape, 1.0 / (2.0 * np.sqrt(np.pi)), dtype=float)

    elif l == 1:
        # avoid divide-by-zero at atomic center
        r_safe = np.where(r > 1e-14, r, 1.0)

        prefactor = 0.5 * np.sqrt(3.0 / np.pi)

        if m == 0:      # pz
            ylm = prefactor * (z / r_safe)
        elif m == 1:    # px
            ylm = prefactor * (x / r_safe)
        elif m == -1:   # py
            ylm = prefactor * (y / r_safe)
        else:
            raise ValueError(f"Unsupported m={m} for l=1")

        # define value at r=0 as 0 for p orbitals
        ylm = np.where(r > 1e-14, ylm, 0.0)
        return ylm

    else:
        raise ValueError(f"Unsupported l={l}; currently only l=0 and l=1 are implemented")
