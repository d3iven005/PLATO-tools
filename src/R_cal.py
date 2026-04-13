import numpy as np


def R(OBT, r):
    """
    Interpolate radial orbital values on distance grid r.

    Parameters
    ----------
    OBT : np.ndarray
        Orbital table of shape (N, 2), column 0 = radius, column 1 = value
    r : np.ndarray or float
        Distance array

    Returns
    -------
    np.ndarray or float
        Interpolated orbital values, with zero beyond cutoff
    """
    OBT = np.asarray(OBT, dtype=float)
    r = np.asarray(r, dtype=float)

    if OBT.ndim != 2 or OBT.shape[1] != 2:
        raise ValueError(f"OBT must have shape (N, 2), got {OBT.shape}")

    x = OBT[:, 0]
    y = OBT[:, 1]
    rmax = x[-1]

    if np.any(r < 0):
        raise ValueError("Distance r must be non-negative")

    results = np.where(r > rmax, 0.0, np.interp(r, x, y))
    return results
