import numpy as np

BOHR = 0.529177


def crgrid(N1, N2, N3, a, b, c, origin=None, origin_unit='bohr'):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if origin is None:
        origin_bohr = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        origin = np.asarray(origin, dtype=float)
        if origin_unit.lower() in ('angstrom', 'ang', 'a'):
            origin_bohr = origin / BOHR
        elif origin_unit.lower() == 'bohr':
            origin_bohr = origin
        else:
            raise ValueError("origin_unit must be 'bohr' or 'angstrom'")
    i = np.arange(N1)
    j = np.arange(N2)
    k = np.arange(N3)
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    r_matrix = (
        origin_bohr
        + ii[..., np.newaxis] * (a / N1)
        + jj[..., np.newaxis] * (b / N2)
        + kk[..., np.newaxis] * (c / N3)
    )
    return r_matrix
