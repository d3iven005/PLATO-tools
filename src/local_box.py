import numpy as np


BOHR = 0.529177


def build_local_box(xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Build an orthogonal local output box from Angstrom bounds.
    """
    bounds = np.array([xmin, xmax, ymin, ymax, zmin, zmax], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Local box bounds must be finite numbers.")

    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise ValueError(
            "Local box requires xmax > xmin, ymax > ymin, and zmax > zmin."
        )

    origin_ang = np.array([xmin, ymin, zmin], dtype=float)
    cell_a = np.array([(xmax - xmin) / BOHR, 0.0, 0.0], dtype=float)
    cell_b = np.array([0.0, (ymax - ymin) / BOHR, 0.0], dtype=float)
    cell_c = np.array([0.0, 0.0, (zmax - zmin) / BOHR], dtype=float)

    return origin_ang, cell_a, cell_b, cell_c
