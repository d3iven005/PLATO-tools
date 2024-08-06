import numpy as np
def Vcell(cell_a,cell_b,cell_c):
    volume = np.abs(np.dot(cell_a,np.cross(cell_b,cell_c)))
    return volume
