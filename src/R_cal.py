import numpy as np
def R(OBT,r):  ## r is a matrix including the distance between the points and atom position (should be calculated by distnt(x,atomp)).
    rmax = OBT[-1][0]
    x = OBT[:, 0]
    y = OBT[:, 1]
    results = np.where(r>=rmax, 0, np.interp(r,x,y))
    return results    
