import numpy as np
def distnt(x,y): ###This programme is for calculating the distance between matrix x and vector y
    r = x - y
    return np.linalg.norm(r,axis=-1)
