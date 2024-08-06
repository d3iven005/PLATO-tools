from scipy.special import sph_harm
import math
import numpy as np

def Y(grid,atomic_position,m,l): 
    #vector_M = grid - atomic_position
    #M_reshape = vector_M.reshape(-1,3)
    #r = np.linalg.norm(M_reshape, axis=1)
    #theta = np.arccos(M_reshape[:,2]/r)
    #phi = np.arctan2(M_reshape[:,1],M_reshape[:,0])
    #phi = np.where(phi<0,2*math.pi+phi,phi)  ##theta 0-180, phi 0-360
    #rslt = sph_harm(m,l,phi,theta)
    #rslt = rslt.reshape(vector_M.shape[:-1])
    #return rslt
    vector_M = grid - atomic_position
    x = vector_M[...,0]
    y = vector_M[...,1]
    z  =vector_M[...,2] 
    r = np.sqrt(x**2+y**2+z**2)
    if l == 0:  ## s orbital
        Ylm = 1/(2*np.sqrt(np.pi))
    if l == 1:  ## p orbital
        if m == 0:  ## pz orbital
            Ylm = 0.5 * np.sqrt(3/np.pi) * (z/r)
        if m == 1:  ## px orbital
            Ylm = 0.5 * np.sqrt(3/np.pi) * (x/r)
        if m == -1:  ## px orbital
            Ylm = 0.5 * np.sqrt(3/np.pi) * (y/r)
    return Ylm
