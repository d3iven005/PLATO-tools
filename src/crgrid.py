import numpy as np
def crgrid(N1,N2,N3,a,b,c): ###N number of grid, a,b,c cell vector
    #r_matrix = []
    #for i in range(int(N1)):
    #    r_matrix.append([])
    #    for j in range(int(N2)):
    #        r_matrix[i].append([])
    #        for k in range(int(N3)):
    #            r_matrix[i][j].append(  i*(np.array(a)/N1)+j*np.array(b)/N2 +k*np.array(c)/N3      )
    i = np.arange(N1)
    j = np.arange(N2)
    k = np.arange(N3)
    ii, jj, kk = np.meshgrid(i, j, k, indexing='ij')
    r_matrix = ii[..., np.newaxis] * (np.array(a) / N1) +jj[..., np.newaxis] * (np.array(b) / N2) + kk[..., np.newaxis] * (np.array(c) / N3)
    return r_matrix
