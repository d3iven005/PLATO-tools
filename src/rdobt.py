import numpy as np
def rdobt(X):   #X the name of file.
    if '.txt' in X:
        obt = open('./Odata/'+X).readlines()
    else:
        obt = open('./Odata/'+X+'.txt').readlines()
    O=[]
    for i in range(len(obt)):
        O.append([float(obt[i].split()[0]) ,  float(obt[i].split()[1])])
    return np.array(O)
###FOLLOWED MOVED TO R_cal
#def R(OBT,r):  ## r is matrix
#    rmax = OBT[-1][0]
#    x = OBT[:, 0]
#    y = OBT[:, 1]
#    results = np.where(r>rmax, 0, np.interp(r,x,y)*r)
#    return results    
