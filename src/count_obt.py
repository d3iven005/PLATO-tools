import numpy as np
def count_obt(X): ###input is the xyzlist
####Assuming that Mg d orbital is not considered
    Elementlist = ['H','He',
                  'Li','Be','B','C','N','O','F','Ne',
                  'Na','Mg','Al','Si','P','S','Cl','Ar',
                  'K','Ca']
    orbital_list = []
    for i in range(len(X)):
        elementindex = Elementlist.index(X[i][0])
        if elementindex <= 2:
            orbital_list.append(1)
        elif 2<elementindex<=12:
            orbital_list.append(4)
        else:
            orbital_list.append(9)
    return np.array(orbital_list)
