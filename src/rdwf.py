import numpy as np
def rdwf(X,y):  #X the name of the wf file, y the orbital number list
    if '.wf' in X:
        wf_file = open(X)
    else:
        wf_file = open(X+'.wf')
    wflines = wf_file.readlines()
    index = []
    for i in range(len(wflines)):
        if 'K-point' in wflines[i]:
            index.append(i)
    index.append(len(wflines))
    K_coe = []
    K_point = []
    phi_coe_list = []
    E_list = []
    Occ_list=[]
    I = 1j
    for i in range(len(index)-1):
        K_coe.append(float(wflines[index[i]].split()[-1]))
        K_point.append([float(wflines[index[i]].split()[2]),float(wflines[index[i]].split()[3]),float(wflines[index[i]].split()[4])])
        phi_coe_list.append([])
        E_list.append([])
        Occ_list.append([])
    for i in range(len(K_coe)):
        for j in range(sum(y)):
            E_list[i].append(   float(wflines[index[i]+1+j*(sum(y)+1)].split()[0])     )
            Occ_list[i].append(float(wflines[index[i]+1+j*(sum(y)+1)].split()[1]))
            phi_coe_list[i].append([])
            if len(wflines[index[i]+1+j*(sum(y)+1)+1].split())==1:
                for k in range(sum(y)):
                    phi_coe_list[i][j].append(  float(wflines[index[i]+1+j*(sum(y)+1)+1+k])    )
            else:
                for k in range(sum(y)):
                    phi_coe_list[i][j].append(  float(wflines[index[i]+1+j*(sum(y)+1)+1+k].split()[0]) + I * float(wflines[index[i]+1+j*(sum(y)+1)+1+k].split()[1])    )
    return np.array(K_coe), np.array(E_list), np.array(Occ_list), np.array(phi_coe_list), np.array(K_point)
