import numpy as np
def get_index(alist,item): #
    num = 0
    index = []
    for i in alist:
        if i == item:
            index.append(num)
        num+=1
    return index
def rdxyz(X): #X the name of the file
    if '.xyz' in X:
        position_file = open(X)
    else:
        position_file = open(X + '.xyz')
    positionflines = position_file.readlines()  
    pstionstart = get_index(positionflines,positionflines[0])
    pstionlistALL = []
    for i in range(int(positionflines[0])):
        pstionlistALL.append(positionflines[pstionstart[-1]+2+i].split())
    for i in range(len(pstionlistALL)):
        pstionlistALL[i][1] = float(pstionlistALL[i][1])
        pstionlistALL[i][2] = float(pstionlistALL[i][2])
        pstionlistALL[i][3] = float(pstionlistALL[i][3])
    cellvector = positionflines[pstionstart[-1]+1].split('Cell =')[1].split()
    cellA = [float(cellvector[0]),float(cellvector[1]),float(cellvector[2])]
    cellB = [float(cellvector[3]),float(cellvector[4]),float(cellvector[5])]
    cellC = [float(cellvector[6]),float(cellvector[7]),float(cellvector[8])]
    return np.array(pstionlistALL),np.array(cellA),np.array(cellB),np.array(cellC)
