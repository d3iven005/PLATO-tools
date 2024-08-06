import os
import datetime
import numpy as np
from src.rdxyz import rdxyz
from src.rdwf import rdwf
from src.count_obt import count_obt
from src.crgrid import crgrid
from src.distnt import distnt
from src.rdobt import rdobt
from src.R_cal import R
from src.Y_cal import Y
from src.PHI_cal import PHInk_c
from src.PHI_MT import PHImt
from src.V_cell import Vcell
from src.ELF import ELF
import input
def main():
    job = input.job   ### 1 for molecular orbital calculation, 2 for electron localisation function
    jobname = input.jobname
    ncpu = input.ncpu
    ifcrystal = input.ifcrystal  ## 0 for non-periodic, 1 for crystal
    vectorflag = input.vectorflag ## 0 for reading from xyz file, 1 for setting up by your self
    vectorA = input.vectorA
    vectorB = input.vectorB
    vectorC = input.vectorC  ##unit: Bohr = 0.529177 Angstrom
    N1 = input.N1
    N2 = input.N2
    N3 = input.N3  ### number of grids for three vectors
    energylevel = input.energylevel   ## If job = 1, choose the orbital you want to calculate
    

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = datetime.datetime.now()
    print('>>>READING INPUT FILE')
    a = rdxyz('00_inputdata/'+jobname)
    if vectorflag == 0:
        vectorA = a[1]
        vectorB = a[2]
        vectorC = a[3]
    else:
        vectorA = np.array(vectorA)
        vectorB = np.array(vectorB)
        vectorC = np.array(vectorC)
    b = count_obt(a[0])
    c = rdwf('00_inputdata/'+jobname,b)
    d = crgrid(N1, N2, N3, vectorA, vectorB, vectorC)
    Hs = rdobt('H_s')
    Os = rdobt('O_s')
    Op = rdobt('O_p')
    Mgs = rdobt('Mg_s')
    Mgp = rdobt('Mg_p')
    Cs = rdobt('C_s')
    Cp = rdobt('C_p')
    AD = {'Hs':Hs,'Os':Os,'Op':Op,'Mgs':Mgs,'Mgp':Mgp,'Cs':Cs,'Cp':Cp}
    print('>>>READING INPUT FILE FINISHED')
    print()
    energylevel_list=[]
    for i in energylevel.split(','):
        if '-' in i:
            for j in range(int(i.split('-')[0]),int(i.split('-')[1])+1):
                energylevel_list.append(j-1)
        else:
            energylevel_list.append(int(i)-1)
    if job == 1:
        k = PHImt(ifcrystal, vectorA, vectorB, vectorC, AD, a[0], c[0], c[4], c[3], d, energylevel_list, ncpu)
    elif job == 2:
        m = ELF(ifcrystal, vectorA, vectorB, vectorC, AD, a[0], c[0], c[4], c[3], d, c[2][0], ncpu)
    end = datetime.datetime.now()
    print('>>>CALCULATION FINISHED')
    print('>>>TIME COST:', end-start)
    print('>>>JOB FINSIED AT:',datetime.datetime.now())
if __name__ == "__main__":
    main()
