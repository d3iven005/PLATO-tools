from src.PHI_cal import PHInk_c
from concurrent.futures import ThreadPoolExecutor
from src.V_cell import Vcell
import numpy as np
def ELF(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list, kpoint_list, phi_coe_list, xyzgrid, energylevel_occ,ncpu):
    print('>>>ELECTRON LOCALISATION FUNCTION CALCULATION STARTING')
    bohr = 0.529177
    V = Vcell(cell_a,cell_b,cell_c)
    N = int(xyzgrid.size/3)
    dV = V/N
    Natom = len(atom_xyz)
    Nx,Ny,Nz,useless = xyzgrid.shape 
    xyz = atom_xyz
    assert len(kpoint_coe_list)==len(kpoint_list)==len(phi_coe_list)
    energylevel_list = []
    for i in range(len(energylevel_occ)):
        if energylevel_occ[i] == 1:
            energylevel_list.append(i)
        else:
            break
    A = energylevel_list
    L = len(energylevel_list)
    L2 = len(kpoint_coe_list)
    kpoint_coe_list = np.repeat(kpoint_coe_list,L)
    kpoint_list = np.tile(kpoint_list,(L,1))
    phi_coe_list = np.concatenate([phi_coe_list]*L)
    energylevel_list = np.repeat(np.array(energylevel_list),L2) 
    ifcrystal = np.array([ifcrystal]*L*L2)
    cell_a = np.array([cell_a]*L*L2)
    cell_b = np.array([cell_b]*L*L2)
    cell_c = np.array([cell_c]*L*L2)
    obtdictionary = np.array([obtdictionary]*L*L2)
    atom_xyz = np.array([atom_xyz]*L*L2) 
    xyzgrid = np.array([xyzgrid]*L*L2)
    xyzgrid_x = xyzgrid + np.array([0.001,0,0])
    xyzgrid_y = xyzgrid + np.array([0,0.001,0])
    xyzgrid_z = xyzgrid + np.array([0,0,0.001])
    with ThreadPoolExecutor(max_workers = ncpu) as executor:
        rslt_o = list(executor.map(lambda p: PHInk_c(*p),zip(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list,kpoint_list, phi_coe_list, xyzgrid, energylevel_list)))
        rslt_x = list(executor.map(lambda p: PHInk_c(*p),zip(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list,kpoint_list, phi_coe_list, xyzgrid_x, energylevel_list)))
        rslt_y = list(executor.map(lambda p: PHInk_c(*p),zip(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list,kpoint_list, phi_coe_list, xyzgrid_y, energylevel_list)))
        rslt_z = list(executor.map(lambda p: PHInk_c(*p),zip(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list,kpoint_list, phi_coe_list, xyzgrid_z, energylevel_list))) 
    Elementlist = ['H','He',
                  'Li','Be','B','C','N','O','F','Ne',
                  'Na','Mg','Al','Si','P','S','Cl','Ar',
                  'K','Ca']
    chargeDo = 0
    chargeDx = 0
    chargeDy = 0
    chargeDz = 0
    KE       = 0
    for i in A:
        PHIO = 0
        PHIX= 0
        PHIY= 0
        PHIZ= 0
        for j in range(len(energylevel_list)):
            if energylevel_list[j]==i:
                PHIO+=rslt_o[j]
                PHIX+=rslt_x[j]
                PHIY+=rslt_y[j]
                PHIZ+=rslt_z[j]
            else:
                continue
        normcoe = np.sqrt(1/np.sum(PHIO*np.conj(PHIO)*dV))
        PHIO = PHIO * normcoe
        PHIX = PHIX * normcoe
        PHIY = PHIY * normcoe
        PHIZ = PHIZ * normcoe
        chargeDo += np.conj(PHIO) * PHIO
        chargeDx += np.conj(PHIX) * PHIX
        chargeDy += np.conj(PHIY) * PHIY
        chargeDz += np.conj(PHIZ) * PHIZ
        KE += np.conj((PHIX - PHIO)/0.001)*(PHIX - PHIO)/0.001 + np.conj((PHIY - PHIO)/0.001)*(PHIY - PHIO)/0.001 +np.conj((PHIZ - PHIO)/0.001)*(PHIZ - PHIO)/0.001
    WKED = 0.25 * 0.99 * (    ((chargeDx-chargeDo)/0.001)**2   +  ((chargeDy-chargeDo)/0.001)**2  +  ((chargeDz-chargeDo)/0.001)**2            )
    UEG = 9.115599744691192 * (chargeDo**(5/3))
    chi = (KE - WKED/chargeDo) / UEG
    ELF = 1/(1+chi**2)
    ELF = np.nan_to_num(ELF)
    #filename = 'ELF'
    #np.savez('./01_results/'+filename+'.npz', CELLA = cell_a[0],CELLB = cell_b[0], CELLC = cell_c[0], AXYZ = atom_xyz[0],RXYZ = xyzgrid[0],PHI = PHI)
    with open('./01_results/'+'CD'+'.cube','w') as f:
        f.write('cube file for electron localisation function'+'\n')
        f.write(''+'\n')
        f.write(str(Natom)+' 0.0 0.0 0.0'+'\n')
        f.write(str(Nx)+' '+str(cell_a[0][0]/Nx)+' '+str(cell_a[0][1]/Nx)+' '+str(cell_a[0][2]/Nx)+'\n')
        f.write(str(Ny)+' '+str(cell_b[0][0]/Ny)+' '+str(cell_b[0][1]/Ny)+' '+str(cell_b[0][2]/Ny)+'\n')
        f.write(str(Nz)+' '+str(cell_c[0][0]/Nz)+' '+str(cell_c[0][1]/Nz)+' '+str(cell_c[0][2]/Nz)+'\n')
        for ii in range(len(xyz)):
            f.write( str(Elementlist.index(xyz[ii][0])+1) +' 0.0 '+str(float(xyz[ii][1])/bohr)+' '+str(float(xyz[ii][2])/bohr)+' '+str(float(xyz[ii][3])/bohr) +'\n'  
    )
        for ii in range(len(chargeDo)):
            for jj in range(len(chargeDo[ii])):
                for kk in range(len(chargeDo[ii][jj])):
                    f.write(str(chargeDo[ii][jj][kk].real)+' ')
                    if kk%6 == 5:
                        f.write('\n')
                f.write('\n')
    with open('./01_results/'+'ELF'+'.cube','w') as f:
         f.write('cube file for electron localisation function'+'\n')
         f.write(''+'\n')
         f.write(str(Natom)+' 0.0 0.0 0.0'+'\n')
         f.write(str(Nx)+' '+str(cell_a[0][0]/Nx)+' '+str(cell_a[0][1]/Nx)+' '+str(cell_a[0][2]/Nx)+'\n')
         f.write(str(Ny)+' '+str(cell_b[0][0]/Ny)+' '+str(cell_b[0][1]/Ny)+' '+str(cell_b[0][2]/Ny)+'\n')
         f.write(str(Nz)+' '+str(cell_c[0][0]/Nz)+' '+str(cell_c[0][1]/Nz)+' '+str(cell_c[0][2]/Nz)+'\n')
         for ii in range(len(xyz)):
             f.write( str(Elementlist.index(xyz[ii][0])+1) +' 0.0 '+str(float(xyz[ii][1])/bohr)+' '+str(float(xyz[ii][2])/bohr)+' '+str(float(xyz[ii][3])/bohr) +'\n'     )
         for ii in range(len(ELF)):
             for jj in range(len(ELF[ii])):
                 for kk in range(len(ELF[ii][jj])):
                     f.write(str(ELF[ii][jj][kk].real)+' ')
                     if kk%6 == 5:
                         f.write('\n')
                 f.write('\n')
    return ELF
