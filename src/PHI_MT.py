from src.PHI_cal import PHInk_c
from concurrent.futures import ThreadPoolExecutor
from src.V_cell import Vcell
import numpy as np
def PHImt(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list, kpoint_list, phi_coe_list, xyzgrid, energylevel_list,ncpu):
    bohr = 0.529177
    V = Vcell(cell_a,cell_b,cell_c)
    N = int(xyzgrid.size/3)
    dV = V/N
    A = energylevel_list
    Natom = len(atom_xyz)
    Nx,Ny,Nz,useless = xyzgrid.shape 
    xyz = atom_xyz
    assert len(kpoint_coe_list)==len(kpoint_list)==len(phi_coe_list)
    L = len(energylevel_list)
    L2 = len(kpoint_coe_list)
    #print(len(phi_coe_list),len(phi_coe_list[0])) 
    kpoint_coe_list = np.repeat(kpoint_coe_list,L)
    kpoint_list = np.tile(kpoint_list,(L,1))
    phi_coe_list = np.concatenate([phi_coe_list]*L)
    energylevel_list = np.repeat(energylevel_list,L2) 
    ifcrystal = np.array([ifcrystal]*L*L2)
    cell_a = np.array([cell_a]*L*L2)
    cell_b = np.array([cell_b]*L*L2)
    cell_c = np.array([cell_c]*L*L2)
    obtdictionary = np.array([obtdictionary]*L*L2)
    atom_xyz = np.array([atom_xyz]*L*L2) 
    xyzgrid = np.array([xyzgrid]*L*L2)
    #print('KPOINT:',kpoint_list)
    #print('PHI_COE_LIST',phi_coe_list)
    #print('ELEVEL:',energylevel_list)
    #print(len(kpoint_list),len(phi_coe_list),len(energylevel_list),len(kpoint_coe_list))
    with ThreadPoolExecutor(max_workers = ncpu) as executor:
        results = list(executor.map(lambda p: PHInk_c(*p),zip(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe_list,kpoint_list, phi_coe_list, xyzgrid, energylevel_list))) 
    #filename = './PHI'+'_'+str(energylevel_list[0])+'-'+str(energylevel_list[-1])+'.npz'
    #np.savez(filename,Kpoint = kpoint_list, Kcoe = kpoint_coe_list, Phicoe = phi_coe_list, CELLA = cell_a[0], CELLB = cell_b[0], CELLC = cell_c[0], AXYZ = atom_xyz[0],RXYZ = xyzgrid[0], NE = energylevel_list, PHI = results)
    #print('ELEVEL:',len(energylevel_list))
    Elementlist = ['H','He',
                  'Li','Be','B','C','N','O','F','Ne',
                  'Na','Mg','Al','Si','P','S','Cl','Ar',
                  'K','Ca']
    for i in A:
        PHI = 0
        for j in range(len(energylevel_list)):
            if energylevel_list[j]==i:
                PHI+=results[j]
            else:
                continue
        normcoe = np.sqrt(1/np.sum(PHI*np.conj(PHI)*dV))
        #print(normcoe)
        PHI = PHI * normcoe        
        filename = 'PHI_'+str(i+1)
        np.savez('./01_results/'+filename+'.npz', CELLA = cell_a[0],CELLB = cell_b[0], CELLC = cell_c[0], AXYZ = atom_xyz[0],RXYZ = xyzgrid[0],PHI = PHI)
        with open('./01_results/'+filename+'_REAL.cube','w') as f:
            f.write('cube file for molecular orbital'+'\n')
            f.write(''+'\n')
            f.write(str(Natom)+' 0.0 0.0 0.0'+'\n')
            f.write(str(Nx)+' '+str(cell_a[0][0]/Nx)+' '+str(cell_a[0][1]/Nx)+' '+str(cell_a[0][2]/Nx)+'\n')
            f.write(str(Ny)+' '+str(cell_b[0][0]/Ny)+' '+str(cell_b[0][1]/Ny)+' '+str(cell_b[0][2]/Ny)+'\n')
            f.write(str(Nz)+' '+str(cell_c[0][0]/Nz)+' '+str(cell_c[0][1]/Nz)+' '+str(cell_c[0][2]/Nz)+'\n')
            for ii in range(len(xyz)):
                f.write( str(Elementlist.index(xyz[ii][0])+1) +' 0.0 '+str(float(xyz[ii][1])/bohr)+' '+str(float(xyz[ii][2])/bohr)+' '+str(float(xyz[ii][3])/bohr) +'\n'     )
            for ii in range(len(PHI)):
                for jj in range(len(PHI[ii])):
                    for kk in range(len(PHI[ii][jj])):
                        f.write(str(PHI[ii][jj][kk].real)+' ')
                        if kk%6 == 5:
                            f.write('\n')
                    f.write('\n')
            
    return results
