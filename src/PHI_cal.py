import numpy as np
from src.count_obt import count_obt
from src.R_cal import R
from src.Y_cal import Y
from src.distnt import distnt
def PHInk_c(ifcrystal,cell_a,cell_b,cell_c,obtdictionary ,atom_xyz, kpoint_coe,kpoint, phi_coe, xyzgrid, energylevel):
##### phi_coe: THE COEFFICIENTS OF KPOINT CHOSEN    
    bohr = 0.529177
    if ifcrystal == 1: ##periodic
        print("CALCULATING KPOINT:",kpoint,"ENERGYLEVEL:",energylevel+1)
        PHI = 0
        obtinfo = count_obt(atom_xyz)
        bloch = np.exp(1j * np.dot(xyzgrid,kpoint))
        for i in range(len(atom_xyz)):
            o_i = sum(obtinfo[:i])
            atom_pvector = np.array(  [float(atom_xyz[i][1]), float(atom_xyz[i][2]), float(atom_xyz[i][3])     ]      )/bohr
            d_M = distnt(xyzgrid, atom_pvector)
            d_M1 = distnt(xyzgrid, atom_pvector+cell_a)
            d_M2 = distnt(xyzgrid, atom_pvector+cell_b)
            d_M3 = distnt(xyzgrid, atom_pvector+cell_c)
            d_M4 = distnt(xyzgrid, atom_pvector-cell_a)
            d_M5 = distnt(xyzgrid, atom_pvector-cell_b)
            d_M6 = distnt(xyzgrid, atom_pvector-cell_c)
            if obtinfo[i] == 1:
                phi = phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M) * Y(xyzgrid,atom_pvector,0,0)
                phi += phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M1) * Y(xyzgrid,atom_pvector+cell_a,0,0)
                phi += phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M2) * Y(xyzgrid,atom_pvector+cell_b,0,0)
                phi += phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M3) * Y(xyzgrid,atom_pvector+cell_c,0,0)
                phi += phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M4) * Y(xyzgrid,atom_pvector-cell_a,0,0)
                phi += phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M5) * Y(xyzgrid,atom_pvector-cell_b,0,0)
                phi += phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M6) * Y(xyzgrid,atom_pvector-cell_c,0,0)
                PHI+= kpoint_coe * phi * bloch
            elif obtinfo[i] == 4:
                elementtype = atom_xyz[i][0]
                phi = phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M) * Y(xyzgrid,atom_pvector,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M) * Y(xyzgrid,atom_pvector,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M) * Y(xyzgrid,atom_pvector,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M) * Y(xyzgrid,atom_pvector,-1,1)
                
                phi += phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M1) * Y(xyzgrid,atom_pvector+cell_a,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M1) * Y(xyzgrid,atom_pvector+cell_a,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M1) * Y(xyzgrid,atom_pvector+cell_a,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M1) * Y(xyzgrid,atom_pvector+cell_a,-1,1)

                phi += phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M2) * Y(xyzgrid,atom_pvector+cell_b,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M2) * Y(xyzgrid,atom_pvector+cell_b,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M2) * Y(xyzgrid,atom_pvector+cell_b,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M2) * Y(xyzgrid,atom_pvector+cell_b,-1,1)

                phi += phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M3) * Y(xyzgrid,atom_pvector+cell_c,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M3) * Y(xyzgrid,atom_pvector+cell_c,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M3) * Y(xyzgrid,atom_pvector+cell_c,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M3) * Y(xyzgrid,atom_pvector+cell_c,-1,1)

                phi += phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M4) * Y(xyzgrid,atom_pvector-cell_a,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M4) * Y(xyzgrid,atom_pvector-cell_a,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M4) * Y(xyzgrid,atom_pvector-cell_a,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M4) * Y(xyzgrid,atom_pvector-cell_a,-1,1)

                phi += phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M5) * Y(xyzgrid,atom_pvector-cell_b,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M5) * Y(xyzgrid,atom_pvector-cell_b,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M5) * Y(xyzgrid,atom_pvector-cell_b,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M5) * Y(xyzgrid,atom_pvector-cell_b,-1,1)

                phi += phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M6) * Y(xyzgrid,atom_pvector-cell_c,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M6) * Y(xyzgrid,atom_pvector-cell_c,0,1) 
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M6) * Y(xyzgrid,atom_pvector-cell_c,1,1)
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M6) * Y(xyzgrid,atom_pvector-cell_c,-1,1)
                PHI+= kpoint_coe * phi * bloch




    else:  ##NONperiodic, cell_a, cell_b, cell_c, kpoint_coe, kpoint not in calculation
        print("NOCALCULATING ENERGYLEVEL:",energylevel+1)
        obtinfo = count_obt(atom_xyz)
        PHI = 0
        for i in range(len(atom_xyz)):
            o_i = sum(obtinfo[:i])
            atom_pvector = np.array(  [float(atom_xyz[i][1]), float(atom_xyz[i][2]), float(atom_xyz[i][3])     ]      ) / bohr
            d_M = distnt(xyzgrid, atom_pvector)
            if obtinfo[i] == 1:
                phi = phi_coe[energylevel][o_i] * R(obtdictionary['Hs'],d_M) * Y(xyzgrid,atom_pvector,0,0)
                #print( phi_coe[energylevel][o_i], R(obtdictionary['Hs'],d_M), Y(xyzgrid,atom_pvector,0,0)   )
                PHI+=phi
            elif obtinfo[i] == 4:
                elementtype = atom_xyz[i][0]
                phi = phi_coe[energylevel][o_i] * R(obtdictionary[elementtype+'s'],d_M) * Y(xyzgrid,atom_pvector,0,0)
                phi += phi_coe[energylevel][o_i+1] * R(obtdictionary[elementtype+'p'],d_M) * Y(xyzgrid,atom_pvector,0,1)
                phi += phi_coe[energylevel][o_i+2] * R(obtdictionary[elementtype+'p'],d_M) * Y(xyzgrid,atom_pvector,1,1) 
                phi += phi_coe[energylevel][o_i+3] * R(obtdictionary[elementtype+'p'],d_M) * Y(xyzgrid,atom_pvector,-1,1)
                #print( phi_coe[energylevel][o_i], R(obtdictionary[elementtype+'s'],d_M), Y(xyzgrid,atom_pvector,0,0)   )
                #print( phi_coe[energylevel][o_i+1], R(obtdictionary[elementtype+'p'],d_M), Y(xyzgrid,atom_pvector,0,1)   )
                #print( phi_coe[energylevel][o_i+2], R(obtdictionary[elementtype+'p'],d_M), Y(xyzgrid,atom_pvector,1,1)   )
                #print( phi_coe[energylevel][o_i+3], R(obtdictionary[elementtype+'p'],d_M), Y(xyzgrid,atom_pvector,-1,1)   )
                PHI+=phi
            elif obtinfor[i] == 9:
                PHI=PHI
    return PHI
