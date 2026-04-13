import os
import datetime
import numpy as np

from src.rdxyz import rdxyz
from src.rdwf import rdwf
from src.count_obt import count_obt
from src.crgrid import crgrid
from src.rdobt import rdobt
from src.PHI_MT import PHImt
from src.ELF import ELF
from src.box_for_molecule import build_molecule_box

import input


def main():
    job = input.job   # 1 for molecular orbital calculation, 2 for electron localisation function
    jobname = input.jobname
    ncpu = input.ncpu
    ifcrystal = input.ifcrystal   # 0 for non-periodic, 1 for crystal
    vectorflag = input.vectorflag # 0 for reading from xyz file, 1 for setting up by yourself
    vectorA = input.vectorA
    vectorB = input.vectorB
    vectorC = input.vectorC       # unit: Bohr
    N1 = input.N1
    N2 = input.N2
    N3 = input.N3
    energylevel = input.energylevel
    padding_distance = input.padding_distance

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = datetime.datetime.now()

    print('>>>READING INPUT FILE')

    a = rdxyz('00_inputdata/' + jobname)
    atom_xyz = a[0]

    # --------------------------------------------------
    # Build box and origin
    # --------------------------------------------------
    origin_ang = None

    if ifcrystal == 0:
        # molecule / non-periodic case:
        # always build a box from atomic coordinates with ±6 Å padding
        origin_ang, vectorA, vectorB, vectorC = build_molecule_box(
            atom_xyz,
            padding_angstrom=padding_distance
        )
        print('>>>NON-PERIODIC MODE: AUTOMATIC MOLECULE BOX ENABLED')
        #print('>>>BOX ORIGIN (Angstrom):', origin_ang)
        #print('>>>BOX VECTORS (Bohr):')
        #print('   A =', vectorA)
        #print('   B =', vectorB)
        #print('   C =', vectorC)

    else:
        # periodic case
        if vectorflag == 0:
            vectorA = np.array(a[1], dtype=float)
            vectorB = np.array(a[2], dtype=float)
            vectorC = np.array(a[3], dtype=float)
        else:
            vectorA = np.array(vectorA, dtype=float)
            vectorB = np.array(vectorB, dtype=float)
            vectorC = np.array(vectorC, dtype=float)

    # --------------------------------------------------
    # Read orbital / wavefunction data
    # --------------------------------------------------
    b = count_obt(atom_xyz)
    c = rdwf('00_inputdata/' + jobname, b)

    # --------------------------------------------------
    # Create grid
    # --------------------------------------------------
    if ifcrystal == 0:
        d = crgrid(N1, N2, N3, vectorA, vectorB, vectorC, origin=origin_ang, origin_unit='angstrom')
    else:
        d = crgrid(N1, N2, N3, vectorA, vectorB, vectorC, origin=origin_ang, origin_unit='angstrom')

    # --------------------------------------------------
    # Read numerical orbitals
    # --------------------------------------------------
    Hs = rdobt('H_s')
    Os = rdobt('O_s')
    Op = rdobt('O_p')
    Mgs = rdobt('Mg_s')
    Mgp = rdobt('Mg_p')
    Cs = rdobt('C_s')
    Cp = rdobt('C_p')
    Ns = rdobt('N_s')
    Np = rdobt('N_p')

    AD = {
        'Hs': Hs,
        'Os': Os,
        'Op': Op,
        'Mgs': Mgs,
        'Mgp': Mgp,
        'Cs': Cs,
        'Cp': Cp,
        'Ns': Ns,
        'Np': Np
    }

    print('>>>READING INPUT FILE FINISHED')
    print()

    # --------------------------------------------------
    # Parse energy level input
    # --------------------------------------------------
    energylevel_list = []
    for item in energylevel.split(','):
        if '-' in item:
            start_level = int(item.split('-')[0])
            end_level = int(item.split('-')[1])
            for j in range(start_level, end_level + 1):
                energylevel_list.append(j - 1)
        else:
            energylevel_list.append(int(item) - 1)

    # --------------------------------------------------
    # Run job
    # --------------------------------------------------
    if job == 1:
        # You may also want to update PHImt to accept origin_ang if it writes cube files
        k = PHImt(
            ifcrystal,
            vectorA,
            vectorB,
            vectorC,
            AD,
            atom_xyz,
            c[0],
            c[4],
            c[3],
            d,
            energylevel_list,
            ncpu,
            origin_ang=origin_ang
        )

    elif job == 2:
        # ELF.py should be updated to accept origin_ang and use it in cube output
        m = ELF(
            ifcrystal,
            vectorA,
            vectorB,
            vectorC,
            AD,
            atom_xyz,
            c[0],
            c[4],
            c[3],
            d,
            c[2][0],
            ncpu,
            origin_ang=origin_ang
        )

    else:
        raise ValueError(f"Unsupported job type: {job}")

    end = datetime.datetime.now()
    print('>>>CALCULATION FINISHED')
    print('>>>TIME COST:', end - start)
    print('>>>JOB FINSIED AT:', datetime.datetime.now())


if __name__ == "__main__":
    main()
