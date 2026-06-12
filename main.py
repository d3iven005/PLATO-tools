import os
import datetime
import numpy as np

from src.rdxyz import rdxyz
from src.rdplato_input import rdplato_input
from src.rdwf import rdwf
from src.count_obt import count_obt
from src.crgrid import crgrid
from src.rdobt import rdobt
from src.PHI_MT import PHImt
from src.ELF import ELF
from src.box_for_molecule import build_molecule_box
from src.local_box import build_local_box

import input


def _cell_has_lattice(cell_a, cell_b, cell_c):
    return not (
        np.allclose(cell_a, 0.0)
        and np.allclose(cell_b, 0.0)
        and np.allclose(cell_c, 0.0)
    )


def _read_geometry(jobname):
    xyz_path = os.path.join('00_inputdata', jobname + '.xyz')
    plato_input_path = os.path.join('00_inputdata', jobname + '.in')

    xyz_data = None
    plato_input_data = None

    if os.path.exists(xyz_path):
        xyz_data = rdxyz(os.path.join('00_inputdata', jobname))

    if os.path.exists(plato_input_path):
        plato_input_data = rdplato_input(plato_input_path)

    if xyz_data is None and plato_input_data is None:
        raise FileNotFoundError(
            f"Neither {xyz_path} nor {plato_input_path} exists for jobname {jobname!r}."
        )

    if xyz_data is None:
        print('>>>GEOMETRY READ FROM PLATO INPUT FILE')
        return plato_input_data

    atom_xyz, cell_a, cell_b, cell_c = xyz_data
    if _cell_has_lattice(cell_a, cell_b, cell_c) or plato_input_data is None:
        print('>>>GEOMETRY READ FROM XYZ FILE')
        return xyz_data

    _, input_cell_a, input_cell_b, input_cell_c = plato_input_data
    print('>>>ATOMS READ FROM XYZ FILE; CELL READ FROM PLATO INPUT FILE')
    return atom_xyz, input_cell_a, input_cell_b, input_cell_c


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
    level_chunk_size = getattr(input, 'level_chunk_size', 2)
    use_local_box = getattr(input, 'use_local_box', False)
    local_box_normalize_orbitals = getattr(
        input,
        'local_box_normalize_orbitals',
        False,
    )

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = datetime.datetime.now()

    print('>>>READING INPUT FILE')

    a = _read_geometry(jobname)
    atom_xyz = a[0]

    # --------------------------------------------------
    # Build box and origin
    # --------------------------------------------------
    origin_ang = None
    grid_origin_ang = None
    grid_vectorA = None
    grid_vectorB = None
    grid_vectorC = None

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

    if use_local_box:
        grid_origin_ang, grid_vectorA, grid_vectorB, grid_vectorC = build_local_box(
            input.box_xmin,
            input.box_xmax,
            input.box_ymin,
            input.box_ymax,
            input.box_zmin,
            input.box_zmax,
        )
        print('>>>LOCAL OUTPUT BOX ENABLED')
        print('>>>LOCAL BOX ORIGIN (Angstrom):', grid_origin_ang)
        print('>>>LOCAL BOX VECTORS (Bohr):')
        print('   A =', grid_vectorA)
        print('   B =', grid_vectorB)
        print('   C =', grid_vectorC)
    else:
        grid_origin_ang = origin_ang
        grid_vectorA = vectorA
        grid_vectorB = vectorB
        grid_vectorC = vectorC

    # --------------------------------------------------
    # Read orbital / wavefunction data
    # --------------------------------------------------
    b = count_obt(atom_xyz)
    c = rdwf('00_inputdata/' + jobname, b)

    # --------------------------------------------------
    # Create grid
    # --------------------------------------------------
    d = crgrid(
        N1,
        N2,
        N3,
        grid_vectorA,
        grid_vectorB,
        grid_vectorC,
        origin=grid_origin_ang,
        origin_unit='angstrom',
    )

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
            origin_ang=grid_origin_ang,
            grid_cell_a=grid_vectorA,
            grid_cell_b=grid_vectorB,
            grid_cell_c=grid_vectorC,
            normalize_orbitals=(
                local_box_normalize_orbitals if use_local_box else True
            ),
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
            c[2],
            ncpu,
            origin_ang=grid_origin_ang,
            level_chunk_size=level_chunk_size,
            grid_cell_a=grid_vectorA,
            grid_cell_b=grid_vectorB,
            grid_cell_c=grid_vectorC,
            normalize_orbitals=(
                local_box_normalize_orbitals if use_local_box else True
            ),
        )

    else:
        raise ValueError(f"Unsupported job type: {job}")

    end = datetime.datetime.now()
    print('>>>CALCULATION FINISHED')
    print('>>>TIME COST:', end - start)
    print('>>>JOB FINSIED AT:', datetime.datetime.now())


if __name__ == "__main__":
    main()
