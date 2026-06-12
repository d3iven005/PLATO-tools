import datetime
import os

import numpy as np

import input
from main import _read_geometry
from src.ELF_reciprocal import ELF_reciprocal
from src.box_for_molecule import build_molecule_box
from src.count_obt import count_obt
from src.crgrid import crgrid
from src.rdobt import rdobt
from src.rdwf import rdwf


def _load_orbital_tables():
    return {
        'Hs': rdobt('H_s'),
        'Os': rdobt('O_s'),
        'Op': rdobt('O_p'),
        'Mgs': rdobt('Mg_s'),
        'Mgp': rdobt('Mg_p'),
        'Cs': rdobt('C_s'),
        'Cp': rdobt('C_p'),
        'Ns': rdobt('N_s'),
        'Np': rdobt('N_p'),
    }


def main():
    jobname = input.jobname
    ncpu = input.ncpu
    ifcrystal = input.ifcrystal
    vectorflag = input.vectorflag
    vectorA = input.vectorA
    vectorB = input.vectorB
    vectorC = input.vectorC
    N1 = input.N1
    N2 = input.N2
    N3 = input.N3
    padding_distance = input.padding_distance
    level_chunk_size = getattr(input, 'level_chunk_size', 2)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    start = datetime.datetime.now()

    print('>>>READING INPUT FILE')
    geom = _read_geometry(jobname)
    atom_xyz = geom[0]

    origin_ang = None
    if ifcrystal == 0:
        origin_ang, vectorA, vectorB, vectorC = build_molecule_box(
            atom_xyz,
            padding_angstrom=padding_distance,
        )
        print('>>>NON-PERIODIC MODE: AUTOMATIC MOLECULE BOX ENABLED')
    else:
        if vectorflag == 0:
            vectorA = np.array(geom[1], dtype=float)
            vectorB = np.array(geom[2], dtype=float)
            vectorC = np.array(geom[3], dtype=float)
        else:
            vectorA = np.array(vectorA, dtype=float)
            vectorB = np.array(vectorB, dtype=float)
            vectorC = np.array(vectorC, dtype=float)

    if ifcrystal != 1:
        raise NotImplementedError("main_reciprocal.py currently supports periodic ELF only.")

    basis_counts = count_obt(atom_xyz)
    wf_data = rdwf('00_inputdata/' + jobname, basis_counts)
    xyzgrid = crgrid(
        N1,
        N2,
        N3,
        vectorA,
        vectorB,
        vectorC,
        origin=origin_ang,
        origin_unit='angstrom',
    )
    orbital_tables = _load_orbital_tables()

    print('>>>READING INPUT FILE FINISHED')
    print()

    ELF_reciprocal(
        ifcrystal,
        vectorA,
        vectorB,
        vectorC,
        orbital_tables,
        atom_xyz,
        wf_data[0],
        wf_data[4],
        wf_data[3],
        xyzgrid,
        wf_data[2],
        ncpu,
        origin_ang=origin_ang,
        level_chunk_size=level_chunk_size,
    )

    end = datetime.datetime.now()
    print('>>>RECIPROCAL ELF CALCULATION FINISHED')
    print('>>>TIME COST:', end - start)
    print('>>>JOB FINISHED AT:', datetime.datetime.now())


if __name__ == "__main__":
    main()
