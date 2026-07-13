#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "tb3.h"

/*
 * System description. Allocated in TB_init. Never released
 *
 * The following structures are stored in the stack and are available to all functions in this module.
 * This is because moverInterface is part of a callback system, so data needs to persist between calls.
 * The larger blocks of memory are allocated on the heap using the pointers on the stack to reach them.
 */
multiTimer_t timers;
sparseGeom_t geom;
simulationInfo_t simInfo;
stateInfo_t stateInfo;
eigenStates_t *eigenStates;
atomsData_t atomsData;
GaussData_t *GaussData;
multipole_t multipole;

//##################################
// New guard for the isolated monopole-only implementation.
static int
TB_IsMonopole3D(void)
{
  return (geom.NPerDim == 3) && (multipole.lmax == 0) && (multipole.sizeQ == 1);
}
//##################################

/**********************************************************************
 * This module contains the functions used to implement tight binding *
 **********************************************************************/

int
moverInterface (mover_t *moverInfo, int mode)
{
  /* This function acts as an interface between tb.c and movers.c.
   * Information passes between them using moverInfo.
   * It is called by moversSimulate which controls the movement of atoms */
  int error, i, j, isConverged, Zi;
  char name[2];

  error = 0;
  switch (mode)
    {
    case -1: /* Test code */
      TB_UnitTest ();
      break;
    case 0: /* Initialise */
      /* Initialise tb.c */
      TB_init (moverInfo->outfp, moverInfo->JobName);

      /* Read charges and spins from the restart file if required */
      if(simInfo.readRestartFile > 0)
        TB_ReadRestart (&geom, &simInfo, &stateInfo, &multipole);

      /* Initialise moverInfo */
      moverInfo->jobType = simInfo.jobType;
      moverInfo->Volume = geom.Volume;
      moverInfo->N = geom.N;
      moverInfo->PBCFlag = simInfo.PBCFlag;

      moverInfo->R = calloc (3*moverInfo->N, sizeof(double));
      moverInfo->V = calloc (3*moverInfo->N, sizeof(double));
      moverInfo->A = calloc (3*moverInfo->N, sizeof(double));
      moverInfo->F = calloc (3*moverInfo->N, sizeof(double));
      moverInfo->mass = calloc (moverInfo->N, sizeof(double));

      for (i = 0; i < 3; i++)
        {
          moverInfo->CellRepeat[i] = geom.CellRepeat[i];
          for (j = 0; j < 3; j++)
            {
              moverInfo->CellVec[i][j] = geom.CellVec[i][j];
              moverInfo->ReciprocalVec[i][j] = geom.ReciprocalVec[i][j];
            }
        }
      for (i = 0; i < 3*moverInfo->N; i++)
        moverInfo->R[i] = geom.R[i];
      if (moverInfo->PBCFlag == 1)
        {
          moverPBC (moverInfo);
          for (i = 0; i < 3*moverInfo->N; i++)
            geom.R[i] = moverInfo->R[i];
        }
      for (i = 0; i < moverInfo->N; i++)
        moverInfo->mass[i] = getAtomicMass(geom.AtomicNumber[i]);

      moverInfo->relaxMethod = simInfo.relaxMethod;
      moverInfo->cellRelaxMode = simInfo.cellRelaxMode;
      moverInfo->step = 0;
      moverInfo->nSteps = simInfo.nSteps;
      moverInfo->fTol = simInfo.fTol;
      moverInfo->dRmax = simInfo.maxDisplacement;
      moverInfo->stressTol = simInfo.stressTol;
      moverInfo->relaxFactor = simInfo.relaxFactor;
      moverInfo->latticeRelaxFactor = simInfo.latticeRelaxFactor;
      moverInfo->MDTimeStep = simInfo.MDTimeStep;
      moverInfo->atomTemperature = simInfo.atomTemperature;
      moverInfo->temperatureTolerance = simInfo.temperatureTolerance;
      moverInfo->kT = simInfo.kT;
      moverInfo->dMmax = simInfo.dMmax;
      moverInfo->MCMode = simInfo.MCMode;

      if (moverInfo->jobType == 4)
        {
          switch (moverInfo->MCMode)
            {
            case 1:
              moverInfo->M = calloc (3*moverInfo->N, sizeof(double));
              for (i = 0; i < moverInfo->N; i++)
                {
                  moverInfo->M[3*i + 0] = multipole.Mx_inp[i][0];
                  moverInfo->M[3*i + 1] = multipole.My_inp[i][0];
                  moverInfo->M[3*i + 2] = multipole.Mz_inp[i][0];
                }
              break;
            case 2:
              moverInfo->b = calloc (3*moverInfo->N, sizeof(double));
              for (i = 0; i < moverInfo->N; i++)
                {
                  moverInfo->b[3*i + 0] = multipole.bx[i];
                  moverInfo->b[3*i + 1] = multipole.by[i];
                  moverInfo->b[3*i + 2] = multipole.bz[i];
                }
              break;
            }
        }
      break;
    case 1: /* Finalise */
      TB_WriteResults ();
      timerStop (&timers.timer[0]);
      timerWrite (&timers, moverInfo->outfp);
      break;
    case 2: /* Update for multistep movement */
      /* Retrieve information from the mover routine */
      simInfo.maxF = moverInfo->maxF;
      simInfo.maxStress = moverInfo->maxStress;
      simInfo.step = moverInfo->step;
      for (i = 0; i < 3*moverInfo->N; i++)
        geom.R[i] = moverInfo->R[i];

      if (moverInfo->jobType == 4)
        {
          switch (moverInfo->MCMode)
            {
            case 1:
              for (i = 0; i < moverInfo->N; i++)
                {
                  multipole.Mx_inp[i][0] = moverInfo->M[3*i + 0];
                  multipole.My_inp[i][0] = moverInfo->M[3*i + 1];
                  multipole.Mz_inp[i][0] = moverInfo->M[3*i + 2];
                }
              break;
            case 2:
              for (i = 0; i < moverInfo->N; i++)
                {
                  multipole.bx[i] = moverInfo->b[3*i + 0];
                  multipole.by[i] = moverInfo->b[3*i + 1];
                  multipole.bz[i] = moverInfo->b[3*i + 2];
                }
              break;
            }
        }

      /* Compute the energy and forces */
      isConverged = TB_EnergyForce ();

      /* Return energy and forces to the mover routine */
      moverInfo->Etotal = simInfo.Etotal;
      for (i = 0; i < 3*moverInfo->N; i++)
        moverInfo->F[i] = simInfo.F[i];
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          moverInfo->Flattice[i][j] = simInfo.Flattice[i][j];

      /* Generate a restart file */
      if (simInfo.writeRestartFile > 0)
        if (simInfo.step%simInfo.writeRestartFile == 0)
          TB_WriteRestart (&geom, &simInfo, &stateInfo, &multipole);

      /* Write out the coordinates to make a movie of the simulation */
      if ((simInfo.writeCoordinatesSteps > 0) && (simInfo.step % simInfo.writeCoordinatesSteps == 0))
        TB_WriteCoordinates (&simInfo, &geom, &atomsData);

      /* If convergence failed, determine whether or not to contine the simulation */
      if (isConverged != 1)
        {
          error = 1;
          // This needs more thinking
          /*
             if(!simInfo.continueRelaxationWhenNotConverged && simInfo.maxLoops > 1)
             {
             printf("%i\n",simInfo.continueRelaxationWhenNotConverged);
             printf ("Failed to reach self-consistency\n");
             TB_Stop (simInfo.outfp, 1, &geom, &simInfo, &stateInfo, &multipole);
             }
           */
        }
      break;
    case 3: /* Update cell vectors and related quantities */
      /* Put Bloch vectors into reduced form */
      if (simInfo.model.TightBindingFlag > 0)
        InverseTransformK (stateInfo.nBloch, stateInfo.k, geom.CellVec);

      /* Put atomic coordinates into reduced form */
      InverseTransformR (moverInfo->N, moverInfo->R, geom.ReciprocalVec);

      /* Update cell vectors */
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          geom.CellVec[i][j] += moverInfo->dCellVec[i][j];

      /* Correct cell vector lengths */
      for (i = 0; i < 3; i++)
        geom.CellSize[i] = sqrt(geom.CellVec[i][0]*geom.CellVec[i][0] + geom.CellVec[i][1]*geom.CellVec[i][1] + geom.CellVec[i][2]*geom.CellVec[i][2]);

      /* Get reciprocal lattice vectors */
      GetReciprocal (geom.CellVec, geom.ReciprocalVec, &geom.Volume);

      /* Transform the k points from reciprocal lattice vector fractions to cartesian coordinates */
      if (simInfo.model.TightBindingFlag > 0)
        TransformK (stateInfo.nBloch, stateInfo.k, geom.ReciprocalVec);

      /* Put atomic coordinates back into cartesian coordinates */
      TransformR (moverInfo->N, moverInfo->R, geom.CellVec);
      for (i = 0; i < 3*moverInfo->N; i++)
        geom.R[i] = moverInfo->R[i];
      break;
    default:
      printf ("TB moverInterface: Unknown mode %i\n", mode);
      error = 1;
      break;
    }

  return error;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_init (FILE *outfp, char *JobName)
{
  /* Initialise the simulation */
  int i, j, k, l, n, s;
  char **AtomFileName;
  char name[2];

  /* Initialise this module */
  static int init = 1;

  /* If the initialisation has not been carried out, then do it now */
  if (init == 1)
    {
      /* Set up timers */
      timers.nTimer = 6;
      timers.timer = malloc (timers.nTimer*sizeof(p_timer_t));
      strncpy(timers.timer[0].label, "Total", T_LABEL_LEN);
      strncpy(timers.timer[1].label, "Build Hamiltonian", T_LABEL_LEN);
      strncpy(timers.timer[2].label, "Diagonalisation", T_LABEL_LEN);
      strncpy(timers.timer[3].label, "Build density matrix", T_LABEL_LEN);
      strncpy(timers.timer[4].label, "Electrostatics", T_LABEL_LEN);
      strncpy(timers.timer[5].label, "Build forces", T_LABEL_LEN);
      timerInit (&timers);
      timerStart (&timers.timer[0]);

      /* Store the pointer to the output file so that it can be accessed elsewhere */
      simInfo.JobName = JobName;

      /* Store the pointer to the output file so that it can be accessed elsewhere */
      simInfo.outfp = outfp;
      simInfo.model.outfp = outfp;

      /* Write header */
      fprintf (simInfo.outfp, "    -----------------\n");
      fprintf (simInfo.outfp, "    | Tight Binding |\n");
      fprintf (simInfo.outfp, "    -----------------\n\n");

      /* Read data from the input file */
      TB_input (&stateInfo, &simInfo);
      simInfo.model.verbosity = simInfo.verbosity;

      /* Set up file name information */
      atomsData.nTypes = simInfo.NTypes;
      atomsData.ad = malloc (atomsData.nTypes*sizeof (atomData_t));
      for (i = 0; i < atomsData.nTypes; i++)
        {
          /* Store the chemical symbols for each atom type */
          if (simInfo.Name[2*i + 1] != ' ')
            {
              /* Two character chemical symbol */
              atomsData.ad[i].name[0] = simInfo.Name[2*i + 0];
              atomsData.ad[i].name[1] = simInfo.Name[2*i + 1];
              atomsData.ad[i].name[2] = '\0';
            }
          else
            {
              /* One character chemical symbol */
              atomsData.ad[i].name[0] = simInfo.Name[2*i + 0];
              atomsData.ad[i].name[1] = '\0';
            }
        }

      /* Read data from the integral files */
      tablesInit (&simInfo.model, &atomsData);

      /* Read in the gaussian expansions */
      if ((simInfo.model.TightBindingFlag == 3)||(simInfo.model.ThreeCenterFlag == 1))
        {
          AtomFileName = malloc(atomsData.nTypes*sizeof(char *));
          for (i = 0; i < atomsData.nTypes; i++)
            AtomFileName[i] = atomsData.ad[i].name;
          GaussData = malloc (atomsData.nTypes*sizeof(GaussData_t));
          GaussRead (simInfo.model.DataPath, AtomFileName, atomsData.nTypes, GaussData);
          free (AtomFileName);
        }
      else
        GaussData = NULL;

      /* Set up geometric data */
      geom.diameterNLV = simInfo.diameterNLV;
      geom.Ntype = simInfo.NTypes;
      geom.N = simInfo.NAtom;
      for (i = 0; i < 3; i++)
        {
          geom.CellSize[i] = simInfo.CellSize[i];
          geom.CellRepeat[i] = simInfo.CellRepeat[i];
          for (j = 0; j < 3; j++)
            geom.CellVec[i][j] = simInfo.CellVec[i][j];
        }
      geom.Z = malloc (geom.N*sizeof(int));
      for (i = 0; i < geom.N; i++)
        geom.Z[i] = simInfo.AtomType[i];
      geom.AtomicNumber = malloc (geom.N*sizeof(int));
      for (i = 0; i < geom.N; i++)
        {
          name[0] = simInfo.Name[2*geom.Z[i]+0];
          name[1] = simInfo.Name[2*geom.Z[i]+1];
          geom.AtomicNumber[i] = getAtomicNumber(name);
        }
      geom.ZCore = malloc (simInfo.NAtom*sizeof(double));
      for (i = 0; i < simInfo.NAtom; i++)
        geom.ZCore[i] = atomsData.ad[geom.Z[i]].ZCore;
      geom.nOrb = malloc (geom.N*sizeof(int));
      for (i = 0; i < geom.N; i++)
        geom.nOrb[i] = atomsData.ad[geom.Z[i]].nOrb;
      geom.i = malloc (geom.N*sizeof(int));
      for (i = 0; i < geom.N; i++)
        geom.i[i] = i;
      geom.radius = malloc (geom.N*sizeof(double));
      for (i = 0; i < geom.N; i++)
        geom.radius[i] = atomsData.ad[geom.Z[i]].rCut;
      geom.R = malloc (3*geom.N*sizeof(double));
      for (i = 0; i < 3*geom.N; i++)
        geom.R[i] = simInfo.Pos[i];
      geom.RCell = malloc (3*geom.N*sizeof(double));
      for (i = 0; i < 3*geom.N; i++)
        geom.RCell[i] = 0.0;

      /* Get reciprocal lattice vectors */
      GetReciprocal (geom.CellVec, geom.ReciprocalVec, &geom.Volume);

      /* Transform the k points from reciprocal lattice vector fractions to cartesian coordinates */
      if (simInfo.model.TightBindingFlag > 0)
        TransformK (stateInfo.nBloch, stateInfo.k, geom.ReciprocalVec);

      if (simInfo.model.TightBindingFlag > 0)
        {
          /* Build the dimension of the main electronic matrices, and pointers into them */
          geom.Hindex = malloc ((geom.N+1)*sizeof(int));
          geom.Hindex[0] = 0;
          geom.nOrbCell = 0;
          for (i = 0; i < geom.N; i++)
            {
              n = geom.nOrb[i];
              geom.nOrbCell += n;
              geom.Hindex[i+1] = geom.Hindex[i] + n;
            }

          /* Initialise eigenstates */
          stateInfo.overlapType = simInfo.model.OverlapFlag;
          stateInfo.Norbitals = geom.nOrbCell;
          statesInit (&stateInfo);
          eigenStatesInit (&stateInfo, &eigenStates);
        }

      /* Set up simulation data */
      if (simInfo.model.TightBindingFlag > 0)
        {
          stateInfo.occ = malloc (stateInfo.nSets*sizeof(double *));
          for (s = 0; s < stateInfo.nSets; s++)
            stateInfo.occ[s] = malloc (eigenStates[s].nStates*sizeof(double));
        }
      simInfo.F0 = malloc (3*geom.N*sizeof(double));
      simInfo.F1 = malloc (3*geom.N*sizeof(double));
      simInfo.F2 = malloc (3*geom.N*sizeof(double));
      simInfo.F = malloc (3*geom.N*sizeof(double));
      simInfo.Fexternal = malloc (3*geom.N*sizeof(double));

      if (simInfo.model.TightBindingFlag > 0)
        {
          /* Compute the number of electrons in the system */
          simInfo.nElectrons = simInfo.electronExcess;
          for (i = 0; i < geom.N; i++)
            simInfo.nElectrons += geom.ZCore[i];

          /* Allocate memory for the multipoles */
          statesMultipoleInit(&simInfo, &stateInfo, &geom, &multipole);

          /* Initialise the Madelung matrix */
          electroInit (&geom, &atomsData, &multipole);

          /* Initialise the Stoner integrals */
          magneticInit (stateInfo.spinType, &simInfo, &geom, &atomsData, &multipole);
        }

      simInfo.maxF = 0.0;
      simInfo.maxStress = 0.0;

      /* Set the flag to show initialisation has been carried out */
      init = 0;
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_EnergyForce ()
{
  /* This function computes the total energy and atomic forces */
  ws_t ws;
  int s, isConverged, i;

  /* Start Hamiltonian timer */
  timerStart (&timers.timer[1]);

  /* Allocate memory to store the matrices */
  TB_Alloc(&ws);

  /* Stop Hamiltonian timer */
  timerStop (&timers.timer[1]);

  /* Find eigenvalues for the overlap matrix.
   * Small values magnify errors in the Hamiltonian matrix, producing large errors in its eigenvalues. */
  if (simInfo.model.OverlapFlag == 1)
    if (simInfo.testOverlapFlag == 1)
      for (s = 0; s < stateInfo.nSets; s++)
        TB_TestOverlap (&eigenStates[s]);

  /* Build the electrostatic multipole moments */
  timerStart (&timers.timer[4]);
  switch (simInfo.model.TightBindingFlag)
    {
    case 0:
      break;
    case 3:
      statesBuildMultipoleIntegrals(&geom, &GaussData, &multipole);
      statesBuildMultipoleMoments(&geom, &multipole, &ws.sparseRho);
      break;
    default:
      tablesBuildMultipoleIntegrals(&geom, &atomsData, &multipole);
      statesBuildMultipoleMoments(&geom, &multipole, &ws.sparseRho);
    }
  timerStop (&timers.timer[4]);

  /* Compute energy and force from external model */
  externalModel (simInfo.externalModelFlag, geom.CellVec, geom.N, geom.R, geom.AtomicNumber, &simInfo.Eexternal, simInfo.Fexternal);

  /* Run the SCF Loop */
  isConverged = TB_SCF(&ws);

  /* Assemble the energy and the forces */
  TB_AssembleEnergy(&ws);
  timerStart (&timers.timer[5]);
  TB_AssembleForce(&ws);
  timerStop (&timers.timer[5]);

  /* Write out the Fock matrix */
  if (simInfo.model.TightBindingFlag > 0)
    if (simInfo.verbosity > 5)
      TB_WriteHamiltonian (&ws, &geom, &simInfo, &stateInfo);

  /* Write overlap matrix to file */
  if (simInfo.model.TightBindingFlag > 0)
    if (simInfo.writeOverlapFlag == 1)
      TB_WriteOverlap (&ws, &geom, &simInfo, &stateInfo);

  /* Free up the memory allocated */
  TB_Free(&ws);

  return isConverged;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_Alloc(ws_t *ws)
{
  /* Allocate the arrays required for evaluating the Hamiltonian */
  int i, j, p, s, isConverged;

  /* Build the extended list of atoms that includes periodic images */
  sparseCellRepeat(&geom);
  sparseExtendedAtomList (&geom);

  /* Build the neighbour list */
  sparseNeighbourList (&geom);

  if (simInfo.model.TightBindingFlag > 0)
    {
      /* Allocate space for the sparse matrices */
      switch (stateInfo.spinType)
        {
        case 0: /* No spin */
          ws->sparseH.n = 1;
          sparseCreateMatrices (&geom, &ws->sparseH, 1);
          break;
        case 1: /* Collinear spin */
          ws->sparseH.n = 2;
          sparseCreateMatrices (&geom, &ws->sparseH, 1);
          break;
        case 2: /* Non-collinear spin */
          ws->sparseH.n = 3;
          sparseCreateMatrices (&geom, &ws->sparseH, 2);

          ws->S_x.n = ws->S_y.n = ws->S_z.n = 3;
          sparseCreateMatrices (&geom, &ws->S_x, 2);
          sparseCreateMatrices (&geom, &ws->S_y, 2);
          sparseCreateMatrices (&geom, &ws->S_z, 2);
          magneticAssignSpinMatrices(stateInfo.spinType, &geom,  &atomsData, &simInfo, &ws->S_x, &ws->S_y, &ws->S_z);

          ws->L_x.n = ws->L_y.n = ws->L_z.n = 3;
          sparseCreateMatrices (&geom, &ws->L_x, 2);
          sparseCreateMatrices (&geom, &ws->L_y, 2);
          sparseCreateMatrices (&geom, &ws->L_z, 2);
          magneticAssignAngularMatrices(stateInfo.spinType, &geom,  &atomsData, &simInfo, &ws->L_x, &ws->L_y, &ws->L_z);

          ws->mu_x.n = ws->mu_y.n = ws->mu_z.n = 3;
          sparseCreateMatrices (&geom, &ws->mu_x, 2);
          sparseCreateMatrices (&geom, &ws->mu_y, 2);
          sparseCreateMatrices (&geom, &ws->mu_z, 2);
          sparseAddMatrices (&geom, &ws->mu_x, -0.5, &ws->L_x, -1.0, &ws->S_x);
          sparseAddMatrices (&geom, &ws->mu_y, -0.5, &ws->L_y, -1.0, &ws->S_y);
          sparseAddMatrices (&geom, &ws->mu_z, -0.5, &ws->L_z, -1.0, &ws->S_z);
          ws->sparseIdentity.n = 3;
          sparseCreateMatrices (&geom, &ws->sparseIdentity, 2);
          sparseIdentityMatrices (&geom, &ws->sparseIdentity);
          break;
        }

      if (simInfo.model.OverlapFlag == 1)
        {
          sparseCreateMatrix (&geom, &ws->sparseS, 1);
          switch (stateInfo.spinType)
            {
            case 0: /* No spin */
            case 1: /* Collinear spin */
              sparseCreateMatrix (&geom, &ws->sparseEmatrix, 1);
              break;
            case 2: /* Non-collinear spin */
              sparseCreateMatrix (&geom, &ws->sparseEmatrix, 2);
              break;
            }
        }
      else
        {
          ws->sparseS.v = NULL;
          ws->sparseEmatrix.v = NULL;
        }

      if (simInfo.model.SCFFlag > 0)
        {
          switch (stateInfo.spinType)
            {
            case 0: /* No spin */
              sparseCreateMatrix (&geom, &ws->sparseRho, 1);
              ws->sparseFock.n = 1;
              sparseCreateMatrices (&geom, &ws->sparseFock, 1);
              ws->sparseM.n = 0;
              ws->sparseM.M = NULL;
              break;
            case 1: /* Collinear spin */
              /* Block[0]: uu
               * Block[1]: dd */
              sparseCreateMatrix (&geom, &ws->sparseRho, 1);
              ws->sparseFock.n = 2;
              sparseCreateMatrices (&geom, &ws->sparseFock, 1);
              ws->sparseHMagnetic.n = 2;
              sparseCreateMatrices (&geom, &ws->sparseHMagnetic, 1);
              ws->sparseM.n = 1;
              sparseCreateMatrices (&geom, &ws->sparseM, 1);
              break;
            case 2: /* Non-collinear spin */
              /* Block[0]: uu
               * Block[1]: dd
               * Block[3]: ud */
              sparseCreateMatrix (&geom, &ws->sparseRho, 2);
              ws->sparseFock.n = 3;
              sparseCreateMatrices (&geom, &ws->sparseFock, 2);
              ws->sparseHMagnetic.n = 3;
              sparseCreateMatrices (&geom, &ws->sparseHMagnetic, 2);
              ws->sparseM.n = 3;
              sparseCreateMatrices (&geom, &ws->sparseM, 2);
              ws->sparseHDipole.n = 3;
              sparseCreateMatrices (&geom, &ws->sparseHDipole, 2);
              ws->sparseSOC.n = 3;
              sparseCreateMatrices (&geom, &ws->sparseSOC, 2);
              break;
            }
        }
      else
        {
          sparseCreateMatrix (&geom, &ws->sparseRho, 1);
          ws->sparseFock.n = 1;
          sparseCreateMatrices (&geom, &ws->sparseFock, 1);
          ws->sparseM.n = 0;
          ws->sparseM.M = NULL;
        }

      multipole.Qint = malloc (multipole.sizeQ*sizeof(sparseMatrix_t));
      for (p = 0; p < multipole.sizeQ; p++)
        sparseCreateMatrix (&geom, &multipole.Qint[p], 1);

      /* Build the Madelung matrix and allocate memory for the electrostatic Hamiltonian */
      if (simInfo.model.SCFFlag > 0)
        {
//##################################
          /* Monopole-only 3D uses electros_monopole.c and does not build Madelung. */
          if (!TB_IsMonopole3D())
            electroMadelung (&geom, &multipole);
//##################################
          switch (stateInfo.spinType)
            {
            case 0: /* No spin */
              ws->sparseHHartree.n = 1;
              sparseCreateMatrices (&geom, &ws->sparseHHartree, 1);
              break;
            case 1: /* Collinear spin */
              ws->sparseHHartree.n = 2;
              sparseCreateMatrices (&geom, &ws->sparseHHartree, 1);
              break;
            case 2: /* Non-collinear spin */
              ws->sparseHHartree.n = 3;
              sparseCreateMatrices (&geom, &ws->sparseHHartree, 1);
              break;
            }
        }
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_Free(ws_t *ws)
{
  /* Frees all of the arrays allocated in TB_Alloc() */

  if (simInfo.model.TightBindingFlag > 0)
    {
      /* Free temporary storage */
      free_sparseMatrices (&geom, &ws->sparseH);
      free_sparseMatrix (&geom, &ws->sparseRho);

      switch (stateInfo.spinType)
        {
        case 0: /* No spin */
          break;
        case 1: /* Collinear spin */
          break;
        case 2: /* Non-collinear spin */
          free_sparseMatrices (&geom, &ws->S_x);
          free_sparseMatrices (&geom, &ws->S_y);
          free_sparseMatrices (&geom, &ws->S_z);
          free_sparseMatrices (&geom, &ws->L_x);
          free_sparseMatrices (&geom, &ws->L_y);
          free_sparseMatrices (&geom, &ws->L_z);
          free_sparseMatrices (&geom, &ws->mu_x);
          free_sparseMatrices (&geom, &ws->mu_y);
          free_sparseMatrices (&geom, &ws->mu_z);
          break;
        }

      if (simInfo.model.OverlapFlag == 1)
        {
          free_sparseMatrix (&geom, &ws->sparseS);
          free_sparseMatrix (&geom, &ws->sparseEmatrix);
        }
      free_sparseMatrices (&geom, &ws->sparseFock);
      if (simInfo.magneticFieldFlag > 0)
        free_sparseMatrices (&geom, &ws->sparseHDipole);
      if (simInfo.socFlag == 1)
        free_sparseMatrices (&geom, &ws->sparseSOC);

      for (int p = 0; p < multipole.sizeQ; p++)
        free_sparseMatrix (&geom, &multipole.Qint[p]);
      free (multipole.Qint);

      if (simInfo.model.SCFFlag > 0)
        {
          free_sparseMatrices (&geom, &ws->sparseHHartree);
          if (stateInfo.spinType > 0)
            {
              free_sparseMatrices (&geom, &ws->sparseHMagnetic);
              free_sparseMatrices (&geom, &ws->sparseM);
            }
        }
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

double
TB_angle_between(double u_x, double u_y, double u_z, double v_x, double v_y, double v_z)
{
  /* Returns the central angle in radians between vectors u and v */
  double norm_u, norm_v;

  /* Normalize u & v */
  norm_u = sqrt(u_x*u_x + u_y*u_y + u_z*u_z);
  u_x /= norm_u;
  u_y /= norm_u;
  u_z /= norm_u;

  norm_v = sqrt(v_x*v_x + v_y*v_y + v_z*v_z);
  v_x /= norm_v;
  v_y /= norm_v;
  v_z /= norm_v;

  return acos(u_x*v_x + u_y*v_y + u_z*v_z);
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_SCF(ws_t *ws)
{
  /* Self-consistency loop */
  int loop, isConverged, i;
  double residue;
  int j, j0, n, Li, Lj, c, ni, nj;

  if (simInfo.model.TightBindingFlag > 0)
    {
      /* Build the sparse Hamiltonian and overlap matrices */
      TB_BuildH(ws);

      isConverged = 0;
      simInfo.mixLevels = 0;
      simInfo.spinMixLevels = 0;
      for (loop = 0; (loop < simInfo.maxLoops) && (isConverged == 0); loop++)
        {
          /* Diagonalize the Hamiltonian */
          TB_DiagonalizeHamiltonian(ws);

          /* Start density matrix timer */
          timerStart (&timers.timer[3]);

          /* Find the occupancies */
          simInfo.mu = statesFindMu (simInfo.nElectrons, simInfo.kT, &stateInfo, &eigenStates);

          /* Build the sparse density matrices */
          statesBuildDensityMatrix (&stateInfo, &geom, &eigenStates, &ws->sparseRho, &ws->sparseEmatrix, &ws->sparseM);

          /* Stop density matrix timer */
          timerStop (&timers.timer[3]);

          /* Build the electrostatic multipole moments */
          timerStart (&timers.timer[4]);
          statesBuildMultipoleMoments(&geom, &multipole, &ws->sparseRho);
          timerStop (&timers.timer[4]);

          /* Build the magnetic multipole moments */
          if (stateInfo.spinType > 0)
            statesBuildMagneticMoments(stateInfo.spinType, &geom, &multipole, &ws->sparseM);

          /* Write out multipoles */
          if (simInfo.verbosity > 2)
            TB_WriteMultipoles (&geom, &simInfo, &stateInfo, &multipole);

          /* Assemble the energy */
          TB_AssembleEnergy(ws);

          /* Update the input moments */
          residue = statesUpdateInputMoments(stateInfo.spinType, &geom, &simInfo, &multipole);

          /* Determine whether or not the simulation is converged */
          if (residue < simInfo.residueTolerance)
            isConverged = 1;

          if(simInfo.model.SCFFlag == 0)
            isConverged = 1;

          /* Report on progress */
          TB_ReportProgress(loop, residue);
        }
    }
  else
    isConverged = 1;

  return isConverged;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_DiagonalizeHamiltonian(ws_t *ws)
{
  /* Diagonalize the Hamiltonian */

  /* Start Hamiltonian timer */
  timerStart (&timers.timer[1]);

  /* Build the Fock Hamiltonian */
  TB_BuildFock(ws);

  /* Stop Hamiltonian timer */
  timerStop (&timers.timer[1]);

  /* Start diagonalisation timer */
  timerStart (&timers.timer[2]);

  /* Build and diagonalise the matrices for each set */
  if (statesDiagonalize (&stateInfo, &geom, &ws->sparseFock, &ws->sparseS, &eigenStates) == 1)
    TB_Stop (simInfo.outfp, 1, &geom, &simInfo, &stateInfo, &multipole);

  /* Stop diagonalisation timer */
  timerStop (&timers.timer[2]);

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_BuildFock(ws_t *ws)
{
  /* This function evaluates the entire Hamiltonian, which is stored in sparseFock */
  /* Assemble the Fock matrix */
  if (simInfo.model.SCFFlag > 0)
    {
      timerStart (&timers.timer[4]);
//##################################
      if (TB_IsMonopole3D())
        electroMonopoleHamiltonian3D (&geom, &multipole, &ws->sparseHHartree.M[0], &ws->sparseS);
      else
        electroHamiltonian (&geom, &multipole, &ws->sparseHHartree.M[0], &ws->sparseS);
//##################################
      timerStop (&timers.timer[4]);
      switch (stateInfo.spinType)
        {
        case 0: /* No spin */
          /* Add Hartree Term */
          sparseAddMatrices (&geom, &ws->sparseFock, 1.0, &ws->sparseH, 1.0, &ws->sparseHHartree);
          break;
        case 1: /* Collinear spin */
          /* Add Hartree Term */
          sparseCopyMatrix (&geom, &ws->sparseHHartree.M[1], &ws->sparseHHartree.M[0]);
          sparseAddMatrices (&geom, &ws->sparseFock, 1.0, &ws->sparseH, 1.0, &ws->sparseHHartree);

          /* Add Stoner exchange term */
          if(simInfo.stonerFlag > 0)
            {
              magneticHamiltonian (stateInfo.spinType, &geom, &multipole, &ws->sparseHMagnetic);
              sparseAddMatrices (&geom, &ws->sparseFock, 1.0, &ws->sparseFock, 1.0, &ws->sparseHMagnetic);
            }
          break;
        case 2: /* Non-collinear spin */
          /* Add Hartree Term */
          sparseCopyMatrix (&geom, &ws->sparseHHartree.M[1], &ws->sparseHHartree.M[0]);
          sparseZeroMatrix (&geom, &ws->sparseHHartree.M[2]);
          sparseAddMatrices (&geom, &ws->sparseFock, 1.0, &ws->sparseH, 1.0, &ws->sparseHHartree);

          /* Add Stoner exchange term */
          if (simInfo.stonerFlag > 0)
            {
              magneticHamiltonian (stateInfo.spinType, &geom, &multipole, &ws->sparseHMagnetic);
              sparseAddMatrices (&geom, &ws->sparseFock, 1.0, &ws->sparseFock, 1.0, &ws->sparseHMagnetic);
            }

          /* Perform Peierls Transformation */
          if (simInfo.magneticFieldFlag > 0)
            if (simInfo.peierlsFlag > 0)
              magneticPeierlsSubstitution_sparse(&geom, &simInfo, &ws->sparseFock, 1);

          break;
        }
    }
  else
    sparseCopyMatrices (&geom, &ws->sparseFock, &ws->sparseH);

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int TB_ReportProgress(int loop, double residue)
{
  /* Report on progress */
  double meanSpin_inp, meanSpin_out;
  int i;

  if (simInfo.model.SCFFlag != 0)
    {
      switch (stateInfo.spinType)
        {
        case 0: /* No spin */
          printf ("SCF: [%3i] E = %14.7g Ry  Residue = %9.2g\n", loop, simInfo.Etotal, residue);
          if (simInfo.verbosity > 2)
            fprintf (simInfo.outfp, "E0 = %14.7f  E1 = %14.7f  E2 = %14.7f\n", simInfo.E0, simInfo.E1, simInfo.E2);
          fprintf (simInfo.outfp, "SCF: [%3i] E = %14.7g Ry  Residue = %9.2g\n", loop, simInfo.Etotal, residue);
          break;
        case 1: /* Collinear spin */
          meanSpin_inp = 0.0;
          meanSpin_out = 0.0;
          for (i = 0; i < geom.N; i++)
            {
              meanSpin_inp += multipole.Mz_inp[i][0];
              meanSpin_out += multipole.Mz_out[i][0];
            }
          meanSpin_inp /= (double)geom.N;
          meanSpin_out /= (double)geom.N;
          printf ("SCF: [%3i] E = %14.7g Ry = %14.7g eV Spin (in) = %7.3f Spin (out) = %7.3f Residue = %9.2g\n", loop, simInfo.Etotal, simInfo.Etotal*RYDBERG/EV, meanSpin_inp, meanSpin_out, residue);
          if (simInfo.verbosity > 2)
            fprintf (simInfo.outfp, "E0 = %14.7f  E1 = %14.7f  E2 = %14.7f\n", simInfo.E0, simInfo.E1, simInfo.E2);
          fprintf (simInfo.outfp, "SCF: [%3i] E = %14.7g Ry = %14.7g eV Spin (in) = %7.3f Spin (out) = %7.3f Residue = %9.2g\n", loop, simInfo.Etotal, simInfo.Etotal*RYDBERG/EV, meanSpin_inp, meanSpin_out, residue);
          if (simInfo.verbosity > 5)
            for (i = 0; i < geom.N; i++)
              printf ("[%4i] %9.5f\n", i, multipole.Mz_out[i][0]);
          break;
        case 2: /* Non-collinear spin */
          meanSpin_inp = 0.0;
          meanSpin_out = 0.0;
          for (i = 0; i < geom.N; i++)
            {
              meanSpin_inp += multipole.Mx_inp[i][0]*multipole.Mx_inp[i][0] + multipole.My_inp[i][0]*multipole.My_inp[i][0] + multipole.Mz_inp[i][0]*multipole.Mz_inp[i][0];
              meanSpin_out += multipole.Mx_out[i][0]*multipole.Mx_out[i][0] + multipole.My_out[i][0]*multipole.My_out[i][0] + multipole.Mz_out[i][0]*multipole.Mz_out[i][0];
            }
          meanSpin_inp = sqrt(meanSpin_inp/(double)geom.N);
          meanSpin_out = sqrt(meanSpin_out/(double)geom.N);
          printf ("SCF: [%3i] E = %14.7g Ry = %14.15g eV Spin (in) = %7.3f Spin (out) = %7.3f Residue = %9.15f\n", loop, simInfo.Etotal, simInfo.Etotal*RYDBERG/EV, meanSpin_inp, meanSpin_out, residue);
          if (simInfo.verbosity > 2)
            fprintf (simInfo.outfp, "E0 = %14.7f  E1 = %14.7f  E2 = %14.7f\n", simInfo.E0, simInfo.E1, simInfo.E2);
          fprintf (simInfo.outfp, "SCF: [%3i] E = %14.7g Ry = %14.15g eV Spin (in) = %7.3f Spin (out) = %7.3f Residue = %9.15f\n", loop, simInfo.Etotal, simInfo.Etotal*RYDBERG/EV, meanSpin_inp, meanSpin_out, residue);
          break;
        }
      fflush(stdout);
      fflush(simInfo.outfp);
    }
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_BuildH(ws_t * ws)
{
  /* TB_BuildH calculates the static Hamiltonian, i.e. terms that are linear in rho and do not depend on time.
   * This should only need to be evaluated once for each configuration of atoms. */

  /* Build the sparse Hamiltonan and overlap matrix */
  int i, j, k, n;
  int Ni, Nj;
  double f, *bufferH;

  /* Build the sparse Hamiltonian and overlap matrices
   * Loop over atoms in the central cell */
  for (i = 0; i < geom.N; i++)
    {
      Ni = geom.nOrb[i];

      /* Loop over neighbours of those atoms */
      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
        {
          j = geom.neighbourList[i].neighbour[n];
          Nj = geom.nOrb[j];

          /* Allocate temporary storage */
          bufferH = malloc(Ni*Nj*sizeof(double));

          /* Add in the one and two centre terms */
          for (k = 0; k < Ni*Nj*ws->sparseH.M[0].dataSize; k++)
            (ws->sparseH.M[0]).v[i][n][k] = 0.0;

          if (simInfo.model.OverlapFlag == 1)
            {
              for (k = 0; k < Ni*Nj; k++)
                ws->sparseS.v[i][n][k] = 0.0;
              tablesBuildBlock (&geom, &atomsData, &simInfo.model, i, j, ws->sparseS.v[i][n], bufferH);
            }
          else
            tablesBuildBlock (&geom, &atomsData, &simInfo.model, i, j, NULL, bufferH);

          switch (ws->sparseH.M[0].dataSize)
            {
            case 1:
              for (k = 0; k < Ni*Nj; k++)
                (ws->sparseH.M[0]).v[i][n][k] += bufferH[k];
              break;
            case 2:
              for (k = 0; k < Ni*Nj; k++)
                (ws->sparseH.M[0]).v[i][n][2*k + 0] += bufferH[k];
              break;
            }

          /* Add in the three centre terms */
          if (simInfo.model.ThreeCenterFlag == 1)
            {
              threeCentreHop (&geom, GaussData, i, j, bufferH);
              switch (ws->sparseH.M[0].dataSize)
                {
                case 1:
                  for (k = 0; k < Ni*Nj; k++)
                    (ws->sparseH.M[0]).v[i][n][k] += bufferH[k];
                  break;
                case 2:
                  for (k = 0; k < Ni*Nj; k++)
                    (ws->sparseH.M[0]).v[i][n][2*k + 0] += bufferH[k];
                  break;
                }
            }

          /* Free temporary storage */
          free (bufferH);
        }
    }

  switch (stateInfo.spinType)
    {
    case 0: /* No spin */
      break;
    case 1: /* Collinear spin */
      sparseCopyMatrix(&geom, &ws->sparseH.M[1], &ws->sparseH.M[0]);
      break;
    case 2: /* Non-collinear spin */
      sparseCopyMatrix(&geom, &ws->sparseH.M[1], &ws->sparseH.M[0]);
      sparseZeroMatrix(&geom, &ws->sparseH.M[2]);

      /* Add SOC term */
      if (simInfo.socFlag == 1)
        {
          magneticSOCHamiltonian (&atomsData, stateInfo.spinType, &geom, &ws->sparseSOC);
          sparseAddMatrices (&geom, &ws->sparseH, 1.0, &ws->sparseH, 1.0, &ws->sparseSOC);
        }

      /* Add magnetic dipole term */
      if ((simInfo.magneticFieldFlag > 0) || (simInfo.spinConstrainFlag > 0))
        {
          magneticDipoleHamiltonian (stateInfo.spinType, &geom, &multipole, &atomsData, &simInfo, &ws->sparseHDipole);
          sparseAddMatrices (&geom, &ws->sparseH, 1.0, &ws->sparseH, 1.0, &ws->sparseHDipole);
        }

      break;
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_AssembleEnergy (ws_t *ws)
{
  /* Compute the sum of atomic energy contributions */
  simInfo.Eatom = 0.0;

  /* Compute the zeroth order energy */
  TB_E0();

  if (simInfo.model.TightBindingFlag > 0)
    {
      /* Compute the first order energy */
      TB_E1(ws);

      /* Compute the second order energy */
      TB_E2();
    }
  else
    {
      simInfo.E1 = 0.0;
      simInfo.E2 = 0.0;
      simInfo.Eentropy = 0.0;
    }

  /* Assemble the total energy */
  simInfo.Etotal = simInfo.E0 + simInfo.E1 + simInfo.E2 - simInfo.Eentropy + simInfo.Eexternal - simInfo.Econstraint;

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_AssembleForce (ws_t *ws)
{
  int i, j;

  /* Build the zeroth order forces */
  TB_F0();

  if (simInfo.model.TightBindingFlag > 0)
    {
      /* Build the first order forces */
      TB_F1(ws);

      /* Build the second order forces */
      TB_F2(ws);
    }
  else
    {
      for (i = 0; i < 3*geom.N; i++)
        {
          simInfo.F1[i] = 0.0;
          simInfo.F2[i] = 0.0;
        }
    }

  /* Assemble the total force */
  for (i = 0; i < 3*geom.N; i++)
    simInfo.F[i] = simInfo.F0[i] + simInfo.F1[i] + simInfo.F2[i] + simInfo.Fexternal[i];
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      simInfo.Flattice[i][j] = simInfo.Flattice0[i][j] + simInfo.Flattice1[i][j] + simInfo.Flattice2[i][j];

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_E0()
{
  /* Compute the zeroth order energy from the pair potential and pair functional */
  int i, j, n, Zi, Zj;
  double Ri[3], Rj[3], dR[3], magdR, sum;

  /* 1. Pair potential energy */
  simInfo.E0 = 0.0;
  for (i = 0; i < geom.N; i++)
    {
      Zi = geom.Z[i];
      Ri[0] = geom.R[3*i+0];
      Ri[1] = geom.R[3*i+1];
      Ri[2] = geom.R[3*i+2];
      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
        {
          j = geom.neighbourList[i].neighbour[n];
          if (j != i)
            {
              Zj = geom.Z[j];
              Rj[0] = geom.R[3*j+0];
              Rj[1] = geom.R[3*j+1];
              Rj[2] = geom.R[3*j+2];

              dR[0] = Rj[0] - Ri[0];
              dR[1] = Rj[1] - Ri[1];
              dR[2] = Rj[2] - Ri[2];
              magdR = sqrt (dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2]);
              simInfo.E0 += 0.5*tablesPP (Zi, Zj, magdR);
            }
        }
    }

  /* 2. Pair functional energy */
  if (simInfo.model.PairFunctionalFlag == 1)
    {

      for (i = 0; i < geom.N; i++)
        {
          Zi = geom.Z[i];
          Ri[0] = geom.R[3*i+0];
          Ri[1] = geom.R[3*i+1];
          Ri[2] = geom.R[3*i+2];
          sum = 0.0;
          for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
            {
              j = geom.neighbourList[i].neighbour[n];
              if (j != i)
                {
                  Zj = geom.Z[j];
                  Rj[0] = geom.R[3*j+0];
                  Rj[1] = geom.R[3*j+1];
                  Rj[2] = geom.R[3*j+2];
                  dR[0] = Rj[0] - Ri[0];
                  dR[1] = Rj[1] - Ri[1];
                  dR[2] = Rj[2] - Ri[2];
                  magdR = sqrt (dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2]);
                  sum += tablesEP (Zi, Zj, magdR);
                }
            }
          simInfo.E0 += tablesEmbed (&atomsData, Zi, sum);
        }
    }

  /* 3. Three centre correction */
  if (simInfo.model.ThreeCenterFlag == 1)
    simInfo.E0 += threeCentreExc (&geom, GaussData);

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_F0()
{
  /* Compute the zeroth order forces from the pair potential and pair functional */
  int i, j, j0, k, n, Zi, Zj;
  double Ri[3], Rj[3], dR[3], magdR, Xi;
  double Fij[3], dphi, df, l;

  /* Initialize the forces */
  for (i = 0; i < 3*geom.N; i++)
    simInfo.F0[i] = 0.0;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      simInfo.Flattice0[i][j] = 0.0;

  /* 1. Pair potential forces */
  for (i = 0; i < geom.N; i++)
    {
      Zi = geom.Z[i];
      Ri[0] = geom.R[3*i+0];
      Ri[1] = geom.R[3*i+1];
      Ri[2] = geom.R[3*i+2];
      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
        {
          j = geom.neighbourList[i].neighbour[n];
          if (j != i)
            {
              Zj = geom.Z[j];
              Rj[0] = geom.R[3*j+0];
              Rj[1] = geom.R[3*j+1];
              Rj[2] = geom.R[3*j+2];

              dR[0] = Rj[0] - Ri[0];
              dR[1] = Rj[1] - Ri[1];
              dR[2] = Rj[2] - Ri[2];
              magdR = sqrt (dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2]);

              dphi = tablesdPP (Zi, Zj, magdR)/magdR;
              Fij[0] = dphi*dR[0];
              Fij[1] = dphi*dR[1];
              Fij[2] = dphi*dR[2];

              /* Atomic forces */
              simInfo.F0[3*i+0] += dphi*dR[0];
              simInfo.F0[3*i+1] += dphi*dR[1];
              simInfo.F0[3*i+2] += dphi*dR[2];

              /* Lattice forces */
              j0 = geom.i[j];
              if (j != j0)
                {
                  dR[0] = Rj[0] - geom.R[3*j0+0];
                  dR[1] = Rj[1] - geom.R[3*j0+1];
                  dR[2] = Rj[2] - geom.R[3*j0+2];
                  for (k = 0; k < 3; k++)
                    {
                      l = (geom.ReciprocalVec[k][0]*dR[0] + geom.ReciprocalVec[k][1]*dR[1] + geom.ReciprocalVec[k][2]*dR[2])/(2.0*PI);
                      simInfo.Flattice0[k][0] -= 0.5*l*Fij[0];
                      simInfo.Flattice0[k][1] -= 0.5*l*Fij[1];
                      simInfo.Flattice0[k][2] -= 0.5*l*Fij[2];
                    }
                }
            }
        }
    }

  /* 2. Pair functional forces */
  if (simInfo.model.PairFunctionalFlag == 1)
    {
      for (i = 0; i < geom.N; i++)
        {
          Zi = geom.Z[i];
          Ri[0] = geom.R[3*i+0];
          Ri[1] = geom.R[3*i+1];
          Ri[2] = geom.R[3*i+2];

          /* Embedding prefactor */
          Xi = 0.0;
          for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
            {
              j = geom.neighbourList[i].neighbour[n];
              if (j != i)
                {
                  Zj = geom.Z[j];
                  Rj[0] = geom.R[3*j+0];
                  Rj[1] = geom.R[3*j+1];
                  Rj[2] = geom.R[3*j+2];

                  dR[0] = Rj[0] - Ri[0];
                  dR[1] = Rj[1] - Ri[1];
                  dR[2] = Rj[2] - Ri[2];
                  magdR = sqrt (dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2]);

                  Xi += tablesEP (Zi, Zj, magdR);
                }
            }
          df = tablesdEmbed (&atomsData, Zi, Xi);

          /* Density gradient terms */
          for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
            {
              j = geom.neighbourList[i].neighbour[n];
              j0 = geom.i[j];
              if (j0 != i)
                {
                  Zj = geom.Z[j];
                  Rj[0] = geom.R[3*j+0];
                  Rj[1] = geom.R[3*j+1];
                  Rj[2] = geom.R[3*j+2];

                  dR[0] = Rj[0] - Ri[0];
                  dR[1] = Rj[1] - Ri[1];
                  dR[2] = Rj[2] - Ri[2];
                  magdR = sqrt (dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2]);

                  dphi = df*tablesdEP (Zi, Zj, magdR)/magdR;

                  /* Atomic forces */
                  Fij[0] = dphi*dR[0];
                  Fij[1] = dphi*dR[1];
                  Fij[2] = dphi*dR[2];

                  simInfo.F0[3*i+0] += Fij[0];
                  simInfo.F0[3*i+1] += Fij[1];
                  simInfo.F0[3*i+2] += Fij[2];

                  simInfo.F0[3*j0+0] -= Fij[0];
                  simInfo.F0[3*j0+1] -= Fij[1];
                  simInfo.F0[3*j0+2] -= Fij[2];

                  /* Lattice forces */
                  j0 = geom.i[j];
                  if (j != j0)
                    {
                      dR[0] = Rj[0] - geom.R[3*j0+0];
                      dR[1] = Rj[1] - geom.R[3*j0+1];
                      dR[2] = Rj[2] - geom.R[3*j0+2];

                      for (k = 0; k < 3; k++)
                        {
                          l = (geom.ReciprocalVec[k][0]*dR[0] + geom.ReciprocalVec[k][1]*dR[1] + geom.ReciprocalVec[k][2]*dR[2])/(2.0*PI);
                          simInfo.Flattice0[k][0] -= l*Fij[0];
                          simInfo.Flattice0[k][1] -= l*Fij[1];
                          simInfo.Flattice0[k][2] -= l*Fij[2];
                        }
                    }
                }
            }
        }
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_E1(ws_t *ws)
{
  /* Compute the first order contribution to the energy */
  int i, j, ik, k, n, Ni, Nj, s, Zi;
  double f;

  /* Compute Tr{rho H} */
  simInfo.E1 = sparseTraceRhoMatrices(&geom, stateInfo.spinType, &ws->sparseRho, &ws->sparseM, &ws->sparseH);

  /* Subtract Tr{rho_0 H} */
  for (i = 0; i < geom.N; i++)
    {
      Zi = geom.Z[i];
      Ni = geom.nOrb[i];
      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
        {
          j = geom.neighbourList[i].neighbour[n];
          Nj = geom.nOrb[j];
          if (j == i)
            for (k = 0; k < Ni; k++)
              switch (stateInfo.spinType)
                {
                case 0: /* No spin */
                  simInfo.E1 -= (ws->sparseH.M[0]).v[i][n][k*Ni + k]*atomsData.ad[Zi].f_o[k];
                  break;
                case 1: /* Collinear spin */
                  simInfo.E1 -= 0.5*((ws->sparseH.M[0]).v[i][n][k*Ni + k] + (ws->sparseH.M[1]).v[i][n][k*Ni + k])*atomsData.ad[Zi].f_o[k];
                  break;
                case 2: /* Non-collinear spin */
                  simInfo.E1 -= 0.5*((ws->sparseH.M[0]).v[i][n][2*(k*Ni + k) + 0] + (ws->sparseH.M[1]).v[i][n][2*(k*Ni + k) + 0])*atomsData.ad[Zi].f_o[k];
                  break;
                }
        }
    }

  /* Compute the electron entropy */
  simInfo.Eentropy = 0.0;
  if (stateInfo.nBloch > 0)
    {
      for (s = 0; s < stateInfo.nSets; s++)
        {
          ik = stateInfo.ik[s];
          for (i = 0; i < eigenStates[s].nStates; i++)
            {
              f = stateInfo.occ[s][i];
              if ((f > 1.0e-10) && (1.0-f > 1.0e-10))
                simInfo.Eentropy -= stateInfo.spinDegeneracy*stateInfo.wtk[ik]*simInfo.kT*(f*log(f) + (1.0-f)*log(1.0-f));
            }
        }
    }
  else
    {
      for (i = 0; i < eigenStates[0].nStates; i++)
        {
          f = stateInfo.occ[0][i];
          if ((f > 1.0e-10) && (1.0-f > 1.0e-10))
            simInfo.Eentropy -= stateInfo.spinDegeneracy*simInfo.kT*(f*log(f) + (1.0-f)*log(1.0-f));
        }
    }

  /* Compute the contribution to the energy from the spin constraint fields */
  simInfo.Econstraint = magneticConstraintEnergy(stateInfo.spinType, &geom, &multipole);

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_F1 (ws_t *ws)
{
  /* Compute the first order forces */
  int c, ci, cj, cij, cji, x, i, j, j0, k, ki, kj, kij, cjk, n, m, p, p0, Ni, Nj, Nk, NNi, Zi;
  double *dS, *dH;
  double peierlsPhi, exp_peierlsPhi_r, exp_peierlsPhi_i;
  double R_I[3], R_J[3], A_I[3], A_J[3], Fij[3], dR[3];
  double a, b, cc, d, e, f, g, h, l;

  /* Initialize the forces */
  for (i = 0; i < 3*geom.N; i++)
    simInfo.F1[i] = 0.0;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      simInfo.Flattice1[i][j] = 0.0;

  /* Add the forces from the first order energy */
  for (i = 0; i < geom.N; i++)
    {
      Ni = geom.nOrb[i];
      Zi = geom.Z[i];
      NNi = geom.neighbourList[i].nNeighbour;
      for (n = 0; n < NNi; n++)
        {
          j = geom.neighbourList[i].neighbour[n];
          j0 = geom.i[j];
          Nj = geom.nOrb[j];

          if (i == j)
            {
              if (simInfo.model.XtalFieldFlag == 1)
                {
                  /* Crystal Field contribution */
                  /* Allocate space for the gradients of one block of the Hamiltonian for each neighbour */
                  dH = malloc (3*Ni*Nj*NNi*sizeof(double));
                  dS = NULL;

                  /* Build the gradients of one block of the Hamiltonian and overlap matrices */
                  tablesGradientBlock (&geom, &atomsData, &simInfo.model, i, j, dS, dH);

                  /* Update the forces */
                  for (m = 0; m < NNi; m++)
                    {
                      p = geom.neighbourList[i].neighbour[m];
                      p0 = geom.i[p];
                      for (k = 0; k < 3; k++)
                        {
                          Fij[k] = 0.0;
                          for (c = 0; c < Ni*Nj; c++)
                            Fij[k] -= ws->sparseRho.v[i][n][ws->sparseRho.dataSize*c]*dH[(3*m + k)*Ni*Nj + c];
                          for (c = 0; c < Ni; c++)
                            Fij[k] += atomsData.ad[Zi].f_o[c]*dH[(3*m + k)*Ni*Nj + c*(Ni+1)];
                          simInfo.F1[3*i  + k] += Fij[k];
                          simInfo.F1[3*p0 + k] -= Fij[k];
                        }

                      /* Lattice forces */
                      dR[0] = geom.R[3*p+0] - geom.R[3*p0+0];
                      dR[1] = geom.R[3*p+1] - geom.R[3*p0+1];
                      dR[2] = geom.R[3*p+2] - geom.R[3*p0+2];
                      for (k = 0; k < 3; k++)
                        {
                          l = (geom.ReciprocalVec[k][0]*dR[0] + geom.ReciprocalVec[k][1]*dR[1] + geom.ReciprocalVec[k][2]*dR[2])/(2.0*PI);
                          simInfo.Flattice1[k][0] -= l*Fij[0];
                          simInfo.Flattice1[k][1] -= l*Fij[1];
                          simInfo.Flattice1[k][2] -= l*Fij[2];
                        }
                    }

                  /* Free the memory of one block of the Hamiltonian */
                  free (dH);
                }
            }
          else
            {
              /* Allocate space for the gradients of one block of the Hamiltonian and overlap matrices */
              dH = malloc (3*Ni*Nj*sizeof(double));
              if (simInfo.model.OverlapFlag > 0)
                dS = malloc (3*Ni*Nj*sizeof(double));
              else
                dS = NULL;

              /* Build the gradients of one block of the Hamiltonian and overlap matrices */
              tablesGradientBlock (&geom, &atomsData, &simInfo.model, i, j, dS, dH);

              /* Update the forces */
              for (k = 0; k < 3; k++)
                Fij[k] = 0.0;
              if (simInfo.peierlsFlag == 0)
                {
                  for (ci = 0; ci < Ni; ci++)
                    for (cj = 0; cj < Nj; cj++)
                      {
                        cij = ci*Nj + cj;
                        cji = cj*Ni + ci;
                        for (k = 0; k < 3; k++)
                          Fij[k] -= ws->sparseRho.v[i][n][ws->sparseRho.dataSize*cij]*dH[k*Ni*Nj + cij];
                      }
                }
              else
                {
                  /* Set the Peierls factor, which is only non-zero for off-diagonal blocks */
                  peierlsPhi = magneticPeierlsPhase(&geom, &simInfo, i, j);
                  exp_peierlsPhi_r = cos(peierlsPhi);
                  exp_peierlsPhi_i = sin(peierlsPhi);
                  a = exp_peierlsPhi_r;
                  b = exp_peierlsPhi_i;

                  for (k = 0; k < 3; k++)
                    {
                      R_I[k] = geom.R[3*i+k];
                      R_J[k] = geom.R[3*j+k];
                    }
                  magneticVectorPotential(&simInfo, A_I, R_I);
                  magneticVectorPotential(&simInfo, A_J, R_J);

                  for (k = 0; k < 3; k++)
                    {
                      // Set the parts of the expression that are the same for all sigma, sigma'
                      e =  0;
                      f =  (A_I[k] - A_J[k]);     // (q/ihbar) = i

                      for (ci = 0; ci < Ni; ci++)
                        for (cj = 0; cj < Nj; cj++)
                          {
                            cij = ci*Nj + cj;
                            cji = cj*Ni + ci;

                            // I use that dH is real here
                            simInfo.F1[3*i  + k] -= (a*ws->sparseRho.v[i][n][2*cij + 0] - b*ws->sparseRho.v[i][n][2*cij + 1])*dH[k*Ni*Nj + cij];
                            simInfo.F1[3*j0 + k] += (a*ws->sparseRho.v[i][n][2*cij + 0] - b*ws->sparseRho.v[i][n][2*cij + 1])*dH[k*Ni*Nj + cij];

                            // Up-Up ( rho_uu = 0.5(rho_0 + rho_z) )
                            cc = 0.5*(ws->sparseRho.v[i][n][2*cij+0] + (ws->sparseM.M[0]).v[i][n][2*cij+0]);
                            d  = 0.5*(ws->sparseRho.v[i][n][2*cij+1] + (ws->sparseM.M[0]).v[i][n][2*cij+1]);
                            g = (ws->sparseFock.M[0]).v[i][n][2*cij+0];
                            h = (ws->sparseFock.M[0]).v[i][n][2*cij+1];
                            simInfo.F1[3*i  + k] -= 2*(a*cc*e*g - a*d*f*g - b*cc*f*g - b*d*e*g - a*cc*f*h - a*d*e*h - b*cc*e*h + b*d*f*h);

                            // Down-Down ( rho_dd = 0.5(rho_0 - rho_z) )
                            cc = 0.5*(ws->sparseRho.v[i][n][2*cij+0] - (ws->sparseM.M[0]).v[i][n][2*cij+0]);
                            d  = 0.5*(ws->sparseRho.v[i][n][2*cij+1] - (ws->sparseM.M[0]).v[i][n][2*cij+1]);
                            g = (ws->sparseFock.M[1]).v[i][n][2*cij+0];
                            h = (ws->sparseFock.M[1]).v[i][n][2*cij+1];
                            simInfo.F1[3*i  + k] -= 2*(a*cc*e*g - a*d*f*g - b*cc*f*g - b*d*e*g - a*cc*f*h - a*d*e*h - b*cc*e*h + b*d*f*h);

                            // Up-Down ( rho_ud = 0.5(rho_x - i rho_y) )
                            cc = 0.5*((ws->sparseM.M[1]).v[i][n][2*cij+0] + (ws->sparseM.M[2]).v[i][n][2*cij+1]);
                            d  = 0.5*((ws->sparseM.M[1]).v[i][n][2*cij+1] - (ws->sparseM.M[2]).v[i][n][2*cij+0]);
                            g = (ws->sparseFock.M[2]).v[i][n][2*cij+0];
                            h = (ws->sparseFock.M[2]).v[i][n][2*cij+1];
                            simInfo.F1[3*i  + k] -= 2*(a*cc*e*g - a*d*f*g - b*cc*f*g - b*d*e*g - a*cc*f*h - a*d*e*h - b*cc*e*h + b*d*f*h);

                            // Down-Up ( rho_du = 0.5(rho_x + i rho_y) )
                            cc = 0.5*((ws->sparseM.M[1]).v[i][n][2*cij+0] - (ws->sparseM.M[2]).v[i][n][2*cij+1]);
                            d  = 0.5*((ws->sparseM.M[1]).v[i][n][2*cij+1] + (ws->sparseM.M[2]).v[i][n][2*cij+0]);
                            g =  (ws->sparseFock.M[2]).v[i][n][2*cji+0];
                            h = -(ws->sparseFock.M[2]).v[i][n][2*cji+1];
                            simInfo.F1[3*i  + k] -= 2*(a*cc*e*g - a*d*f*g - b*cc*f*g - b*d*e*g - a*cc*f*h - a*d*e*h - b*cc*e*h + b*d*f*h);
                          }
                    }
                }

              if (simInfo.model.OverlapFlag > 0)
                for (k = 0; k < 3; k++)
                  for (c = 0; c < Ni*Nj; c++)
                    Fij[k] += ws->sparseEmatrix.v[i][n][ws->sparseEmatrix.dataSize*c]*dS[k*Ni*Nj + c];

              for (k = 0; k < 3; k++)
                {
                  simInfo.F1[3*i  + k] += Fij[k];
                  simInfo.F1[3*j0 + k] -= Fij[k];
                }

              /* Lattice forces */
              dR[0] = geom.R[3*j+0] - geom.R[3*j0+0];
              dR[1] = geom.R[3*j+1] - geom.R[3*j0+1];
              dR[2] = geom.R[3*j+2] - geom.R[3*j0+2];
              for (k = 0; k < 3; k++)
                {
                  l = (geom.ReciprocalVec[k][0]*dR[0] + geom.ReciprocalVec[k][1]*dR[1] + geom.ReciprocalVec[k][2]*dR[2])/(2.0*PI);
                  simInfo.Flattice1[k][0] -= l*Fij[0];
                  simInfo.Flattice1[k][1] -= l*Fij[1];
                  simInfo.Flattice1[k][2] -= l*Fij[2];
                }

              /* Free the memory of one block of the Hamiltonian and overlap matrices */
              if (simInfo.peierlsFlag == 0)
                free (dH);
              if (simInfo.model.OverlapFlag > 0)
                free (dS);
            }
        }
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_E2()
{
  /* Compute the second order contribution to the energy */
  simInfo.E2 = 0.0;

  if (simInfo.model.SCFFlag > 0)
    {
      /* Add the electrostatic contribution */
//##################################
      if (TB_IsMonopole3D())
        simInfo.E2 += electroMonopoleEnergy3D (&geom, &multipole);
      else
        simInfo.E2 += electroEnergy (&geom, &multipole);
//##################################

      /*  Add the magnetic contribution */
      if (stateInfo.spinType > 0 && simInfo.stonerFlag)
        simInfo.E2 += magneticEnergy (stateInfo.spinType, &geom, &multipole);
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_F2(ws_t *ws)
{
  /* Compute the second order forces */
  int i, j, p, n, k;
  double *Fmag;

  /* Initialise the forces */
  for (i = 0; i < 3*geom.N; i++)
    simInfo.F2[i] = 0.0;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      simInfo.Flattice2[i][j] = 0.0;

  /* Evaluate the forces if charge self-consistency used */
  if (simInfo.model.SCFFlag > 0)
    {
      /* Allocate space for the gradients of the multipole integrals. */
      multipole.dQint = malloc (3*sizeof(sparseMatrix_t *));
      for (k = 0; k < 3; k++)
        {
          multipole.dQint[k] = malloc (multipole.sizeQ*sizeof(sparseMatrix_t));
          for (p = 0; p < multipole.sizeQ; p++)
            sparseCreateMatrix (&geom, &multipole.dQint[k][p], 1);
        }

      /* Evaluate the gradients of the multipole integrals */
      if (simInfo.model.TightBindingFlag == 3)
        statesDerivativeMultipoleIntegrals(&geom, &GaussData, &multipole);
      else
        tablesDerivativeMultipoleIntegrals(&geom, &atomsData, &multipole);

      /* Allocate space for the gradients of the multipoles */
      multipole.dQ_out = malloc (geom.N*sizeof(double ***));
      for (i = 0; i < geom.N; i++)
        {
          multipole.dQ_out[i] = malloc (multipole.sizeQ*sizeof(double **));
          for (p = 0; p < multipole.sizeQ; p++)
            {
              multipole.dQ_out[i][p] = malloc (geom.neighbourList[i].nNeighbour*sizeof(double *));
              for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
                multipole.dQ_out[i][p][n] = malloc (3*sizeof(double));
            }
        }

      /* Evaluate the gradients of the multipoles */
      statesDerivativeMultipoleMoments (&geom, &multipole, &ws->sparseRho);

      /* Evaluate the electrostatic forces */
//##################################
      if (TB_IsMonopole3D())
        electroMonopoleForce3D (&geom, &multipole, simInfo.F2, simInfo.Flattice2);
      else
        electroForce (&geom, &multipole, simInfo.F2, simInfo.Flattice2);
//##################################

      /* Free the space for the gradients of the multipoles. */
      for (i = 0; i < geom.N; i++)
        {
          for (p = 0; p < multipole.sizeQ; p++)
            {
              for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
                free (multipole.dQ_out[i][p][n]);
              free (multipole.dQ_out[i][p]);
            }
          free (multipole.dQ_out[i]);
        }
      free (multipole.dQ_out);

      /* Magnetic forces */
      if (stateInfo.spinType > 0)
        {
          /* Allocate space for the gradients of the magnetic multipoles */
          switch (stateInfo.spinType)
            {
            case 0:
              break;
            case 1:
              multipole.dMz_out = malloc (geom.N*sizeof(double ***));
              for (i = 0; i < geom.N; i++)
                {
                  multipole.dMz_out[i] = malloc (multipole.sizeQ*sizeof(double **));
                  for (p = 0; p < multipole.sizeQ; p++)
                    {
                      multipole.dMz_out[i][p] = malloc (geom.neighbourList[i].nNeighbour*sizeof(double *));
                      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
                        multipole.dMz_out[i][p][n] = malloc (3*sizeof(double));
                    }
                }
              break;
            case 2:
              multipole.dMx_out = malloc (geom.N*sizeof(double ***));
              multipole.dMy_out = malloc (geom.N*sizeof(double ***));
              multipole.dMz_out = malloc (geom.N*sizeof(double ***));
              for (i = 0; i < geom.N; i++)
                {
                  multipole.dMx_out[i] = malloc (multipole.sizeQ*sizeof(double **));
                  multipole.dMy_out[i] = malloc (multipole.sizeQ*sizeof(double **));
                  multipole.dMz_out[i] = malloc (multipole.sizeQ*sizeof(double **));
                  for (p = 0; p < multipole.sizeQ; p++)
                    {
                      multipole.dMx_out[i][p] = malloc (geom.neighbourList[i].nNeighbour*sizeof(double *));
                      multipole.dMy_out[i][p] = malloc (geom.neighbourList[i].nNeighbour*sizeof(double *));
                      multipole.dMz_out[i][p] = malloc (geom.neighbourList[i].nNeighbour*sizeof(double *));
                      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
                        {
                          multipole.dMx_out[i][p][n] = malloc (3*sizeof(double));
                          multipole.dMy_out[i][p][n] = malloc (3*sizeof(double));
                          multipole.dMz_out[i][p][n] = malloc (3*sizeof(double));
                        }
                    }
                }
              break;
            }

          /* Evaluate the gradients of the magnetic multipoles */
          statesDerivativeMagneticMoments(stateInfo.spinType, &geom, &multipole, &ws->sparseM);

          /* Evaluate the magnetic forces */
          Fmag = malloc (3*geom.N*sizeof(double));
          magneticForce (stateInfo.spinType, &geom, &multipole, Fmag);
          for (i = 0; i < 3*geom.N; i++)
            simInfo.F2[i] += Fmag[i];
          free (Fmag);

          /* Free the space for the gradients of the magnetic multipoles. */
          switch (stateInfo.spinType)
            {
            case 0:
              break;
            case 1:
              for (i = 0; i < geom.N; i++)
                {
                  for (p = 0; p < multipole.sizeQ; p++)
                    {
                      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++)
                        free (multipole.dMz_out[i][p][n]);
                      free (multipole.dMz_out[i][p]);
                    }
                  free (multipole.dMz_out[i]);
                }
              free (multipole.dMz_out);
              break;
            case 2:
              for (i = 0; i < geom.N; i++)
                {
                  for (p = 0; p < multipole.sizeQ; p++)
                    {
                      for (n = 0; n < geom.neighbourList[i].nNeighbour; n++){
                          free (multipole.dMx_out[i][p][n]);
                          free (multipole.dMy_out[i][p][n]);
                          free (multipole.dMz_out[i][p][n]);
                      }
                      free (multipole.dMx_out[i][p]);
                      free (multipole.dMy_out[i][p]);
                      free (multipole.dMz_out[i][p]);
                    }
                  free (multipole.dMx_out[i]);
                  free (multipole.dMy_out[i]);
                  free (multipole.dMz_out[i]);
                }
              free (multipole.dMx_out);
              free (multipole.dMy_out);
              free (multipole.dMz_out);
              break;
            }
        }

      /* Free the space for the gradients of the multipole integrals. */
      for (k = 0; k < 3; k++)
        {
          for (p = 0; p < multipole.sizeQ; p++)
            free_sparseMatrix (&geom, &multipole.dQint[k][p]);
          free (multipole.dQint[k]);
        }
      free (multipole.dQint);
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_TestOverlap (eigenStates_t *eigenStates)
{
  /* Find lowest eigenvalue of overlap matrix.
   * This is a way to test if their is linear dependency in the basis set */
  eigenStates_t overlapEigenstates;

  /* Set up the data structure for diagonalising the overlap matrix  */
  overlapEigenstates.testFlag = 0;
  overlapEigenstates.isComplexE = eigenStates->isComplexE;
  overlapEigenstates.isComplexV = eigenStates->isComplexV;
  overlapEigenstates.matrixType = eigenStates->matrixType;
  overlapEigenstates.matrixFormat = eigenStates->matrixFormat;
  overlapEigenstates.problemType = 0;
  overlapEigenstates.nElement = eigenStates->nElement;
  overlapEigenstates.nStates = eigenStates->nStates;
  overlapEigenstates.nElement = eigenStates->nElement;
  overlapEigenstates.H = eigenStates->S;
  overlapEigenstates.S = NULL;
  overlapEigenstates.e = malloc (overlapEigenstates.nStates*sizeof(double));
  if (overlapEigenstates.isComplexV == 1)
    overlapEigenstates.v = malloc (2*overlapEigenstates.nStates*overlapEigenstates.nElement*sizeof(double));
  else
    overlapEigenstates.v = malloc (overlapEigenstates.nStates*overlapEigenstates.nElement*sizeof(double));

  /* Diagonalise the overlap matrix */
  if (eigen (&overlapEigenstates) > 0)
    {
      printf ("Unable to find eigenstates of the overlap matrix.\n");
      TB_Stop (simInfo.outfp, 1, &geom, &simInfo, &stateInfo, &multipole);
    }

  /* Test the eigenspectrum of the overlap matrix */
  printf ("The smallest eigenvalue of the overlap matrix is %f\n", overlapEigenstates.e[0]);

  /* Free temporary storage */
  free(overlapEigenstates.e);
  free(overlapEigenstates.v);

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_WriteResults ()
{
  /* Write out summary of calculation */
  int i, j, k, l, m, n, Zi, s, q;
  int ki, kj, Ni, Nj;
  double xi, yi, zi;
  double d0, d1, d2;
  double Fext_x, Fext_y, Fext_z, Ftb_x, Ftb_y, Ftb_z;
  char FileName[BUF_LEN];
  FILE *wffp;

  /* Write out geometric information */
  TB_WriteGeometry (&geom, &simInfo, &atomsData);

  /* Write out energy information */
  TB_WriteEnergy (&geom, &simInfo);

  /* Charge and spin multipoles */
  if (simInfo.model.TightBindingFlag > 0)
    TB_WriteMultipoles (&geom, &simInfo, &stateInfo, &multipole);

  /* Diagnostic information */
  if (simInfo.verbosity > 0)
    {
      if (simInfo.externalModelFlag == 0)
        {
          /* Forces */
          fprintf (simInfo.outfp, "\nForces (Ry/a0)\n");
          fprintf (simInfo.outfp, "==============\n");
          fprintf (simInfo.outfp, "index   Fx (TB)    Fy (TB)    Fz (TB)\n");
          for (i = 0; i < geom.N; i++)
            {
              Ftb_x  = simInfo.F[3*i + 0];
              Ftb_y  = simInfo.F[3*i + 1];
              Ftb_z  = simInfo.F[3*i + 2];
              fprintf (simInfo.outfp, "%4i %10.3f %10.3f %10.3f\n", i, Ftb_x, Ftb_y, Ftb_z);
            }
          fflush (simInfo.outfp);
        }
      else
        {
          /* Forces */
          fprintf (simInfo.outfp, "\nForces (Ry/a0)\n");
          fprintf (simInfo.outfp, "==============\n");
          fprintf (simInfo.outfp, "index   Fx (TB)    Fy (TB)    Fz (TB)   Fx (ext)   Fy (ext)   Fz (ext)\n");
          for (i = 0; i < geom.N; i++)
            {
              Fext_x = simInfo.Fexternal[3*i + 0];
              Fext_y = simInfo.Fexternal[3*i + 1];
              Fext_z = simInfo.Fexternal[3*i + 2];
              Ftb_x  = simInfo.F[3*i + 0] - Fext_x;
              Ftb_y  = simInfo.F[3*i + 1] - Fext_y;
              Ftb_z  = simInfo.F[3*i + 2] - Fext_z;
              fprintf (simInfo.outfp, "%4i %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", i, Ftb_x, Ftb_y, Ftb_z, Fext_x, Fext_y, Fext_z);
            }
          fflush (simInfo.outfp);
        }
    }

  if (simInfo.model.TightBindingFlag > 0)
    if (simInfo.verbosity > 5)
      {
        /* Hamiltonian and overlap matrices */
        fprintf (simInfo.outfp, "\nHamiltonian (Ry) and overlap\n");
        fprintf (simInfo.outfp, "============================\n");
        for (s = 0; s < stateInfo.nSets; s++)
          {
            fprintf (simInfo.outfp, "\nState: %4i\n", s+1);
            fprintf (simInfo.outfp, "-----------\n");
            for (i = 0; i < geom.N; i++)
              {
                Ni = atomsData.ad[geom.Z[i]].nOrb;
                for (j = 0; j < geom.N; j++)
                  {
                    Nj = atomsData.ad[geom.Z[j]].nOrb;
                    fprintf (simInfo.outfp, "\natom 1: %4i   atom 2: %4i\n\n", i+1, j+1);
                    for (ki = 0; ki < Ni; ki++)
                      for (kj = 0; kj < Nj; kj++)
                        {
                          k = (geom.Hindex[i]+ki)*eigenStates[s].nElement + (geom.Hindex[j]+kj);
                          switch (eigenStates[s].matrixType)
                            {
                            case 0:
                            case 3:
                              fprintf (simInfo.outfp, "[%2i, %2i] %14.7g  %14.7g\n", ki, kj, eigenStates[s].H[k], eigenStates[s].S[k]);
                              break;
                            case 1:
                            case 2:
                            case 4:
                              fprintf (simInfo.outfp, "[%2i, %2i] %14.7g + i %14.7g     %14.7g + i %14.7g\n", ki, kj, eigenStates[s].H[2*k + 0], eigenStates[s].H[2*k + 1], eigenStates[s].S[2*k + 0], eigenStates[s].S[2*k + 1]);
                              break;
                            }
                        }
                  }
              }
          }
        fflush (simInfo.outfp);
      }

  if (simInfo.model.TightBindingFlag > 0)
    {
      /* Write eigenvalues and occupancies to file */
      TB_WriteEigenvalues (&geom, &simInfo, &stateInfo, eigenStates);

      /* Write wavefunctions to file */
      if (simInfo.writeWavefunctionFlag == 1)
        {
          /* Open the wavefunction file for write */
          strcpy (FileName, simInfo.JobName);
          strcat (FileName, ".wf");
          if ((wffp = fopen (FileName, "w")) == NULL)
            {
              printf ("Unable to open the wavefunction file %s\n", FileName);
              return 0;
            }
          statesWriteEigenstates (&atomsData, &stateInfo, &geom, &eigenStates, wffp);
          fclose (wffp);
        }
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_UnitTest ()
{
  /* This routine checks that the finite differences of the energy agrees with the forces */
  int i, j, choice, fret;
  double dx, E0p, E0m, E1p, E1m, E2p, E2m, Etp, Etm;
  double Delta0, Delta1, Delta2, Delta3, tolerance;
  double *F0, *F1, *F2, *F, G_a[3], G_n[3];
  double Flattice0[3][3], Flattice1[3][3], Flattice2[3][3], Flattice[3][3];

  /* Set the atomic displacement */
  dx = 0.001;

  /* Prompt the user for the test required */
  printf ("Tests available:\n");
  printf ("  0. Exit\n");
  printf ("  1. Atomic forces\n");
  printf ("  2. Lattice forces\n");
  printf ("\nEnter your choice > ");
  fflush(stdout);
  fret = scanf("%i", &choice);

  /* Perform the requested test */
  switch (choice)
    {
    case 0: /* Exit */
      return 0;
      break;
    case 1: /* Atomic forces */
      /* Allocate memory for the analytic forces */
      F0 = malloc (3*geom.N*sizeof(double));
      F1 = malloc (3*geom.N*sizeof(double));
      F2 = malloc (3*geom.N*sizeof(double));
      F  = malloc (3*geom.N*sizeof(double));

      /* Compute the analytic forces */
      printf ("Computing analytic forces ...\n");
      TB_EnergyForce ();
      for (i = 0; i < 3*geom.N; i++)
        {
          F0[i] = simInfo.F0[i];
          F1[i] = simInfo.F1[i];
          F2[i] = simInfo.F2[i];
          F[i]  = simInfo.F[i];
        }

      /* Compute the finite differences of the energy and compare with the forces */
      for (i = 0; i < 3*geom.N; i++)
        {
          printf ("Computing E[+dx] ...\n");
          geom.R[i] += dx;
          TB_EnergyForce ();
          E0p = simInfo.E0;
//        E1p = simInfo.E1 - simInfo.Eentropy;
          E1p = simInfo.E1;
          E2p = simInfo.E2;
          Etp = simInfo.Etotal;
          geom.R[i] -= dx;

          printf ("Computing E[-dx] ...\n");
          geom.R[i] -= dx;
          TB_EnergyForce ();
          E0m = simInfo.E0;
//        E1m = simInfo.E1 - simInfo.Eentropy;
          E1m = simInfo.E1;
          E2m = simInfo.E2;
          Etm = simInfo.Etotal;
          geom.R[i] += dx;

          printf ("\nAnalytic forces   : F0 = %9.5f F1 = %9.5f F2 = %9.5f F  = %9.5f\n", F0[i], F1[i], F2[i], F[i]);
          printf ("Finite differences:      %9.5f      %9.5f      %9.5f      %9.5f\n", -(E0p - E0m)/(dx+dx), -(E1p - E1m)/(dx+dx), -(E2p - E2m)/(dx+dx), -(Etp - Etm)/(dx+dx));
          Delta0 = F0[i]+(E0p - E0m)/(dx+dx), Delta1 = F1[i]+(E1p - E1m)/(dx+dx), Delta2 = F2[i]+(E2p - E2m)/(dx+dx), Delta3 = F[i]+(Etp - Etm)/(dx+dx);
          printf ("Delta:                   %9.5f      %9.5f      %9.5f      %9.5f\n", Delta0, Delta1, Delta2, Delta3);
        }

      /* Free temporary memory */
      free (F0);
      free (F1);
      free (F2);
      break;
    case 2: /* Lattice forces */
      /* Compute the analytic forces */
      TB_EnergyForce ();
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          {
            Flattice0[i][j] = simInfo.Flattice0[i][j];
            Flattice1[i][j] = simInfo.Flattice1[i][j];
            Flattice2[i][j] = simInfo.Flattice2[i][j];
            Flattice[i][j] = simInfo.Flattice[i][j];
          }

      /* Compute the finite differences of the energy and compare with the forces */
      for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
          {
            TB_AdjustCell (i, j, dx);
            TB_EnergyForce ();
            E0p = simInfo.E0;
            E1p = simInfo.E1 - simInfo.Eentropy;
            E2p = simInfo.E2;
            Etp = simInfo.Etotal;
            TB_AdjustCell (i, j, -dx);

            TB_AdjustCell (i, j, -dx);
            TB_EnergyForce ();
            E0m = simInfo.E0;
            E1m = simInfo.E1 - simInfo.Eentropy;
            E2m = simInfo.E2;
            Etm = simInfo.Etotal;
            TB_AdjustCell (i, j, dx);

            printf ("\nAnalytic forces   : Flattice0 = %9.5f Flattice1 = %9.5f Flattice2 = %9.5f Flattice  = %9.5f\n", Flattice0[i][j], Flattice1[i][j], Flattice2[i][j], Flattice[i][j]);
            printf ("Finite differences:             %9.5f             %9.5f             %9.5f             %9.5f\n", -(E0p - E0m)/(dx+dx), -(E1p - E1m)/(dx+dx), -(E2p - E2m)/(dx+dx), -(Etp - Etm)/(dx+dx));
            Delta0 = Flattice0[i][j] + (E0p - E0m)/(dx+dx);
            Delta1 = Flattice1[i][j] + (E1p - E1m)/(dx+dx);
            Delta2 = Flattice2[i][j] + (E2p - E2m)/(dx+dx);
            Delta3 = Flattice[i][j]  + (Etp - Etm)/(dx+dx);
            printf ("Delta:                          %9.5f             %9.5f             %9.5f             %9.5f\n", Delta0, Delta1, Delta2, Delta3);
          }
      break;
    default:
      printf ("Unknown test %i\n", choice);
      break;
    }

  return 0;
}

/*-------------------------------------------------------------------------------------------------------*/

int
TB_AdjustCell (int i, int j, double dx)
{
  int n;

  /* Put Bloch vectors into reduced form */
  InverseTransformK (stateInfo.nBloch, stateInfo.k, geom.CellVec);

  /* Transform cell vectors */
  geom.CellVec[i][j] += dx;

  /* Correct cell vector lengths */
  for (n = 0; n < 3; n++)
    geom.CellSize[n] = sqrt(geom.CellVec[n][0]*geom.CellVec[n][0] + geom.CellVec[n][1]*geom.CellVec[n][1] + geom.CellVec[n][2]*geom.CellVec[n][2]);

  /* Get reciprocal lattice vectors */
  GetReciprocal (geom.CellVec, geom.ReciprocalVec, &geom.Volume);

  /* Transform the k points from reciprocal lattice vector fractions to cartesian coordinates */
  TransformK (stateInfo.nBloch, stateInfo.k, geom.ReciprocalVec);

  return 0;
}
