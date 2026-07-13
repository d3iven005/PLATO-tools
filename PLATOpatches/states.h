#ifndef PLATO_STATES_LIB_H
#define PLATO_STATES_LIB_H

#include "../constants.h"
#include "../misc/misc.h"
#include "../sparse/sparse.h"
#include "../eigen/eigen.h"
#include "../gauss/gauss.h"
#include "../angular/angular.h"

/* Parameters */
#define STATES_MAX_E1_XTAL_MIXED_METHODS 5
#define BUF_LEN 1024
#define MTYPES 150

struct model_Struc
{
  /* Location of integral files */
  char *DataPath;

  /* Model definition */
  int PairFunctionalFlag;
  int OverlapFlag;
  int XtalFieldFlag;
  int XtalFieldCorrectionFlag;
  int ThreeCenterFlag;
  int SCFFlag;
  int writeOrbitalProperties;
  int maxMultipoleOrder;
  int TightBindingFlag;

  /* Output variables for diagnostic information */
  int verbosity;
  FILE *outfp;
};
typedef struct model_Struc model_t;

struct tableSpline_Struc
{
  int np;
  double *x, *y, *y2;
};
typedef struct tableSpline_Struc tableSpline_t;

struct multiSpline_Struc
{
  int nspline;
  tableSpline_t *spline;
};
typedef struct multiSpline_Struc multiSpline_t;

struct twoCentreData_Struc
{
  /* Hopping, overlap and crystal field.
   * The first two indices are the orbital shells.
   * The index for the spline is the axial angular momentum (sigma, pi, delta, ...) */
  multiSpline_t **h;                       /* Hopping integral tables */
  multiSpline_t **s;                       /* Overlap integral tables */
  multiSpline_t **xf;                      /* Crystal field integral tables */
  multiSpline_t **sn;                      /* Sankey-Niklewski integrals for many-body xtal field corrections */
  multiSpline_t **xfXcContrib;             /* Exchange-correlation part of the crystal field integrals */
  multiSpline_t **hopXcContrib;            /* Exchange-correlation part of the hopping integrals */
  multiSpline_t **hopNonLocXcCorrsOneBody; /* 1-body Non-local exchange-correlation contribution to hopping integral */
  multiSpline_t **hopNonLocXcCorrsTwoBody; /* 2-body (includes 1-body) Non-local exchange-correlation contribution to hopping integral */
  multiSpline_t **xtalFieldNonLocXc;       /* 2-body Non-local exchange-corelation xtal integral */

  /* Multipole moments.
   * The first index is the i orbital shell.
   * The second index is the multipole moment l.
   * The third index is the expansion state angular momentum.
   * The fourth index is the j orbital shell.
   * The index for the spline is the axial angular momentum (sigma, pi, delta, ...) */
  multiSpline_t ****mp;  /* Multipole integral tables */

  /* Potentials */
  tableSpline_t pp;                  /* Pair potential table */
  tableSpline_t pairPotNonLocXcCorr; /* Non-local Xc contribution to Exc */
  tableSpline_t ep;                  /* Embedding term pair interaction table */
  tableSpline_t nij;                 /* Density overlaps used to approximate many-body exchange-correlation effects */
};
typedef struct twoCentreData_Struc twoCentreData_t;

struct bdtTables_Struc
{
  int nShellsA;
  int nShellsB;
  struct bdtTablesSet_Struc *kineticIntegrals;
  struct bdtTablesSet_Struc *overlapIntegrals;
  struct bdtTablesSet_Struc *hopIntegrals;
  struct bdtTablesSet_Struc *hopXcIntegrals;
  struct bdtTablesSet_Struc *hopNonLocXcCorrsOneBody;
  struct bdtTablesSet_Struc *hopNonLocXcCorrsTwoBody;
  struct bdtTablesSet_Struc *pairPot;
  struct bdtTablesSet_Struc *pairPotNonLocXcCorr;
  struct bdtTablesSet_Struc *pairFunct;
  struct bdtTablesSet_Struc *snIntegrals;
  struct bdtTablesSet_Struc *nijIntegrals;
  struct bdtTablesSet_Struc *xtalFieldTotal;
  struct bdtTablesSet_Struc *xtalFieldNoXc;
  struct bdtTablesSet_Struc *xtalFieldNonLocXc;
  struct bdtTablesSet_Struc *nonLocPP;

  //These tables hold corrections to other ints which may or may not be applied.
  //They are NOT produced by plato
  struct bdtTablesSet_Struc *pairPotCorr;
  struct bdtTablesSet_Struc *hopIntCorr;
};
typedef struct bdtTables_Struc bdtTables_t;

struct bdtTablesSet_Struc
{
  /* Purpose of this struct is to hold ALL information on a single type of integral (e.g. hopping). */
  int intTypeFlag;    /* 0 for “atom”, 1 for “orbital”, possibly 2 for pseudopot  */
  int nTables;
  char intLabel[100]; /* Label, currently used to determine identity of header on output bdt file */
  int splinesApplied; /* 0 = splines not applied, 1 = splines applied */
  struct singleBdtTableInfo_Struc *tables;
};
typedef struct bdtTablesSet_Struc bdtTablesSet_t;

struct singleBdtTableInfo_Struc
{
  /* Purpose of this is to hold one integral table and all relevant information */
  int shellIdxA;
  int shellIdxB;
  int angMomA;
  int angMomB;
  int axAngMom; /* 0=sigma, 1=pi, 2=delta. */
  struct tableSpline_Struc intTable;
};
typedef struct singleBdtTableInfo_Struc singleBdtTableInfo_t;

struct atomData_Struc
{
  /* General */
  char name[3];
  double ZCore;
  double rCut;
  double U;
  double I;
  double excNonLocCorr;

  /* Orbital shells */
  int nShell;
  int *l_s;
  double *soc_s;

  /* Orbitals */
  int nOrb;
  int *n_o;
  int *l_o;
  int *m_o;
  double *e_o;
  double *f_o;
  double *r_o;

  /* Onsite integrals */
  double **h; /* Hamiltonian */
  double **s; /* Overlap */
  double **hvxc; /* Contribution of vxc to Hamiltonian */
  double **vxcNonLocCorr; /* Contribution of non-local xc (e.g. gradient corrections) to vxc */

  /* Energy of the atom */
  double atomEnergy;

  /* Embedding term */
  tableSpline_t embed;
};
typedef struct atomData_Struc atomData_t;

struct atomsData_Struc
{
  int nTypes;
  atomData_t *ad;
};
typedef struct atomsData_Struc atomsData_t;

struct stateInfo_s
{
  /* Bloch state related information */
  int nBloch;              /* 0 ==> not using Bloch states, >0 ==> number of Bloch states */
  double *k;               /* Bloch vectors (if needed) */
  double *wtk;             /* Weight for the k point (if needed) */

  /* The eigenstates */
  int spinType;            /* 0 ==> No spin
                              1 ==> Collinear spin
                              2 ==> Non-collinear spin */
  int overlapType;         /* 0 ==> No overlap matrix
                              1 ==> Has overlap matrix */
  int nSets;               /* The numbers of sets of eigenstates.
                              nSets = max(1,nBloch) for no spin or non-collinear spin,
                              nSets = 2*max(1,nBloch) for collinear spin. */
  int *ik;                 /* The k point index for each set */
  int *is;                 /* The spin state for each set
                              0 ==> No spin
                              1 ==> Collinear spin up
                              2 ==> Collinear spin down
                              3 ==> Noncollinear spin */
  int Norbitals;           /* The number of atomic orbitals */
  double spinDegeneracy;   /* The spin degeneracy used for the occupancies */
  double **occ;            /* The occupancy of each orbital */
};
typedef struct stateInfo_s stateInfo_t;

struct multipole_Struc
{
  /*
   * Multipole index p is given by the following rules:
   * monopole  : p = 0
   * dipole    : p = 1, 2, 3
   * quadrupole: p = 4, 5, 6, 7, 8
   * etc
   */
  /**********************************/
  /*** CHARGE AND SPIN INTEGRALS ****/
  /**********************************/
  /* The highest multipole order to use */
  int lmax;

  /* The number of entries needed to store the multipoles */
  int sizeQ;

  /*
   * Integrals Qint[p][i][n][k]:
   * p is the multipole index
   * i is an atom index
   * n is a neighbour index
   * k is index for pairs of orbitals
   * NOTE: This order is chosen so we can use the sparse matrix memory allocator
   *
   * Derivatives of integrals dQint[c][p][i][n][k]:
   * c is a cartesian direction for the gradient
   */
  sparseMatrix_t *Qint;
  sparseMatrix_t **dQint;

  /************************/
  /******** CHARGE ********/
  /************************/
  /* Multipoles Q[i][p]:
   * i is atom index
   * p is a multipole index
   *
   * For the input multipoles, we need a stack for mixing.
   * The first index is the position in the stack
   *
   * The derivatives of the output moments have the indices
   * dQ[i][p][n][k]:
   * i is atom index
   * p is a multipole index
   * n is the neighbour index
   * k is a cartesian direction
   */
  int NQ;
  double ***Q_inp;
  double ***Q_res;
  double **Q_out;
  double ****dQ_out;

  /************************/
  /********* SPIN *********/
  /************************/
  /* Spin moment multipoles Mx[i][p] etc:
   * i is atom index
   * p is a multipole index
   */
  double **Mx_inp;
  double **My_inp;
  double **Mz_inp;
  double **Mx_out;
  double **My_out;
  double **Mz_out;
  double ****dMx_out;
  double ****dMy_out;
  double ****dMz_out;

  /************************************/
  /******** MADELUNG INTEGRALS ********/
  /************************************/
  /* a is the exponent for the Gaussian distribution of the charge
   *
   * Integrals are madelung[i][j][pi][pj]:
   * i is atom index
   * j is atom index
   * pi is a multipole index
   * pj is a multipole index
   *
   * The derivatives of the integrals are dMadelungdR[i][j][k][pi][pj]:
   * k is a cartesian index (0->x, 1->y, 2->z)
   *
   * The lattice derivatives for periodic systems are
   * dMadelungda[i][j][n][k][pi][pj]:
   * n is the lattice vector index
   * k is a cartesian index
   */
  double *a;
  double ****madelung;
  double *****dMadelungdR;
  double ******dMadelungda;

  /**********************************/
  /******** STONER INTEGRALS ********/
  /**********************************/
  double *I;

  /**********************************/
  /*** SPIN CONSTRAINT PARAMETERS ***/
  /**********************************/
  int spinConstrainFlag;
  double *bx, *by, *bz;		 /* The constraining magnetic fields */
};
typedef struct multipole_Struc multipole_t;

/* Simulation information */
struct simulationInfo_Struc
{
  /* Job name */
  char *JobName;

  /* Output variables */
  FILE *outfp;
  int verbosity;
  int writeBondLengthFlag;
  int writeBondAngleFlag;
  int writeWavefunctionFlag;
  int writeOverlapFlag;
  double bondLengthThreshold;
  int readRestartFile;
  int writeRestartFile;

  /* Model definition */
  model_t model;

  /* Other flags */
  int jobType;               /* The type of simulation */
  int testOverlapFlag;       /* Flag determining if the overlap matrix is tested for small eigenvalues */
  int denHistoFlag;          /* The density histogram flag */
  int writeHamFileFlag;      /* Whether to write *.ham file out regardless of verbosity (0=no, 1=yes) */
  int externalModelFlag;     /* Flag indicating which external model is to be used to compute energies and forces */
  int vdWFlag;               /* Flag indicating inclusion of van der Waals */

  /* Electron population information */
  double electronExcess;
  double mu, kT, nElectrons;
  double *Mx, *My, *Mz;      /* The initial magnetic moments */
  double *bx, *by, *bz; 		 /* The constraining magnetic fields */

  /* Self-consistency flags */
  int maxLoops;              /* The maximum number of SCF loops */
  int maxMixLevels;          /* The maximum number of previous densities used in Pulay mixing */
  int mixScheme;             /* 0 ==> Linear mixing; 1 ==> Pulay mixing */
  double mixFactor;          /* The fraction of output density added to the input density */
  int mixLevels;             /* The number of previous densities available to be used in Pulay mixing */
  int spinMixScheme;         /* 0 ==> Linear mixing; 1 ==> Pulay mixing */
  int socFlag;               /* 0 ==> No SOC; 1 ==> Include SOC */
  int peierlsFlag;           /* 0 ==> No Peierls; 1 ==> Perform Peierls transformation */
  int stonerFlag;            /* 0 ==> Exclude Stoner Exchange; 1 ==> Includ eStoner Exchange */
  double spinMixFactor;      /* The fraction of output spin density added to the input spin density */
  int spinMixLevels;         /* The number of previous spin densities available to be used in Pulay mixing */
  double residueTolerance;   /* The maximum allowed root mean square difference between input and output densities */
  double energyTolerance;    /* The maximum allowed energy difference between sequential SCF steps */
  int continueRelaxationWhenNotConverged; /* Determines whether to exit Plato when a relaxation step fails to reach self-consistency */

  /* Forces and energies */
  double *F0, *F1, *F2, *F, *Fexternal;
  double Flattice0[3][3], Flattice1[3][3], Flattice2[3][3], Flattice[3][3], FlatticeExternal[3][3];
  double Etotal, Eentropy, Eatom, EsingleParticle, Eexternal, Econstraint;
  double E0, E0_KE, E0_NA, E0_ES, E0_XC, E0_NL;
  double E1, E1_KE, E1_NA, E1_ES, E1_XC, E1_NL;
  double E2, E2_ES, E2_XC;

  /* Relaxation parameters*/
  int relaxMethod;           /* The method to be used to perform relaxation, both atomic and cell */
  int cellRelaxMode;         /* The mode to be used to perform cell relaxation */
  int step;                  /* The step in the relaxation */
  int nSteps;                /* The maximum number of steps in the relaxation */
  int writeCoordinatesSteps; /* The number of steps between writes of the atomic coordinates */
  int PBCFlag;               /* Determines if atoms leaving cell are translated back in. 0 ==> No. 1 ==> Yes. */
  double maxF;               /* The maximum force at a given step */
  double maxStress;          /* The maximum stress at a given step */
  double fTol;               /* The maximum allowed atomic force for a relaxed structure. The simulation halts when maxF < ftol */
  double maxDisplacement;    /* The maximum allowed atomic dispacement */
  double stressTol;          /* The maximum allowed cell force for a relaxed structure. The simulation halts when maxStress < stressTol */
  double relaxFactor;        /* The default factor for computing a displacement from a force in a steepest descent calculation */
  double latticeRelaxFactor; /* The default factor for computing a cell displacement from a lattice force in a steepest descent calculation */

  /* MD and MC parameters */
  double MDTimeStep;
  double atomTemperature;
  double temperatureTolerance;
  double dMmax;
  int MCMode;

  /* DFT3 parameters */
  int nR, nTheta, nPhi;

  /* DFT2 parameters */
  int psiFlag;
  int inverseSKFlag;
  int xcFunctional;
  int xcExpansion;
  int mcWedaXcFlag;
  int momExpansionNPoly;
  int momExpansionIntMethod;
  int meshType;
  int addCorrectingPPFromBdtFlag; // Determine whether to add tabulated PP correction to tabulated PP
  int e0Method;
  int e0xcCorr;
  int e0XcCorrOrder;
  int e0NonLocXcCorr;
  int numbE1XtalFlags;
  int e1XtalXcMethod[STATES_MAX_E1_XTAL_MIXED_METHODS];
  int e1XtalNonLocXcCorr;
  int e1HopXcMethod; // Only 1 hop-method can currently be used at once
  int e1HopNonLocXcCorr;
  int e1XtalVnaMethod;
  int e1HopVnaMethod;
  int e1XtalVnlMethod;
  int e1HopVnlMethod;
  int kineticMatrixElementsFlag;
  double diameterNLV;
  double gridSpacing;
  double xtalE1ScaleFactors[STATES_MAX_E1_XTAL_MIXED_METHODS]; // Max of 5 scale factors
  double denHistoSpacing;

  /* Env-Dep TB options */
  int calcScreenFunctFlag;
  int screenDampFunctFlag;
  int screenFunctAngDepFlag;
  double screenDampParam;
  double *screenFunctParams;

  /* Temporary storage for input */
  int NTypes;
  int NAtom;
  int *AtomType;
  int CellRepeat[3];
  double *Pos;
  double CellVec[3][3];
  double CellSize[3];
  char *Name;

  /* External Fields */
  int spinConstrainFlag;
  int magneticFieldFlag;
  double B_unit[3];
  double B_parameters[3];
};
typedef struct simulationInfo_Struc simulationInfo_t;

/* These two structs are to get relevant indices in Spline_Strucs used in TB1/TBInt */
struct IntegralIdxer_Struc
{
  int   numbInts;  // Total number of different integrals
  int * index;     // All non empty indices in Spline_Struc
  int * atomAIdx;
  int * atomBIdx;
  int * orbAIdx;
  int * orbBIdx;
  int * orbAxAngMom;
};

struct IntegralIdxInfo_Struc
{
  int nTypes;      // Types of atom present
  int maxQuantNum;
  int maxL;
  int *nlList;     // nlList[1] contains number of shells for atom type with index "1"
  int *lVals;      // lVals[atomIdx*maxL+orbIdx] should give the angular momentum of orbital "orbIdx" on atom "atomIdx"
  int *ppNlList;   // ppNlList[i] = number of projectors for PP on the i-th atom type
  int *ppLVals;    // l values of pseudopot projectors. same indexing as lVals
};

/* Various enums */
enum states_manipbdttabs_errors { STATES_MANIPBDTTABS_ERROR_DIFF_NUMB_POINTS = -1,
    STATES_MANIPBDTTABS_ERROR_XVALS_DIFFERENT = -2 };

/* Prototypes */
int EchoCitation (simulationInfo_t *simInfo);
int ReadCell (simulationInfo_t *simInfo, char **FileTok, int NFileTok);
int ReadCoordinates (simulationInfo_t *simInfo, char **FileTok, int NFileTok);
int ReadModelType (simulationInfo_t *simInfo);

int statesInit (stateInfo_t *stateInfo);
int statesFree (stateInfo_t *stateInfo);
int statesMultipoleInit(simulationInfo_t *simInfo, stateInfo_t *stateInfo, sparseGeom_t *geom, multipole_t *mp);
int statesMultipoleCopy(stateInfo_t *stateInfo, sparseGeom_t *geom, multipole_t *mp1, multipole_t *mp2);

int statesDiagonalize (stateInfo_t *stateInfo, sparseGeom_t *geom, sparseMatrices_t *sparseH, sparseMatrix_t *sparseS, eigenStates_t **eigenStates);
int statesBuildDensityMatrix (stateInfo_t *stateInfo, sparseGeom_t *geom, eigenStates_t **eigenStates, sparseMatrix_t *sparseRho, sparseMatrix_t *sparseEmatrix, sparseMatrices_t *sparseM);
double statesOccupy (double mu, double kT, stateInfo_t *stateInfo, eigenStates_t **eigenStates);
double statesFindMu (double nElec, double kT, stateInfo_t *stateInfo, eigenStates_t **eigenStates);
int statesWriteEigenstates (atomsData_t *ad, stateInfo_t *stateInfo, sparseGeom_t *geom, eigenStates_t **eigenStates, FILE *wffp);
int statesBuildMultipoleIntegrals (sparseGeom_t *geom, GaussData_t **GaussData, multipole_t *multipole);
int statesBuildMultipoleMoments (sparseGeom_t *geom, multipole_t *multipole, sparseMatrix_t *sparseRho);
int statesDerivativeMultipoleIntegrals(sparseGeom_t *geom, GaussData_t **GaussData, multipole_t *multipole);
int statesDerivativeMultipoleMoments (sparseGeom_t *geom, multipole_t *multipole, sparseMatrix_t *sparseRho);
int statesBuildMagneticMoments(int spinType, sparseGeom_t *geom, multipole_t *multipole, sparseMatrices_t *sparseM);
double statesUpdateInputMoments(int spinType, sparseGeom_t *geom, simulationInfo_t *simInfo, multipole_t *multipole);
double statesOverlap(stateInfo_t *stateInfo, sparseGeom_t *geom, int m_i, int m_j, eigenStates_t **TEMO, eigenStates_t **IE, sparseMatrices_t *I);
double statesOrbitalExpectation(stateInfo_t *stateInfo, sparseGeom_t *geom, int m, eigenStates_t **eigenStates, sparseMatrices_t* Operator);
int statesIntegral(stateInfo_t *stateInfo, sparseGeom_t *geom, int m_i, int m_j, eigenStates_t **p1, eigenStates_t **p2, sparseMatrices_t *Operator, double *retval_r, double *retval_i);

int ReadInput (stateInfo_t *stateInfo, simulationInfo_t *simInfo);

twoCentreData_t ** getTcdStruct();
int tablesInit (model_t *model, atomsData_t *atomsData);
int tablesBuildBlock (sparseGeom_t *geom, atomsData_t *atomsData, model_t *model, int i, int j, double *S, double *H);
int tablesGradientBlock (sparseGeom_t *geom, atomsData_t *atomsData, model_t *model, int i, int j, double *dS, double *dH);
int tablesSK (double Ri[3], int nShell_i, int *lList_i, double Rj[3], int nShell_j, int *lList_j, multiSpline_t **t, double *SKBlock);
double tablesPP (int Zi, int Zj, double r);
double tablesPP_tcdArg(int Zi, int Zj, double r, twoCentreData_t **twoCentTable);
double tablesdPP (int Zi, int Zj, double r);
double tablesEP (int Zi, int Zj, double r);
double tablesEP_tcdArg (int Zi, int Zj, double r, twoCentreData_t **twoCentTable);
double tablesdEP (int Zi, int Zj, double r);
double tablesEmbed (atomsData_t *atomsData, int Zi, double x);
double tablesdEmbed (atomsData_t *atomsData, int Zi, double x);

int tablesBuildMultipoleIntegrals(sparseGeom_t *geom, atomsData_t *atomsData, multipole_t *multipole);
int tablesDerivativeMultipoleIntegrals(sparseGeom_t *geom, atomsData_t *atomsData, multipole_t *multipole);

int electroInit (sparseGeom_t *g, atomsData_t *ad, multipole_t *mp);
int electroMadelung (sparseGeom_t *g, multipole_t *mp);
int electroMadelungDerivative (sparseGeom_t *g, multipole_t *mp);
int electroHamiltonian (sparseGeom_t *g, multipole_t *mp, sparseMatrix_t *H, sparseMatrix_t *S);
int electroForce (sparseGeom_t *g, multipole_t *mp, double *F, double Flattice[3][3]);
double electroEnergy (sparseGeom_t *g, multipole_t *mp);
//##################################
// New isolated monopole-only electrostatics entry points.
int electroMonopoleHamiltonian3D (sparseGeom_t *g, multipole_t *mp, sparseMatrix_t *H, sparseMatrix_t *S);
int electroMonopoleForce3D (sparseGeom_t *g, multipole_t *mp, double *F, double Flattice[3][3]);
double electroMonopoleEnergy3D (sparseGeom_t *g, multipole_t *mp);
//##################################

int magneticInit (int spinType, simulationInfo_t *simInfo, sparseGeom_t *g, atomsData_t *ad, multipole_t *mp);
int magneticForce (int spinType, sparseGeom_t *g, multipole_t *mp, double *F);
int magneticHamiltonian (int spinType, sparseGeom_t *g, multipole_t *mp, sparseMatrices_t *Hmag);
int magneticDipoleHamiltonian (int spinType, sparseGeom_t *g, multipole_t *mp, atomsData_t *atomsData, simulationInfo_t *simInfo, sparseMatrices_t *Hdip);
int magneticField (simulationInfo_t *simInfo, double *B);
int magneticVectorPotential(simulationInfo_t *simInfo, double *A, double *R);
int magneticSpinConstraintHamiltonian (int spinType, sparseGeom_t *g, multipole_t *mp, sparseMatrices_t *Hmag);
double magneticPenaltyEnergy (sparseGeom_t *g, multipole_t *mp);
double magneticEnergy (int spinType, sparseGeom_t *g, multipole_t *mp);
double magneticConstraintEnergy(int spinType, sparseGeom_t *g, multipole_t *mp);
int magneticSOCHamiltonian (atomsData_t *atomsData, int spinType, sparseGeom_t *g, sparseMatrices_t *HSOC);
int magneticPeierlsSubstitution_sparse(sparseGeom_t *g, simulationInfo_t *simInfo, sparseMatrices_t *sparseMatrices, int direction);
int magneticPeierlsSubstitution_sparseMatrix(sparseGeom_t *g, simulationInfo_t *simInfo, sparseMatrix_t *sparseMatrix, int direction);
int magneticPeierlsScalarPotTransHamiltonian( sparseGeom_t *g, simulationInfo_t *simInfo, sparseMatrices_t *sparseMatrices);
int statesDerivativeMagneticMoments(int spinType, sparseGeom_t *geom, multipole_t *multipole, sparseMatrices_t *sparseM);
double magneticPeierlsPhase(sparseGeom_t *g, simulationInfo_t *simInfo, int i, int j);
int magneticAssignSpinMatrices (int spinType, sparseGeom_t *g, atomsData_t *atomsData, simulationInfo_t *simInfo, sparseMatrices_t *S_x, sparseMatrices_t *S_y, sparseMatrices_t *S_z);
int magneticAssignAngularMatrices (int spinType, sparseGeom_t *g, atomsData_t *atomsData, simulationInfo_t *simInfo, sparseMatrices_t *L_x, sparseMatrices_t *L_y, sparseMatrices_t *L_z);

int getBdtTableStructFieldsInArray(bdtTables_t *bdtTables, bdtTablesSet_t ***outArray);
int initBdtTablesStruc(bdtTables_t *bdtTables);
int getTablesFromBdtFile (char *filePath, bdtTables_t *bdtTables);
int writeBdtTables(char *pathName, bdtTables_t *tables);
int appendIntegralSetToFileForm4(FILE *fp, bdtTablesSet_t *intSet);
int applySplinesToAllBdtTables(bdtTables_t *bdtTables);
int getReqIntsFromBdtFile(char *filePath, twoCentreData_t *outData, model_t *modelData);
int getAllIntsFromBdtFileToTwoCentStruct(char *filePath, twoCentreData_t *outData);
int getAllIntsFromBdtTableToTwoCentStruct(bdtTables_t *bdtTabs, twoCentreData_t *outData);
int getBdtFileFormatFlag(char *filePath);

int states_addTablesSetBToSetA(bdtTablesSet_t *setA, bdtTablesSet_t *setB);
int states_applyPPCorrToPPIfPresent(bdtTables_t *inpTables);
int states_applyHopCorrToHopIntsIfPresent(bdtTables_t *inpTables);

int freeBdtTablesStrucInternals(bdtTables_t *bdtTables);
int freeBdtSetStruc(bdtTablesSet_t *tablesSet);
void freeTableSplineStrucInternals(tableSpline_t *tabSpline);
void freeMultiTableSplineStrucInternals(multiSpline_t *multiTab);

int getIdxListIntegrals(struct IntegralIdxer_Struc * integralIdxs, struct IntegralIdxInfo_Struc *integralIdxInfo, int maxNumbIntegrals, char * integralType);
int getIdxListIntegralsOneAtomPair(struct IntegralIdxer_Struc * integralIdxs, struct IntegralIdxInfo_Struc *integralIdxInfo, int maxNumbIntegrals, char * integralType, int atomAIdx, int atomBIdx);
int getIdxListIntegralsForInitialisationOneAtomPair(struct IntegralIdxer_Struc * integralIdxs, struct IntegralIdxInfo_Struc *integralIdxInfo, int maxNumbIntegrals, char * integralType, int atomAIdx, int atomBIdx);
void freeIntegralIdxerStructInternals(struct IntegralIdxer_Struc * integralIdxs);

int tablesReadAtomData (model_t *model, atomsData_t *atomsData);
int freeBdtSetStrucInternals(bdtTablesSet_t *tablesSet);
void statesTranspose(double *array, int n, int dataSize);
#endif
