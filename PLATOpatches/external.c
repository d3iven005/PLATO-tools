#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "external.h"
#include "../misc/misc.h"

#define BUFFERSIZE 1024

int
externalModel (int flag, double a[3][3], int N, double *R, int *Z, double* E, double* F)
{
  int i;

  /* Invoke the relevant model */
  switch (flag)
    {
    case 0:
      /* No external program */
      *E = 0.0;
      for (i = 0; i < 3*N; i++)
        F[i] = 0.0;
      break;
    case 1:
      /* Invoke LAMMPS */
      callLAMMPS(a, N, R, Z, E, F);
      break;
    case 2:
        //printf("flag = %d\n", flag);
        printf("Running MACE model \n");
        //printf("Matrix a[3][3]:\n");
        //for (int i = 0; i < 3; i++) {
        //    for (int j = 0; j < 3; j++) {
        //        printf("%.3f ", a[i][j]*0.529177);
        //    }
        //    printf("\n");
        //}
        //printf("N = %d\n", N);
        //printf("Z (atomic numbers):\n");
        //for (int i = 0; i < N; i++) {
        //    printf("Atom %d: %d\n", i+1, Z[i]);
        //}
        //printf("R (positions):\n");
        //for (int i = 0; i < N; i++) {
        //    printf("Atom %d: \n %.3f %.3f %.3f\n", i+1, R[3*i], R[3*i+1], R[3*i+2]);
        //}
        //printf("E = %15.8f\n", *E);
        //printf("Forces F:\n");
        //for (int i = 0; i < N; i++) {
        //    printf("Atom %d: \n %.3f\n %.3f\n %.3f\n", i+1, F[3*i], F[3*i+1], F[3*i+2]);
        //}
        writexyz(N, a, R, Z, "mace.xyz",E, F); 
        break;
    case 3:
        printf("Running Quantum Espresso \n");
        runqe(N, a, R, Z, "qe.xyz", E, F);
        break;
    default:
      /* Unknown program */
      printf ("Unknown external program %i requested.\n", flag);
      printf ("Allowed values are:\n");
      printf ("0: No external program\n");
      printf ("1: LAMMPS\n");
      exit (0);
      break;
    }

  return 0;
}

/*------------------------------------------------------------------------------*/

int
callLAMMPS(double a[3][3], int N, double *R, int *Z, double* E, double* F)
{
  int i, j, p, Zi, atomTypeIn, Nin, *atomType, NatomType;
  double Fx, Fy, Fz;
  char buffer[BUFFERSIZE];
  FILE *xyzfp, *efsfp;
  char *command = "bash runLAMMPS.sh";

  /* Determine the number of atom types and the type for each atom */
  atomType = malloc (N*sizeof(int));
  for (i = 0; i < N; i++)
    atomType[i] = -1;
  NatomType = 0;
  for (i = 0; i < N; i++)
    {
      if (atomType[i] == -1)
        {
          NatomType++;
          Zi = Z[i];
          for (j = i; j < N; j++)
            {
              if (Z[j] == Z[i])
                atomType[j] = NatomType;
            }
        }
    }

  /* Write out XYZ file for input to external program
   * The standard XYZ format is used, with the lattice parameters being inserted in the comment line. */
  if ((xyzfp = fopen ("plato.xyz", "w")) == 0)
    {
      printf ("Unable to open output file plato.xyz.\n");
      exit(0);
    }
  fprintf (xyzfp, "LAMMPS simulation to support TB3\n\n");
  fprintf (xyzfp, "%5i                     atoms\n", N);
  fprintf (xyzfp, "%5i                     atom types\n", NatomType);
  fprintf (xyzfp, "%12.5f %12.5f xlo xhi\n", 0.0, a[0][0]*BOHRRADIUS/ANGSTROM);
  fprintf (xyzfp, "%12.5f %12.5f ylo yhi\n", 0.0, a[1][1]*BOHRRADIUS/ANGSTROM);
  fprintf (xyzfp, "%12.5f %12.5f zlo zhi\n", 0.0, a[2][2]*BOHRRADIUS/ANGSTROM);
  fprintf (xyzfp, "\nAtoms\n\n");
  for (i = 0; i < N; i++)
    fprintf (xyzfp, "%5i %5i %15.5f %15.5f %15.5f\n", i+1, atomType[i], R[3*i + 0]*BOHRRADIUS/ANGSTROM, R[3*i + 1]*BOHRRADIUS/ANGSTROM, R[3*i + 2]*BOHRRADIUS/ANGSTROM);
  fclose (xyzfp);

  /* Run the program */
  p = system (command);

  /* Read in the energy and forces from file
   * The format of the file is very similar to XYZ. It is:
   *   + Line 1: Number of atoms
   *   + Line 2: The energy in eV
   *   + Lines 3 etc: The position of the atom in the original XYZ file (starting at 0),
   *     then the x, y, and z components of the force in eV/angstrom
   */
  if ((efsfp = fopen ("plato.efs", "r")) == 0)
    {
      printf ("Unable to open output file plato.efs.\n");
      exit(0);
    }
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  sscanf (buffer, "%i", &Nin);
  if (Nin != N)
    {
      printf ("Incorrect number of atoms from external program output: %i\n", Nin);
      exit(0);
    }
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  for (i = 0; i < N; i++)
    {
      fgets (buffer, BUFFERSIZE, efsfp);
      sscanf (buffer, "%i %i %lf %lf %lf", &p, &atomTypeIn, &Fx, &Fy, &Fz);
      p--;
      if (atomTypeIn == atomType[i])
        {
          if ((p >= 0) && (p < N))
            {
              F[3*p + 0] = Fx*(EV/ANGSTROM)/(RYDBERG/BOHRRADIUS);
              F[3*p + 1] = Fy*(EV/ANGSTROM)/(RYDBERG/BOHRRADIUS);
              F[3*p + 2] = Fz*(EV/ANGSTROM)/(RYDBERG/BOHRRADIUS);
            }
          else
            {
              printf ("Incorrect index for force: %i.\n", p);
              exit(0);
            }
        }
      else
        {
          printf ("Incorrect atom type: %i.\n", atomTypeIn);
          exit(0);
        }
    }
  fgets (buffer, BUFFERSIZE, efsfp);
  fgets (buffer, BUFFERSIZE, efsfp);
  sscanf (buffer, "%lf", E);
  *E *= EV/RYDBERG;
  fclose (efsfp);

  /* Clean up memory */
  free (atomType);

  return 0;
}
/*------------------------------------------------------------------------------*/
/*-----------------------------function for MACE--------------------------------*/
static const char *atomicSymbol(int atomicNumber)
{
    static const char *symbols[] = {
        "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"
    };

    if (atomicNumber >= 1 && atomicNumber <= 20) {
        return symbols[atomicNumber];
    }
    if (atomicNumber == 29) {
        return "Cu";
    }
    if (atomicNumber == 35) {
        return "Br";
    }
    return "X";
}

int writexyz(int N, double a[3][3], double* R, int* Z, const char* filename, double* E, double* F){
    FILE *xyzfp;
    FILE *fp;
    char buffer[BUFFERSIZE];  
    int i;
    char species[N][3];
    if ((xyzfp = fopen(filename, "w")) == NULL) {
        printf("Unable to open output file %s.\n", filename);
        return 1;  // Error opening file
    }
    fprintf(xyzfp, "%d\n", N);  // Write number of atoms
    fprintf(xyzfp, "Lattice=\" ");
    for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                fprintf(xyzfp, "%.3f ", a[i][j]*0.529177);
            }
        }
    fprintf(xyzfp," \" ");
    fprintf(xyzfp, " Properties=species:S:1:pos:R:3 pbc= \" T T T \" \n" );
    // Write the atom data
    for (int i = 0; i < N; i++) {
        const char *atomType = atomicSymbol(Z[i]);
        fprintf(xyzfp, "%s  %15.8f  %15.8f  %15.8f\n", atomType, 
                R[3*i] * 0.529177, R[3*i+1] * 0.529177, R[3*i+2] * 0.529177); 
    }
    fclose(xyzfp);
    int ret = system("mace_eval_configs --configs mace.xyz --model mace.model --output mace.out --default_dtype float32 > /dev/null 2>&1");
    fp =fopen("mace.out","r");
    fgets(buffer, BUFFERSIZE, fp);
    sscanf(buffer, "%d", &N);
    fgets(buffer, BUFFERSIZE, fp);
    char *energy_ptr = strstr(buffer, "MACE_energy=");
    if (energy_ptr != NULL) {
        sscanf(energy_ptr + 12, "%lf", E);
        //printf("Energy: %lf eV\n", *E);
        *E /= 13.6057;
        printf("E_MACE = %lf Ry\n", *E);
    }
    for (i = 0; i < N; i++) {
        fgets(buffer, BUFFERSIZE, fp);
        sscanf(buffer, "%2s %*f %*f %*f %lf %lf %lf", species[i], &F[3*i], &F[3*i+1], &F[3*i+2]);
    //    printf("Atom %d: %s  Force: (%.5f, %.5f, %.5f) eV/A \n", i+1, species[i],F[3*i], F[3*i+1], F[3*i+2]);
        F[3*i] /= 25.7111;
        F[3*i + 1] /= 25.7111;
        F[3*i + 2] /= 25.7111;
    }
    int rm = system("rm mace.xyz mace.out");
    return 0;
}
/*------------------------------------------------------------------------------*/
/*-----------------------------function for QE----------------------------------*/
int runqe(int N, double a[3][3], double* R, int* Z, const char* filename, double* E, double* F){
    FILE *xyzfp;
    FILE *fp;
    char buffer[BUFFERSIZE];  
    int i;
    char species[N][3];
    if ((xyzfp = fopen(filename, "w")) == NULL) {
        printf("Unable to open output file %s.\n", filename);
        return 1;  // Error opening file
    }
    fprintf(xyzfp, "%d\n", N);  // Write number of atoms
    fprintf(xyzfp, "Lattice=\" ");
    for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                fprintf(xyzfp, "%.3f ", a[i][j]*0.529177);
            }
        }
    fprintf(xyzfp," \" ");
    fprintf(xyzfp, " Properties=species:S:1:pos:R:3 pbc= \" T T T \" \n" );
    // Write the atom data
    for (int i = 0; i < N; i++) {
        const char *atomType = atomicSymbol(Z[i]);
        fprintf(xyzfp, "%s  %15.8f  %15.8f  %15.8f\n", atomType, 
                R[3*i] * 0.529177, R[3*i+1] * 0.529177, R[3*i+2] * 0.529177); 
    }
    fclose(xyzfp);
    int xyztoqe = system("python3 1_xyztoqe.py qe.xyz -o qe.in --prefix 1 --pseudo_dir ./PSV/ --kmesh 1 1 1");
    int runqe = system("pw.x <qe.in>qe.out");
    int qetoxyz = system("python3 2_qetoxyz.py --qe_in qe.in --qe_out qe.out -o qe.xyz");
    fp =fopen("qe.xyz","r");
    fgets(buffer, BUFFERSIZE, fp);
    sscanf(buffer, "%d", &N);
    fgets(buffer, BUFFERSIZE, fp);
    char *energy_ptr = strstr(buffer, "QE_energy=");
    if (energy_ptr != NULL) {
        sscanf(energy_ptr + 10, "%lf", E);
        printf("Energy: %lf eV\n", *E);
        *E /= 13.6057;
        printf("E_QE = %lf Ry\n", *E);
    }
    for (i = 0; i < N; i++) {
        fgets(buffer, BUFFERSIZE, fp);
        sscanf(buffer, "%2s %*f %*f %*f %lf %lf %lf", species[i], &F[3*i], &F[3*i+1], &F[3*i+2]);
    //    printf("Atom %d: %s  Force: (%.5f, %.5f, %.5f) eV/A \n", i+1, species[i],F[3*i], F[3*i+1], F[3*i+2]);
        F[3*i] /= 25.7111;
        F[3*i + 1] /= 25.7111;
        F[3*i + 2] /= 25.7111;
    }
    //int rm = system("rm -r 1.save 1.xml qe.in qe.out qe.xyz");
    return 0;
}
