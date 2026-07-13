//##################################
// New isolated 3D periodic monopole-only electrostatics implementation.
// This file avoids building or reading the full Madelung matrix.
//##################################

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "states.h"

/* Largest value of an exponent in an Ewald sum */
#define MAXEXP 10.0

int electroMonopolePotential3D (sparseGeom_t *g, multipole_t *mp, double *Q, double *V);

static int
electroMonopoleCheck3D (sparseGeom_t *g, multipole_t *mp)
{
  return (g->NPerDim == 3) && (mp->lmax == 0) && (mp->sizeQ == 1);
}

int
electroMonopolePotential3D (sparseGeom_t *g, multipole_t *mp, double *Q, double *V)
{
  int i;
  int NG0, NG1, NG2, iG0, iG1, iG2;
  double G[3], GG, magG0, magG1, magG2;
  double amax, ai, b, common;
  double phase, sumCos, sumSin, s, c, amp, weightedCharge;
  double *phaseSin, *phaseCos, *atomAmp;

  if (!electroMonopoleCheck3D (g, mp))
    {
      printf ("electroMonopolePotential3D: only valid for 3D periodic monopole-only electrostatics.\n");
      return 1;
    }

  phaseSin = malloc (g->N*sizeof(double));
  phaseCos = malloc (g->N*sizeof(double));
  atomAmp = malloc (g->N*sizeof(double));

  for (i = 0; i < g->N; i++)
    V[i] = 0.0;

  amax = mp->a[0];
  for (i = 1; i < g->N; i++)
    {
      ai = mp->a[i];
      if (ai > amax)
        amax = ai;
    }
  b = sqrt(amax/2.0);

  magG0 = g->ReciprocalVec[0][0]*g->ReciprocalVec[0][0] + g->ReciprocalVec[0][1]*g->ReciprocalVec[0][1] + g->ReciprocalVec[0][2]*g->ReciprocalVec[0][2];
  magG1 = g->ReciprocalVec[1][0]*g->ReciprocalVec[1][0] + g->ReciprocalVec[1][1]*g->ReciprocalVec[1][1] + g->ReciprocalVec[1][2]*g->ReciprocalVec[1][2];
  magG2 = g->ReciprocalVec[2][0]*g->ReciprocalVec[2][0] + g->ReciprocalVec[2][1]*g->ReciprocalVec[2][1] + g->ReciprocalVec[2][2]*g->ReciprocalVec[2][2];
  NG0 = 1+(int)(sqrt(MAXEXP*4.0*b*b/magG0));
  NG1 = 1+(int)(sqrt(MAXEXP*4.0*b*b/magG1));
  NG2 = 1+(int)(sqrt(MAXEXP*4.0*b*b/magG2));

  for (iG0 = -NG0; iG0 <= NG0; iG0++)
    for (iG1 = -NG1; iG1 <= NG1; iG1++)
      for (iG2 = -NG2; iG2 <= NG2; iG2++)
        {
          G[0] = ((double)iG0)*g->ReciprocalVec[0][0] + ((double)iG1)*g->ReciprocalVec[1][0] + ((double)iG2)*g->ReciprocalVec[2][0];
          G[1] = ((double)iG0)*g->ReciprocalVec[0][1] + ((double)iG1)*g->ReciprocalVec[1][1] + ((double)iG2)*g->ReciprocalVec[2][1];
          G[2] = ((double)iG0)*g->ReciprocalVec[0][2] + ((double)iG1)*g->ReciprocalVec[1][2] + ((double)iG2)*g->ReciprocalVec[2][2];
          GG = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];

          if (GG > 1.0e-10)
            {
              common = (8.0*PI/g->Volume)/GG;
              sumCos = 0.0;
              sumSin = 0.0;

              for (i = 0; i < g->N; i++)
                {
                  phase = G[0]*g->R[3*i+0] + G[1]*g->R[3*i+1] + G[2]*g->R[3*i+2];
                  s = sin(phase);
                  c = cos(phase);
                  amp = exp(-0.25*GG/mp->a[i]);

                  phaseSin[i] = s;
                  phaseCos[i] = c;
                  atomAmp[i] = amp;

                  weightedCharge = Q[i]*amp;
                  sumCos += weightedCharge*c;
                  sumSin += weightedCharge*s;
                }

              for (i = 0; i < g->N; i++)
                V[i] += common*atomAmp[i]*(phaseCos[i]*sumCos + phaseSin[i]*sumSin);
            }
        }

  free (phaseSin);
  free (phaseCos);
  free (atomAmp);

  return 0;
}

double
electroMonopoleEnergy3D (sparseGeom_t *g, multipole_t *mp)
{
  int i;
  double sum, *Q, *V;

  if (!electroMonopoleCheck3D (g, mp))
    {
      printf ("electroMonopoleEnergy3D: only valid for 3D periodic monopole-only electrostatics.\n");
      return 0.0;
    }

  Q = malloc (g->N*sizeof(double));
  V = malloc (g->N*sizeof(double));
  for (i = 0; i < g->N; i++)
    Q[i] = mp->Q_out[i][0];

  electroMonopolePotential3D (g, mp, Q, V);

  sum = 0.0;
  for (i = 0; i < g->N; i++)
    sum += Q[i]*V[i];
  sum *= 0.5;

  free (Q);
  free (V);

  return sum;
}

int
electroMonopoleForce3D (sparseGeom_t *g, multipole_t *mp, double *F, double Flattice[3][3])
{
  int i, j, j0, k, n;
  int NG0, NG1, NG2, iG0, iG1, iG2;
  double G[3], GG, magG0, magG1, magG2;
  double amax, ai, b, common;
  double phase, sumCos, sumSin, s, c, amp, weightedCharge;
  double dR[3], l;
  double *phaseSin, *phaseCos, *atomAmp;
  double *F_Q, *F_B, *Q, *V;

  if (!electroMonopoleCheck3D (g, mp))
    {
      printf ("electroMonopoleForce3D: only valid for 3D periodic monopole-only electrostatics.\n");
      return 1;
    }

  phaseSin = malloc (g->N*sizeof(double));
  phaseCos = malloc (g->N*sizeof(double));
  atomAmp = malloc (g->N*sizeof(double));
  F_Q = malloc(3*g->N*sizeof(double));
  F_B = malloc(3*g->N*sizeof(double));
  Q = malloc (g->N*sizeof(double));
  V = malloc (g->N*sizeof(double));

  for (i = 0; i < 3*g->N; i++)
    {
      F_Q[i] = 0.0;
      F_B[i] = 0.0;
      F[i] = 0.0;
    }
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      Flattice[i][j] = 0.0;

  amax = mp->a[0];
  for (i = 1; i < g->N; i++)
    {
      ai = mp->a[i];
      if (ai > amax)
        amax = ai;
    }
  b = sqrt(amax/2.0);

  magG0 = g->ReciprocalVec[0][0]*g->ReciprocalVec[0][0] + g->ReciprocalVec[0][1]*g->ReciprocalVec[0][1] + g->ReciprocalVec[0][2]*g->ReciprocalVec[0][2];
  magG1 = g->ReciprocalVec[1][0]*g->ReciprocalVec[1][0] + g->ReciprocalVec[1][1]*g->ReciprocalVec[1][1] + g->ReciprocalVec[1][2]*g->ReciprocalVec[1][2];
  magG2 = g->ReciprocalVec[2][0]*g->ReciprocalVec[2][0] + g->ReciprocalVec[2][1]*g->ReciprocalVec[2][1] + g->ReciprocalVec[2][2]*g->ReciprocalVec[2][2];
  NG0 = 1+(int)(sqrt(MAXEXP*4.0*b*b/magG0));
  NG1 = 1+(int)(sqrt(MAXEXP*4.0*b*b/magG1));
  NG2 = 1+(int)(sqrt(MAXEXP*4.0*b*b/magG2));

  for (iG0 = -NG0; iG0 <= NG0; iG0++)
    for (iG1 = -NG1; iG1 <= NG1; iG1++)
      for (iG2 = -NG2; iG2 <= NG2; iG2++)
        {
          G[0] = ((double)iG0)*g->ReciprocalVec[0][0] + ((double)iG1)*g->ReciprocalVec[1][0] + ((double)iG2)*g->ReciprocalVec[2][0];
          G[1] = ((double)iG0)*g->ReciprocalVec[0][1] + ((double)iG1)*g->ReciprocalVec[1][1] + ((double)iG2)*g->ReciprocalVec[2][1];
          G[2] = ((double)iG0)*g->ReciprocalVec[0][2] + ((double)iG1)*g->ReciprocalVec[1][2] + ((double)iG2)*g->ReciprocalVec[2][2];
          GG = G[0]*G[0] + G[1]*G[1] + G[2]*G[2];

          if (GG > 1.0e-10)
            {
              common = (8.0*PI/g->Volume)/GG;
              sumCos = 0.0;
              sumSin = 0.0;

              for (i = 0; i < g->N; i++)
                {
                  phase = G[0]*g->R[3*i+0] + G[1]*g->R[3*i+1] + G[2]*g->R[3*i+2];
                  s = sin(phase);
                  c = cos(phase);
                  amp = exp(-0.25*GG/mp->a[i]);

                  phaseSin[i] = s;
                  phaseCos[i] = c;
                  atomAmp[i] = amp;

                  weightedCharge = mp->Q_out[i][0]*amp;
                  sumCos += weightedCharge*c;
                  sumSin += weightedCharge*s;
                }

              for (i = 0; i < g->N; i++)
                {
                  weightedCharge = mp->Q_out[i][0]*atomAmp[i]*common*(phaseSin[i]*sumCos - phaseCos[i]*sumSin);
                  F_B[3*i + 0] += weightedCharge*G[0];
                  F_B[3*i + 1] += weightedCharge*G[1];
                  F_B[3*i + 2] += weightedCharge*G[2];
                }
            }
        }

  for (i = 0; i < g->N; i++)
    Q[i] = mp->Q_out[i][0];
  electroMonopolePotential3D (g, mp, Q, V);

  for (i = 0; i < g->N; i++)
    for (n = 0; n < g->neighbourList[i].nNeighbour; n++)
      {
        j = g->neighbourList[i].neighbour[n];
        j0 = g->i[j];
        G[0] = V[i]*mp->dQ_out[i][0][n][0];
        G[1] = V[i]*mp->dQ_out[i][0][n][1];
        G[2] = V[i]*mp->dQ_out[i][0][n][2];

        F_Q[3*j0 + 0] -= G[0];
        F_Q[3*j0 + 1] -= G[1];
        F_Q[3*j0 + 2] -= G[2];

        if (j != j0)
          {
            dR[0] = g->R[3*j+0] - g->R[3*j0+0];
            dR[1] = g->R[3*j+1] - g->R[3*j0+1];
            dR[2] = g->R[3*j+2] - g->R[3*j0+2];
            for (k = 0; k < 3; k++)
              {
                l = (g->ReciprocalVec[k][0]*dR[0] + g->ReciprocalVec[k][1]*dR[1] + g->ReciprocalVec[k][2]*dR[2])/(2.0*PI);
                Flattice[k][0] += l*G[0];
                Flattice[k][1] += l*G[1];
                Flattice[k][2] += l*G[2];
              }
          }
      }

  for (i = 0; i < 3*g->N; i++)
    F[i] = F_Q[i] + F_B[i];

  free (phaseSin);
  free (phaseCos);
  free (atomAmp);
  free (F_Q);
  free (F_B);
  free (Q);
  free (V);

  return 0;
}

int
electroMonopoleHamiltonian3D (sparseGeom_t *g, multipole_t *mp, sparseMatrix_t *H, sparseMatrix_t *S)
{
  int i, j, j0, ji, ni, nj, nji, ki, kj, li, lj, Ni, Nj;
  double Rij[3], Rji[3], D[3], magD2;
  double *Q, *V0, V0ij;

  if (!electroMonopoleCheck3D (g, mp))
    {
      printf ("electroMonopoleHamiltonian3D: only valid for 3D periodic monopole-only electrostatics.\n");
      return 1;
    }

  Q = malloc (g->N*sizeof(double));
  V0 = malloc (g->N*sizeof(double));

  for (i = 0; i < g->N; i++)
    Q[i] = mp->Q_inp[0][i][0];
  electroMonopolePotential3D (g, mp, Q, V0);
  free (Q);

  if (S->v == NULL)
    {
      /* Orthogonal TB */
      for (i = 0; i < g->N; i++)
        {
          Ni = g->nOrb[i];
          for (ni = 0; ni < g->neighbourList[i].nNeighbour; ni++)
            {
              j = g->neighbourList[i].neighbour[ni];
              if (j == i)
                for (li = 0; li < Ni; li++)
                  {
                    ki = Ni*li + li;
                    H->v[i][ni][ki] = V0[i];
                  }
            }
        }
    }
  else
    {
      /* Non-orthogonal TB */
      for (i = 0; i < g->N; i++)
        {
          Ni = g->nOrb[i];
          for (ni = 0; ni < g->neighbourList[i].nNeighbour; ni++)
            {
              j = g->neighbourList[i].neighbour[ni];
              j0 = g->i[j];
              Nj = g->nOrb[j];
              V0ij = 0.5*(V0[i] + V0[j0]);
              for (ki = 0; ki < Ni*Nj; ki++)
                H->v[i][ni][ki] = V0ij*S->v[i][ni][ki];
            }
        }
    }

  free (V0);

  return 0;
}

//##################################
// End of new isolated 3D periodic monopole-only electrostatics implementation.
//##################################
