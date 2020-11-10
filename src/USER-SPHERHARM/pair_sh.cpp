/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_sh.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"

#include "atom_vec_shperatom.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairSH::PairSH(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  writedata = 1;
  centroidstressflag = 1;

  //  JY Added

  // Single steps are for force and energy of a single pairwise interaction between 2 atoms
  // Energy calculation not enabled, as we don't yet have pairwise potential
  single_enable = 0;
  restartinfo = 0; // Not figured out how to do this yet
  writedata = 0; // Ditto
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairSH::~PairSH()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairSH::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

//void PairSH::allocate()
//{
//  allocated = 1;
//  int n = atom->ntypes;
//
//  memory->create(setflag,n+1,n+1,"pair:setflag");
//  for (int i = 1; i <= n; i++)
//    for (int j = i; j <= n; j++)
//      setflag[i][j] = 0;
//
//  memory->create(cutsq,n+1,n+1,"pair:cutsq");
//  memory->create(cut,n+1,n+1,"pair:cut");
//}

void PairSH::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag,n+1,n+1,"pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++)
            setflag[i][j] = 0;

    memory->create(cutsq,n+1,n+1,"pair:cutsq");

    memory->create(cut,n+1,n+1,"pair:cut");
    memory->create(epsilon,n+1,n+1,"pair:epsilon");
    memory->create(sigma,n+1,n+1,"pair:sigma");
    memory->create(lj1,n+1,n+1,"pair:lj1");
    memory->create(lj2,n+1,n+1,"pair:lj2");
    memory->create(lj3,n+1,n+1,"pair:lj3");
    memory->create(lj4,n+1,n+1,"pair:lj4");
    memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
   JY - Not defining a global cut off, as this must come from the
   atom style, where the maximum particle radius is stored
------------------------------------------------------------------------- */

void PairSH::settings(int narg, char **arg) {
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   JY - Only type pairs are defined here, no other parameters. The global
   cutoff is taken from the atom style here.
------------------------------------------------------------------------- */

void PairSH::coeff(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double cut_one = cut_global;
  avec->get_cut_global(cut_one);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSH::init_style()
{
  avec = (AtomVecShperatom *) atom->style_match("shperatom");
  if (!avec) error->all(FLERR,"Pair SH requires atom style shperatom");

  neighbor->request(this,instance_me);
  cut_respa = NULL;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   JY - Need to set up for different types, although both types must use the
   spherical harmonic atom style. Both would share the same cut so no mixing
   would be required here. The only mixing would be in the coefficients used
   in the contact model, i.e stiffness, but this will need to be explored later
------------------------------------------------------------------------- */

double PairSH::init_one(int i, int j)
{
  // No epsilon and no sigma used for the spherical harmonic atom style
  if (setflag[i][j] == 0) {
    cut[i][j] = cut_global;
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

 // Offset_flag is true if shift value = yes in the pair_modify command. "The
 // shift keyword determines whether a Lennard-Jones potential is shifted at
 // its cutoff to 0.0". Not applicable for this pairstyle
//  if (offset_flag && (cut[i][j] > 0.0)) {
//    double ratio = sigma[i][j] / cut[i][j];
//    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
//  } else offset[i][j] = 0.0;
  offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

//  if (cut_respa && cut[i][j] < cut_respa[3])
//    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

//    When the tail keyword is set to yes, certain pair styles will add a long-range
//    VanderWaals tail “correction” to the energy and pressure. These corrections are
//    bookkeeping terms which do not affect dynamics, unless a constant-pressure simulation
//    is being performed. See the doc page for individual styles to see which support this option.
//    These corrections are included in the calculation and printing of thermodynamic
//    quantities (see the thermo_style command). Their effect will also be included in
//    constant NPT or NPH simulations where the pressure influences the simulation box
//    dimensions (e.g. the fix npt and fix nph commands). The formulas used for the
//    long-range corrections come from equation 5 of (Sun).

//  if (tail_flag) {
//    int *type = atom->type;
//    int nlocal = atom->nlocal;
//
//    double count[2],all[2];
//    count[0] = count[1] = 0.0;
//    for (int k = 0; k < nlocal; k++) {
//      if (type[k] == i) count[0] += 1.0;
//      if (type[k] == j) count[1] += 1.0;
//    }
//    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);
//
//    double sig2 = sigma[i][j]*sigma[i][j];
//    double sig6 = sig2*sig2*sig2;
//    double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
//    double rc6 = rc3*rc3;
//    double rc9 = rc3*rc6;
//    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
//      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
//    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
//      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
//  }

  return cut[i][j];
}


