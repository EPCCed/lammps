/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------
   Contributing authors: James Young (UoE)
                         Kevin Hanley (UoE)

   Please cite the related publication:
   TBC
------------------------------------------------------------------------- */

#include "compute_erotate_spherharm.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include "math_extra.h"

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeERotateSpherharm::ComputeERotateSpherharm(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute erotate/sphere command");

  scalar_flag = 1;
  extscalar = 1;

  // error check

  if (!atom->spherharm_flag)
    error->all(FLERR,"Compute erotate/sphere requires atom style spherharm");
}

/* ---------------------------------------------------------------------- */

void ComputeERotateSpherharm::init()
{
  pfactor = 0.5 * force->mvv2e;
}

/* ---------------------------------------------------------------------- */

double ComputeERotateSpherharm::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double **angmom = atom->angmom;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **pinertia_byshape = atom->pinertia_byshape;
  int *shtype = atom->shtype;
  double **quat = atom->quat;
  int ishtype;
  double wbody[3];
  double rot[3][3];
  
  // sum rotational energy for each particle, use angular momentum as angular velocity lags by half a time step.

  double erotate = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      ishtype = shtype[i];
      MathExtra::quat_to_mat(quat[i],rot);
      MathExtra::transpose_matvec(rot,angmom[i],wbody);
      erotate += (wbody[0]*wbody[0]/pinertia_byshape[ishtype][0]) + // E_rot = 0.5*I.w^2, w=L/I => E_rot = 0.5(L/I)^2.I
                (wbody[1]*wbody[1]/pinertia_byshape[ishtype][1]) +
                (wbody[2]*wbody[2]/pinertia_byshape[ishtype][2]);
    }
  }

  scalar = 0.0;
  MPI_Allreduce(&erotate,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  scalar *= pfactor;
  return scalar;
}
