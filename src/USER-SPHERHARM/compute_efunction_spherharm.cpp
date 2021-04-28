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

#include "compute_efunction_spherharm.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include "math_extra.h"

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

ComputeEFunctionSpherharm::ComputeEFunctionSpherharm(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute erotate/sphere command");

  scalar_flag = 1;
  extscalar = 1;
  workstore = 0.0;

  // error check

  if (!atom->spherharm_flag)
    error->all(FLERR,"Compute erotate/sphere requires atom style spherharm");
}

/* ---------------------------------------------------------------------- */

void ComputeEFunctionSpherharm::init()
{
  pfactor = force->mvv2e; // No 0.5 here because not explicitly calculating 0.5mv^2, but are in units of mv^2
  dw_back = dw_back_nextstep = 0.0;
}

/* ---------------------------------------------------------------------- */
// Incremental work done by potential dw = -(F.vdt + T.wdt) = -dt(F.v + T.w)
// The work done by the potential over a time step is a combination of that done from the "back half" t -> t+dt/2 and
// the "front half" t+dt/2->t+dt of the time step. At the end of the time step, i.e t+dt, it is not possible
// to calculate the "back half" of the work done, so this must be carried forward from the previous time step.
/* ---------------------------------------------------------------------- */
double ComputeEFunctionSpherharm::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  double **angmom = atom->angmom;
  double **omega = atom->omega;
  double **f = atom->f;
  double **v = atom->v;
  double **torque = atom->torque;
  double **quat = atom->quat;
  double **inertia = atom->pinertia_byshape;
  int *mask = atom->mask;
  int *shtype = atom->shtype;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double dt = update->dt;
  int itype, ishtype;
  double fv, mw, temp, dt2, dtf, dtfm;
  double vhalf[3], whalf[3], angmomhalf[3], qfoo[4];

  dt2 = dt/2.0;
  dtf = 0.5 * update->dt * force->ftm2v;
  dw_back = dw_back_nextstep; // Getting the back half from the contribution calculated in the previous time step
  double efunction = 0.0;
  double efuture = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      itype = type[i];
      dtfm = dtf / mass[itype];
      ishtype = shtype[i];

      // "front half"
      vhalf[0] = v[i][0] - dtfm * f[i][0]; // Pushing velocity back half a step
      vhalf[1] = v[i][1] - dtfm * f[i][1];
      vhalf[2] = v[i][2] - dtfm * f[i][2];
      fv = MathExtra::dot3(f[i], vhalf);
      mw = MathExtra::dot3(torque[i], omega[i]);
      efunction += -dt2*(fv+mw); // Getting the energy contribution from front half of timestep

      // "back half for next time step"
      vhalf[0] = v[i][0] + dtfm * f[i][0]; // Pushing velocity forward half a step
      vhalf[1] = v[i][1] + dtfm * f[i][1];
      vhalf[2] = v[i][2] + dtfm * f[i][2];
      fv = MathExtra::dot3(f[i], vhalf);
      angmomhalf[0] = angmom[i][0] + dtf * torque[i][0]; // Pushing ang mom forward half a step
      angmomhalf[1] = angmom[i][1] + dtf * torque[i][1];
      angmomhalf[2] = angmom[i][2] + dtf * torque[i][2];
      qfoo[0] = quat[i][0];
      qfoo[1] = quat[i][1];
      qfoo[2] = quat[i][2];
      qfoo[3] = quat[i][3];
      MathExtra::mq_to_omega(angmomhalf,quat[i],inertia[ishtype],whalf);
      MathExtra::richardson(qfoo,angmomhalf,whalf,inertia[ishtype],dt2); // Pushing ang velocity forward half a step
      mw = MathExtra::dot3(torque[i], whalf);
      efuture += -dt2*(fv+mw); // Getting the energy contribution from front half of timestep
    }
  }

  temp = 0.0;
  MPI_Allreduce(&efunction,&temp,1,MPI_DOUBLE,MPI_SUM,world);
  temp *= pfactor;
  workstore += (long double)(temp + dw_back);
  scalar = double(workstore);
  //scalar += temp + dw_back;

  temp = 0.0;
  MPI_Allreduce(&efuture,&temp,1,MPI_DOUBLE,MPI_SUM,world);
  temp *= pfactor;
  dw_back_nextstep = temp;

  return scalar;
}
