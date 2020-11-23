/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributead under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_spherharm_unittests.h"
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "fix_adapt.h"
#include "error.h"
#include "memory.h"

using namespace LAMMPS_NS;
/* ---------------------------------------------------------------------- */

AtomVecSpherharmUnitTests::AtomVecSpherharmUnitTests(LAMMPS *lmp) : AtomVecSpherharm(lmp)
{
  ellipsoidshape = nullptr;
}

AtomVecSpherharmUnitTests::~AtomVecSpherharmUnitTests()
{
  memory->sfree(ellipsoidshape);
}

/* ----------------------------------------------------------------------
   process sub-style args
------------------------------------------------------------------------- */

void AtomVecSpherharmUnitTests::process_args(int narg, char **arg) {

  AtomVecSpherharm::process_args(narg, arg);

  for (int i=0; i<nshtypes; i++){
    pinertia_byshape[i][0] /=441.0;
    pinertia_byshape[i][1] /=441.0;
    pinertia_byshape[i][2] /=441.0;
  }

  MPI_Bcast(&(pinertia_byshape[0][0]), nshtypes * 3, MPI_DOUBLE, 0, world);

  memory->create(ellipsoidshape, nshtypes, 3, "AtomVecSpherharmUnitTests:ellipsoidshape");
}

void AtomVecSpherharmUnitTests::get_shape(int i, double &shapex, double &shapey, double &shapez)
{
  ellipsoidshape[0][0] = 0.5;
  ellipsoidshape[0][1] = 0.5;
  ellipsoidshape[0][2] = 2.5;

  shapex = ellipsoidshape[shtype[i]][0];
  shapey = ellipsoidshape[shtype[i]][1];
  shapez = ellipsoidshape[shtype[i]][2];
}