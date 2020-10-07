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

#include "atom_vec_shfew.h"
#include <cstring>
#include <iostream>
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "fix_adapt.h"
#include "math_const.h"
#include "error.h"
#include "memory.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

AtomVecShfew::AtomVecShfew(LAMMPS *lmp) : AtomVec(lmp)
{
  mass_type = 0;
  molecular = 0;

  atom->sphere_flag = 1; // <- JUST FOR TESTING, NEED TO DELETE LATER
  atom->spherharm_flag = 1;
  atom->radius_flag = atom->rmass_flag = atom->omega_flag =
    atom->torque_flag = atom -> angmom_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "radius rmass omega torque shtype angmom quat inertia";
  fields_copy = (char *) "radius rmass omega shtype angmom quat inertia";
  fields_comm = (char *) "";
  fields_comm_vel = (char *) "omega angmom";
  fields_reverse = (char *) "torque";
  fields_border = (char *) "radius rmass shtype";
  fields_border_vel = (char *) "radius rmass omega shtype angmom";
  fields_exchange = (char *) "radius rmass omega shtype angmom";
  fields_restart = (char *) "radius rmass omega shtype angmom";
  fields_create = (char *) "radius rmass omega shtype angmom";
  fields_data_atom = (char *) "id type radius rmass x shtype quat inertia";
  fields_data_vel = (char *) "id v omega angmom";
}

/* ----------------------------------------------------------------------
   process sub-style args
   optional arg = 0/1 for static/dynamic particle radii
------------------------------------------------------------------------- */

void AtomVecShfew::process_args(int narg, char **arg)
{
  if (narg != 0 && narg != 1)
    error->all(FLERR,"Illegal atom_style shfewnc command");

  radvary = 0;
  if (narg == 1) {
    radvary = utils::numeric(FLERR,arg[0],true,lmp);
    if (radvary < 0 || radvary > 1)
      error->all(FLERR,"Illegal atom_style shfewnc command");
  }

  // dynamic particle radius and mass must be communicated every step

  if (radvary) {
    fields_comm = (char *) "radius rmass";
    fields_comm_vel = (char *) "radius rmass omega";
  }

  // delay setting up of fields until now

  setup_fields();
}

/* ---------------------------------------------------------------------- */

void AtomVecShfew::init()
{
  AtomVec::init();

  // check if optional radvary setting should have been set to 1

  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"adapt") == 0) {
      FixAdapt *fix = (FixAdapt *) modify->fix[i];
      if (fix->diamflag && radvary == 0)
        error->all(FLERR,"Fix adapt changes particle radii "
                   "but atom_style sphere is not dynamic");
    }
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecShfew::grow_pointers()
{
  radius = atom->radius;
  rmass = atom->rmass;
  omega = atom->omega;
  shtype = atom->shtype;
  angmom = atom -> angmom;
  inertia = atom -> inertia;
  quat = atom -> quat;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecShfew::create_atom_post(int ilocal)
{
  radius[ilocal] = 0.5;
  rmass[ilocal] = 4.0*MY_PI/3.0 * 0.5*0.5*0.5;
  shtype[ilocal] = 0;
  quat[ilocal][0] = 1.0;
  quat[ilocal][1] = 0.0;
  quat[ilocal][2] = 0.0;
  quat[ilocal][3] = 0.0;
  inertia[ilocal][0] = 0.0;
  inertia[ilocal][1] = 0.0;
  inertia[ilocal][2] = 0.0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecShfew::data_atom_post(int ilocal)
{
  radius_one = 0.5 * atom->radius[ilocal];
  radius[ilocal] = radius_one;
  if (radius_one > 0.0)
    rmass[ilocal] *= 4.0*MY_PI/3.0 * radius_one*radius_one*radius_one;

  if (rmass[ilocal] <= 0.0)
    error->one(FLERR,"Invalid density in Atoms section of data file");

  omega[ilocal][0] = 0.0;
  omega[ilocal][1] = 0.0;
  omega[ilocal][2] = 0.0;
}

/* ----------------------------------------------------------------------
   modify values for AtomVec::pack_data() to pack
------------------------------------------------------------------------- */

void AtomVecShfew::pack_data_pre(int ilocal)
{
  radius_one = radius[ilocal];
  rmass_one = rmass[ilocal];

  radius[ilocal] *= 2.0;
  if (radius_one!= 0.0)
    rmass[ilocal] =
      rmass_one / (4.0*MY_PI/3.0 * radius_one*radius_one*radius_one);
}

/* ----------------------------------------------------------------------
   unmodify values packed by AtomVec::pack_data()
------------------------------------------------------------------------- */

void AtomVecShfew::pack_data_post(int ilocal)
{
  radius[ilocal] = radius_one;
  rmass[ilocal] = rmass_one;
}
