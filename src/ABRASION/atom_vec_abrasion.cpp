/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_abrasion.h"
#include "atom.h"

//#include "fix.h"
//#include "fix_adapt.h"
//#include "modify.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecAbrasion::AtomVecAbrasion(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::MOLECULAR;
  bonds_allow = angles_allow = 1;
  mass_type = PER_ATOM;

  //  atom->sphere_flag = 1;
  //  atom->molecule_flag = 1;
  atom->rmass_flag = 1;
  atom->radius_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = {"radius",      "rmass",      "area",        "normal",
                 "molecule",    "num_bond",   "bond_type",   "bond_atom",
                 "num_angle",   "angle_type", "angle_atom1", "angle_atom2",
                 "angle_atom3", "nspecial",   "special"};
  fields_copy = {"radius",      "rmass",      "area",        "normal",
                 "molecule",    "num_bond",   "bond_type",   "bond_atom",
                 "num_angle",   "angle_type", "angle_atom1", "angle_atom2",
                 "angle_atom3", "nspecial",   "special"};
  fields_border = {"radius", "rmass", "molecule", "area", "normal"};
  fields_border_vel = {"radius", "rmass", "molecule", "area", "normal"};
  fields_exchange = {"radius",      "rmass",       "area",       "normal",
                     "molecule",    "num_bond",   "bond_type",   "bond_atom",
                     "num_angle",   "angle_type", "angle_atom1", "angle_atom2",
                     "angle_atom3", "nspecial",   "special"};
  fields_restart = {"radius",     "rmass",
                    "molecule",   "num_bond",    "bond_type",   "bond_atom",  "num_angle",
                    "angle_type", "angle_atom1", "angle_atom2", "angle_atom3"};
  fields_create = {"radius",   "rmass",     "area",     "normal",
                   "molecule", "num_bond", "num_angle", "nspecial"};
  fields_data_atom = {"id", "type", "molecule", "radius", "x"};
  fields_data_vel = {"id", "v"};

  setup_fields();

  bond_per_atom = angle_per_atom = 0;
  bond_negative = angle_negative = nullptr;
}

/* ---------------------------------------------------------------------- */

AtomVecAbrasion::~AtomVecAbrasion()
{
  delete[] bond_negative;
  delete[] angle_negative;
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecAbrasion::grow_pointers()
{
  radius = atom->radius;
  rmass = atom->rmass;
  num_bond = atom->num_bond;
  bond_type = atom->bond_type;
  num_angle = atom->num_angle;
  angle_type = atom->angle_type;
  nspecial = atom->nspecial;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecAbrasion::create_atom_post(int ilocal)
{
  radius[ilocal] = 0.5;
  rmass[ilocal] = 0.5;
}

/* ----------------------------------------------------------------------
   modify values for AtomVec::pack_restart() to pack
------------------------------------------------------------------------- */

void AtomVecAbrasion::pack_restart_pre(int ilocal)
{
  // insure negative vectors are needed length

  if (bond_per_atom < atom->bond_per_atom) {
    delete[] bond_negative;
    bond_per_atom = atom->bond_per_atom;
    bond_negative = new int[bond_per_atom];
  }
  if (angle_per_atom < atom->angle_per_atom) {
    delete[] angle_negative;
    angle_per_atom = atom->angle_per_atom;
    angle_negative = new int[angle_per_atom];
  }

  // flip any negative types to positive and flag which ones

  any_bond_negative = 0;
  for (int m = 0; m < num_bond[ilocal]; m++) {
    if (bond_type[ilocal][m] < 0) {
      bond_negative[m] = 1;
      bond_type[ilocal][m] = -bond_type[ilocal][m];
      any_bond_negative = 1;
    } else
      bond_negative[m] = 0;
  }

  any_angle_negative = 0;
  for (int m = 0; m < num_angle[ilocal]; m++) {
    if (angle_type[ilocal][m] < 0) {
      angle_negative[m] = 1;
      angle_type[ilocal][m] = -angle_type[ilocal][m];
      any_angle_negative = 1;
    } else
      angle_negative[m] = 0;
  }
}

/* ----------------------------------------------------------------------
   unmodify values packed by AtomVec::pack_restart()
------------------------------------------------------------------------- */

void AtomVecAbrasion::pack_restart_post(int ilocal)
{
  // restore the flagged types to their negative values

  if (any_bond_negative) {
    for (int m = 0; m < num_bond[ilocal]; m++)
      if (bond_negative[m]) bond_type[ilocal][m] = -bond_type[ilocal][m];
  }

  if (any_angle_negative) {
    for (int m = 0; m < num_angle[ilocal]; m++)
      if (angle_negative[m]) angle_type[ilocal][m] = -angle_type[ilocal][m];
  }
}

/* ----------------------------------------------------------------------
   initialize other atom quantities after AtomVec::unpack_restart()
------------------------------------------------------------------------- */

void AtomVecAbrasion::unpack_restart_init(int ilocal)
{
  nspecial[ilocal][0] = 0;
  nspecial[ilocal][1] = 0;
  nspecial[ilocal][2] = 0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecAbrasion::data_atom_post(int ilocal)
{
  num_bond[ilocal] = 0;
  num_angle[ilocal] = 0;
  nspecial[ilocal][0] = 0;
  nspecial[ilocal][1] = 0;
  nspecial[ilocal][2] = 0;
}
