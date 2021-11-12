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

#include "compute_property_spherharmatom.h"
#include <cmath>
#include <cstring>
#include "math_extra.h"
#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
#include "atom_vec_body.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePropertySpherharmAtom::ComputePropertySpherharmAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  index(nullptr), pack_choice(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal compute property/atom command");

  peratom_flag = 1;
  nvalues = narg - 3;
  if (nvalues == 1) size_peratom_cols = 0;
  else size_peratom_cols = nvalues;

  // parse input values
  // customize a new keyword by adding to if statement

  pack_choice = new FnPtrPack[nvalues];
  index = new int[nvalues];

  int i;
  for (int iarg = 3; iarg < narg; iarg++) {
    i = iarg-3;

    if (strcmp(arg[iarg],"quatw") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertySpherharmAtom::pack_quatw;
    } else if (strcmp(arg[iarg],"quati") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertySpherharmAtom::pack_quati;
    } else if (strcmp(arg[iarg],"quatj") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertySpherharmAtom::pack_quatj;
    } else if (strcmp(arg[iarg],"quatk") == 0) {
      avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
      avec_body = (AtomVecBody *) atom->style_match("body");
      if (!avec_ellipsoid && !avec_body)
        error->all(FLERR,"Compute property/atom for "
                   "atom property that isn't allocated");
      pack_choice[i] = &ComputePropertySpherharmAtom::pack_quatk;
    }
  }

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputePropertySpherharmAtom::~ComputePropertySpherharmAtom()
{
  delete [] pack_choice;
  delete [] index;
  memory->destroy(vector_atom);
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

void ComputePropertySpherharmAtom::init()
{
  avec_ellipsoid = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  avec_line = (AtomVecLine *) atom->style_match("line");
  avec_tri = (AtomVecTri *) atom->style_match("tri");
  avec_body = (AtomVecBody *) atom->style_match("body");
}

/* ---------------------------------------------------------------------- */

void ComputePropertySpherharmAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow vector or array if necessary

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    if (nvalues == 1) {
      memory->destroy(vector_atom);
      memory->create(vector_atom,nmax,"property/atom:vector");
    } else {
      memory->destroy(array_atom);
      memory->create(array_atom,nmax,nvalues,"property/atom:array");
    }
  }

  // fill vector or array with per-atom values

  if (nvalues == 1) {
    buf = vector_atom;
    (this->*pack_choice[0])(0);
  } else {
    if (nmax) buf = &array_atom[0][0];
    else buf = nullptr;
    for (int n = 0; n < nvalues; n++)
      (this->*pack_choice[n])(n);
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputePropertySpherharmAtom::memory_usage()
{
  double bytes = nmax*nvalues * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   one method for every keyword compute property/atom can output
   the atom property is packed into buf starting at n with stride nvalues
   customize a new keyword by adding a method
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void ComputePropertySpherharmAtom::pack_quatw(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[0];
      else buf[n] = 0.0;
      n += nvalues;
    }

  } else {
    AtomVecBody::Bonus *bonus = avec_body->bonus;
    int *body = atom->body;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && body[i] >= 0)
        buf[n] = bonus[body[i]].quat[0];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertySpherharmAtom::pack_quati(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[1];
      else buf[n] = 0.0;
      n += nvalues;
    }

  } else {
    AtomVecBody::Bonus *bonus = avec_body->bonus;
    int *body = atom->body;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && body[i] >= 0)
        buf[n] = bonus[body[i]].quat[1];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertySpherharmAtom::pack_quatj(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[2];
      else buf[n] = 0.0;
      n += nvalues;
    }

  } else {
    AtomVecBody::Bonus *bonus = avec_body->bonus;
    int *body = atom->body;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && body[i] >= 0)
        buf[n] = bonus[body[i]].quat[2];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertySpherharmAtom::pack_quatk(int n)
{
  if (avec_ellipsoid) {
    AtomVecEllipsoid::Bonus *bonus = avec_ellipsoid->bonus;
    int *ellipsoid = atom->ellipsoid;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && ellipsoid[i] >= 0)
        buf[n] = bonus[ellipsoid[i]].quat[3];
      else buf[n] = 0.0;
      n += nvalues;
    }

  } else {
    AtomVecBody::Bonus *bonus = avec_body->bonus;
    int *body = atom->body;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if ((mask[i] & groupbit) && body[i] >= 0)
        buf[n] = bonus[body[i]].quat[3];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

