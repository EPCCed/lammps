/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(property/spherharmatom,ComputePropertySpherharmAtom)

#else

#ifndef LMP_COMPUTE_PROPERTY_SPHERHARMATOM_H
#define LMP_COMPUTE_PROPERTY_SPHERHARMATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputePropertySpherharmAtom : public Compute {
 public:
  ComputePropertySpherharmAtom(class LAMMPS *, int, char **);
  ~ComputePropertySpherharmAtom();
  void init();
  void compute_peratom();
  double memory_usage();

 private:
  int nvalues;
  int nmax;
  int *index;
  double *buf;
  class AtomVecEllipsoid *avec_ellipsoid;
  class AtomVecLine *avec_line;
  class AtomVecTri *avec_tri;
  class AtomVecBody *avec_body;

  typedef void (ComputePropertySpherharmAtom::*FnPtrPack)(int);
  FnPtrPack *pack_choice;              // ptrs to pack functions

  void pack_quatw(int);
  void pack_quati(int);
  void pack_quatj(int);
  void pack_quatk(int);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute property/atom for atom property that isn't allocated

Self-explanatory.

E: Compute property/atom integer vector does not exist

The command is accessing a vector added by the fix property/atom
command, that does not exist.

E: Compute property/atom floating point vector does not exist

The command is accessing a vector added by the fix property/atom
command, that does not exist.

E: Invalid keyword in compute property/atom command

Self-explanatory.

*/
