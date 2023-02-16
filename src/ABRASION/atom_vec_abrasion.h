/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(abrasion,AtomVecAbrasion);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_ABRASION_H
#define LMP_ATOM_VEC_ABRASION_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecAbrasion : virtual public AtomVec {
 public:
  AtomVecAbrasion(class LAMMPS *);
  ~AtomVecAbrasion() override;

  void grow_pointers() override;
  void create_atom_post(int) override;
  void pack_restart_pre(int) override;
  void pack_restart_post(int) override;
  void unpack_restart_init(int) override;
  void data_atom_post(int) override;

 protected:
  double *radius, *rmass;

  int *num_bond, *num_angle;
  int **bond_type, **angle_type;
  int **nspecial;

  int any_bond_negative, any_angle_negative;
  int bond_per_atom, angle_per_atom;
  int *bond_negative, *angle_negative;
};

}    // namespace LAMMPS_NS

#endif
#endif
