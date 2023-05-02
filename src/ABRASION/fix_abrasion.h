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

#ifdef FIX_CLASS
// clang-format off
FixStyle(abrasion,FixAbrasion);
// clang-format on
#else

#ifndef LMP_FIX_ABRASION_H
#define LMP_FIX_ABRASION_H

#include "fix.h"

namespace LAMMPS_NS {

    class FixAbrasion : public Fix {
    public:
      FixAbrasion(class LAMMPS *, int, char **);
      ~FixAbrasion();
      int setmask() override;
      void init() override;
      void init_list(int, class NeighList *) override;
      void post_force(int) override;
      double memory_usage() override;
      void grow_arrays(int) override;
      void copy_arrays(int, int, int) override;
      void set_arrays(int) override;
      int pack_exchange(int, double *) override;
      int unpack_exchange(int, double *) override;
      int pack_restart(int, double *) override;
      void unpack_restart(int, int) override;
      int maxsize_restart() override;
      int size_restart(int) override;

      double **vertexdata; //~ Public to allow access from pairstyles
    private:
      double pf;                          // hardness
      double mu;                          // a constant used to calculate the shear hardness
      class NeighList *list;
      void areas_and_normals();
      bool gap_is_shrinking(int, int, double[3], double[3], double*);
      void displacement_of_atom(int, double, double, double[3], double[3]);
    };

}    // namespace LAMMPS_NS

#endif
#endif
