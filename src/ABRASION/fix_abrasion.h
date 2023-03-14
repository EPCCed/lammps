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
        int setmask();
        void pre_force(int);
        void post_force(int);
        double memory_usage();
        void grow_arrays(int);
        void copy_arrays(int, int, int);
        void set_arrays(int);
        int pack_exchange(int, double *);
        int unpack_exchange(int, double *);
        int pack_restart(int, double *);
        void unpack_restart(int, int);
        int maxsize_restart();
        int size_restart(int);

        double **vertexdata; //~ Public to allow access from pairstyles
    private:
        void areas_and_normals();
    };

}    // namespace LAMMPS_NS

#endif
#endif
