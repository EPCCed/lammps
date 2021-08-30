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
FixStyle(particle_meshp,FixParticleMeshP);
// clang-format on
#else

#ifndef LMP_FIX_PARTICLE_MESHP_H
#define LMP_FIX_PARTICLE_MESHP_H

#include "fix.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include "types.h"

namespace LAMMPS_NS {

    class FixParticleMeshP : public Fix {
    public:
        FixParticleMeshP(class LAMMPS *, int, char **);
        ~FixParticleMeshP();
        int setmask();
        void setup(int);
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

        double **mesh; //~ Public to allow access from pairstyles

    private:
        typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
        typedef Kernel::Point_3 Point3;
        typedef Kernel::Vector_3 Vector;
        typedef std::tuple<int, Point3, Vector> PointVectorTuple;
        typedef std::vector<PointVectorTuple> PointListwithindex;
        void octreefrompoints(PointListwithindex points, Octree& octree,double min_radius = -1);
    };

}    // namespace LAMMPS_NS

#endif
#endif