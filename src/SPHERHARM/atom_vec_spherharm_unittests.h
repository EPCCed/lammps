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

#ifdef ATOM_CLASS

AtomStyle(spherharmunittests,AtomVecSpherharmUnitTests)

#else

#ifndef LMP_ATOM_VEC_SPHERHARMUNITTESTS_H
#define LMP_ATOM_VEC_SPHERHARMUNITTESTS_H

#include "atom_vec_spherharm.h"

namespace LAMMPS_NS {

class AtomVecSpherharmUnitTests : public AtomVecSpherharm {
 public:

  // Mandatory LAMMPS methods
  AtomVecSpherharmUnitTests(class LAMMPS *);
  void process_args(int, char **);
  ~AtomVecSpherharmUnitTests();

  // Public methods required to access per-shape arrays
  void get_shape(int, double &, double &, double &);            // FOR ELLIPSOID TEST ONLY

 private:
  double **ellipsoidshape;    // FOR ELLIPSOID TEST ONLY

  void check_rotations(int, int);// Calculate the expansion factors of each shape using the quadrature points
  void check_sphere_normals();
  void check_ellipsoid_normals();
  void get_cog();
  void dump_ply();
  void dump_shapenormals();
  void compare_areas();
  void validate_rotation();
  void spher_sector_volumetest(int num_pole_quad, double iang);
  void spher_cap_volumetest(int num_pole_quad, double iang);
  double back_calc_coeff(int n, int m, int num_pole_quad);
  void boost_test();
  void volumetest_boost_test();
  void surfacearea_boost_test();
  void surfarea_int_tests(int num_pole_quad, double iang);
  void sphere_line_intersec_tests();
  void print_normals();
  double get_shape_radius_compensated_boost(int sht, double theta, double phi);
};

}

#endif
#endif
