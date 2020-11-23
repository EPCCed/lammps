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

AtomStyle(spherharm,AtomVecSpherharm)

#else

#ifndef LMP_ATOM_VEC_SPHERHARM_H
#define LMP_ATOM_VEC_SPHERHARM_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecSpherharm : public AtomVec {
 public:

  // Mandatory LAMMPS methods
  AtomVecSpherharm(class LAMMPS *);
  void process_args(int, char **);
  void init();
  void grow_pointers();
  void create_atom_post(int);
  void data_atom_post(int);
  void pack_data_pre(int);
  void pack_data_post(int);
  ~AtomVecSpherharm();

  // Public methods required to access per-shape arrays
  double** get_quadrature_angs(){return &angles[0];}
  double* get_max_rads(){return &maxrad_byshape[0];}
  double** get_quat_init(){return &quatinit_byshape[0];}
  double** get_pinertia_init(){return &pinertia_byshape[0];}
  int check_contact(int, double, double, double, double &);
  void get_shape(int, double &, double &, double &);            // FOR ELLIPSOID TEST ONLY
  double** get_quadrature_rads(int &num_quad2){
    num_quad2 = num_quadrature*num_quadrature;
    return &quad_rads_byshape[0];
  }

// private:
  protected:
  // per-atom arrays
  double **omega;
  int *shtype;                 // Links atom to the SH shape type that it uses
  double **angmom;
  double **quat;               // Current quat of the atom

  // per-shape arrays
  double **shcoeffs_byshape;   // Array of coefficients for each shape
  double **pinertia_byshape;   // Principle inertia for each shape
  double **quatinit_byshape;   // Initial quaternion for each shape (pricinple axis rotation from global axis)
  double **expfacts_byshape;   // The expansion factors for each shape, each SH degree has an expansion factor
  double **quad_rads_byshape;  // Radii at each point of guassian quadrature, for each shape (index is [shape][point])
  double *maxrad_byshape;      // The maximum radius of each shape at the maximum SH degree (maxshexpan)

  // Gaussian quadrature arrays
  int num_quadrature;         // Order of quadrature used (used defined in input file)
  double **angles;            // Array of (theta,phi) angles for each point of quadrature (same for all shapes)
  double *weights;            // Weights of gaussian quadrature (same for all shapes)

  // Global SH properties
  int maxshexpan;             // Maximum degree of the shperical harmonic expansion
  int nshtypes;               // Number of spherical harmonic shapes

  double **ellipsoidshape;    // FOR ELLIPSOID TEST ONLY

  void read_sh_coeffs(char *, int); // Reads the spherical harmonic coefficients from file
  void get_quadrature_values();     // Get the gaussian quadrature angles and weights
  void getI();                      // Calculate the inertia of each shape
  void calcexpansionfactors();      // Calculate the expansion factors of each shape using a regular grid
  void calcexpansionfactors_gauss();// Calculate the expansion factors of each shape using the quadrature points
};

}

#endif
#endif
