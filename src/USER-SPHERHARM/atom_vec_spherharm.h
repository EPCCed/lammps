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

  // Public methods required to access per-shape arrays [TO BE DELETED, CURRENTLY FOR TOY PAIR STYLE]
  double** get_quadrature_angs(){return &angles[0];}
  double** get_quadrature_rads(int &num_quad2){
    num_quad2 = num_quadrature*num_quadrature;
    return &quad_rads_byshape[0];
  }

  // Public methods used for contact detection. These are called by the pair_style and ensure that shcoeffs_byshape and
  // expfacts_byshape remain local to the atom style.
  double get_shape_radius(int sht, double theta, double phi); // Get the shape radius given theta and phi
  double get_shape_radius_and_normal(int sht, double theta, double phi, double rnorm[3]); // As above, with unit normal
  int check_contact(int, double, double, double, double &); // Check for contact given shape, theta, phi, and distance

 protected:
  // per-atom arrays
  double **omega;              // Angular velocity
  int *shtype;                 // Links atom to the SH shape type that it uses
  double **angmom;             // Angular momentum
  double **quat;               // Current quat of the atom

  // per-shape arrays
  double **shcoeffs_byshape;   // Array of coefficients for each shape
  double **pinertia_byshape;   // Principle inertia for each shape
  double **quatinit_byshape;   // Initial quaternion for each shape (pricinple axis rotation from global axis)
  double **expfacts_byshape;   // The expansion factors for each shape, each SH degree has an expansion factor
  double *maxrad_byshape;      // The maximum radius of each shape at the maximum SH degree (maxshexpan)
  double **quad_rads_byshape;  // Radii at each point of guassian quadrature, for each shape (index is [shape][point])

  // Gaussian quadrature arrays
  int num_quadrature;         // Order of quadrature used (used defined in input file)
  double **angles;            // Array of (theta,phi) angles for each point of quadrature (same for all shapes)
  double *weights;            // Weights of gaussian quadrature (same for all shapes)

  // Global SH properties
  int maxshexpan;             // Maximum degree of the shperical harmonic expansion
  int nshtypes;               // Number of spherical harmonic shapes

  // Testing properties (not for release)
  int verbose_out;            // Whether to print all the cout statements used in testing

  void read_sh_coeffs(char *, int); // Reads the spherical harmonic coefficients from file
  void get_quadrature_values();     // Get the gaussian quadrature angles and weights
  void getI();                      // Calculate the inertia of each shape
  void calcexpansionfactors();      // Calculate the expansion factors of each shape using a regular grid
  void calcexpansionfactors_gauss();// Calculate the expansion factors of each shape using the quadrature points
  void get_normal(double theta, double phi, double r, double rp, double rt, double rnorm[3]);

  double ***extentpoints_byshape;
  void gen_extent_box_points(int);     // Generate the points used to calculate the extent box for the shape
};

}

#endif
#endif
