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

#ifndef LMP_MATH_SPHERHARM_H
#define LMP_MATH_SPHERHARM_H

#include <cmath>

namespace MathSpherharm {

  // Inline methods
  inline void quat_to_spherical(double m[4], double &theta, double &phi);
  inline void spherical_to_quat(double theta, double phi, double q[4]);

  // Normalised Legendre polynomials
  double plegendre( int l,  int m,  double x);
  double plegendre_nn( int l,  double x,  double Pnm_nn);
  double plegendre_recycle( int l,  int m,  double x,  double pnm_m1,  double pnm_m2);
  // Not normalised Legendre polynomials
  double plgndr(int l, int m, double x);

  // Gaussian quadrature methods
  // A struct for containing a Node-Weight pair
  struct QuadPair {
    double theta, weight;

    // A function for getting the node in x-space
    double x() {return cos(theta);}

    // A constructor
    QuadPair(double t, double w) : theta(t), weight(w) {}
    QuadPair() {}
  };
  // Function for getting Gauss-Legendre nodes & weights
  // Theta values of the zeros are in [0,pi], and monotonically increasing.
  // The index of the zero k should always be in [1,n].
  // Compute a node-weight pair:
  QuadPair GLPair(size_t, size_t);

  double besseljzero(int);
  double besselj1squared(int);
  QuadPair GLPairS(size_t, size_t);
  QuadPair GLPairTabulated(size_t, size_t);


  // For calculating factorials, borrowed from compute_orientorder_atom.cpp
  double factorial(int);

}

/* ----------------------------------------------------------------------
  Convert quaternion into spherical theta, phi values
------------------------------------------------------------------------- */
inline void MathSpherharm::quat_to_spherical(double q[4], double &theta, double &phi)
{
  double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  theta = 2*acos(sqrt((q[0]*q[0] + q[3]*q[3])/norm));
  phi = atan2(q[3], q[0]) + atan2(-q[1], q[2]);
}

/* ----------------------------------------------------------------------
  Convert spherical theta, phi values into a quaternion
  // https://github.com/moble/quaternion/blob/master/src/quaternion.c
  // https://quaternion.readthedocs.io/en/latest/Package%20API%3A/quaternion/
------------------------------------------------------------------------- */
inline void MathSpherharm::spherical_to_quat(double theta, double phi, double q[4])
{
  double ct = cos(theta/2.0);
  double cp = cos(phi/2.0);
  double st = sin(theta/2.0);
  double sp = sin(phi/2.0);
  q[0] = cp*ct;
  q[1] = -sp*st;
  q[2] = st*cp;
  q[3] = sp*ct;
}
#endif
