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

  // Normalised Legendre polynomials
  double plegendre(const int l, const int m, const double x);
  double plegendre_nn(const int l, const double x, const double Pnm_nn);
  double plegendre_recycle(const int l, const int m, const double x, const double pnm_m1, const double pnm_m2);

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

}

#endif
