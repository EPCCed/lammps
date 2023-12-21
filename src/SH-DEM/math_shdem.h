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

#ifndef LMP_MATH_SHDEM_H
#define LMP_MATH_SHDEM_H

#include "cmath"
#include "math_extra.h"
#include "iostream"

namespace MathSHDEM {

  // Gaussian quadrature methods
  // A struct for containing a Node-Weight pair
  // See https://people.math.sc.edu/Burkardt/cpp_src/fastgl/fastgl.html
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
  // Finding the intersections with a ray defined by origin and normal with a sphere, plane and cylinder
  int line_sphere_intersection(const double rad, const double circcentre[3], const double linenorm[3],
                               const double lineorigin[3], double &sol1, double &sol2);
  int line_plane_intersection(double (&p0)[3], double (&l0)[3], double (&l)[3], double (&n)[3], double &sol);
  int line_cylinder_intersection(const double xi[3], const double (&unit_line_normal)[3], double &t1,
          double &t2, double cylradius);

  // Contact point between bounding sphere and plane or cylinder
  int get_contact_point_plane(double rada, double xi[3], double (&linenorm)[3],
                               double (&lineorigin)[3], double (&p0)[3],
                               double (&cp)[3]);
  int get_contact_point_cylinder(double rada, double xi[3], double (&linenorm)[3],
                                  double(&lineorigin)[3], double (&cp)[3], double cylradius, bool inside);

  void get_contact_quat(double (&xvecdist)[3], double (&quat)[4]);
  double get_sphere_overlap_volume(double r1, double r2, double d);
}

#endif
