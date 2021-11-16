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
#include <math_extra.h>
#include <iostream>

namespace MathSpherharm {

  // Inline methods
  inline void quat_to_spherical(double q[4], double &theta, double &phi);
  inline void spherical_to_quat(double theta, double phi, double q[4]);
  inline int quat_to_euler(double q[4], double &alpha, double &beta, double &gamma, const std::string& seq = "ZYX");
  inline int quat_to_euler_test(double q[4], double &alpha, double &beta, double &gamma, const std::string& seq = "ZXZ");
  inline void times4(const double m[4][4], const double m2[4][4], double ans[4][4]);
  inline bool invert4(const double m[4][4], double invOut[4][4]);
  inline void matvec4(const double M[4][4], const double v[4], double ans[4]);

  // Normalised Legendre polynomials
  // See Numerical Recipies 3rd Edition Section 6.7 Spherical Harmonics
  double plegendre( int l,  int m,  double x);
  double plegendre_nn( int l,  double x,  double Pnm_nn);
  double plegendre_recycle( int l,  int m,  double x,  double pnm_m1,  double pnm_m2);
  // Not normalised Legendre polynomials
  double plgndr(int l, int m, double x);

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
  int line_ellipsoid_intersection(const double elipsoid_centre[3], const double elipse_x_axis[3],
                                  const double elipse_y_axis[3],const double elipse_z_axis[3],
                                  const double line_centre[3], const double line_normal[3],
                                  double &t);

  // Contact point between bounding sphere and plane or cylinder
  int get_contact_point_plane(double rada, double xi[3], double (&linenorm)[3],
                               double (&lineorigin)[3], double (&p0)[3],
                               double (&cp)[3]);
  int get_contact_point_cylinder(double rada, double xi[3], double (&linenorm)[3],
                                  double(&lineorigin)[3], double (&cp)[3], double cylradius, bool inside);

  void get_contact_quat(double (&xvecdist)[3], double (&quat)[4]);
  double get_sphere_overlap_volume(double r1, double r2, double d);
}

/* ----------------------------------------------------------------------
  Convert quaternion into spherical theta, phi values
------------------------------------------------------------------------- */
inline void MathSpherharm::quat_to_spherical(double q[4], double &theta, double &phi)
{
  double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  theta = 2.0*acos(sqrt((q[0]*q[0] + q[3]*q[3])/norm));
  phi = atan2(q[3], q[0]) + atan2(-q[1], q[2]);
}

/* ----------------------------------------------------------------------
  Convert quaternion into z-y-z convention euler angles alpha, beta, and gamma
  Theory from MATLABs quat2rotm and rotm2eul
------------------------------------------------------------------------- */
inline int MathSpherharm::quat_to_euler(double q[4], double &alpha, double &beta, double &gamma, const std::string& seq)
{

  if (seq=="ZYX") {
    double aSI;
    aSI = -2*(q[1]*q[3]-q[0]*q[2]);
    aSI = aSI > 1.0 ? 1.0 : aSI; // cap aSI to 1
    aSI = aSI < -1.0 ? -1.0 : aSI; // cap aSI to -1

    alpha = std::atan2(2*(q[1]*q[2]+q[0]*q[3]), q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]);
    beta = std::asin(aSI);
    gamma = std::atan2(2*(q[2]*q[3]+q[0]*q[1]), q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]);
    return 1;
  }
  else if (seq=="ZYZ") {
    double sy;
    double R[3][3];
    bool singular;
    MathExtra::quat_to_mat(q, R);

    sy = std::sqrt(R[2][1] * R[2][1] + R[2][0] * R[2][0]);
    singular = sy < 1e-6;

    alpha = -std::atan2(R[1][2], -R[0][2]);
    beta = -std::atan2(sy, R[2][2]);
    gamma = -std::atan2(R[2][1], R[2][0]);

    if (singular) {
      alpha = 0.0;
      beta = -std::atan2(sy, R[2][2]);
      gamma = -std::atan2(-R[1][0], R[1][1]);
    }
    return 1;
  }
  return 0;
}

inline int MathSpherharm::quat_to_euler_test(double q[4], double &alpha, double &beta, double &gamma, const std::string& seq)
{

  double sy, temp;
  double R[3][3];
  bool singular;
  int setting[4];
  int firstAxis, repetition, parity, movingFrame;
  int i, j, k;
  int nextAxis[4];
  nextAxis[0] = 1;
  nextAxis[1] = 2;
  nextAxis[2] = 0;
  nextAxis[3] = 1;

  if (seq=="ZYX") {
    setting[0] = 0;
    setting[1] = 0;
    setting[2] = 0;
    setting[3] = 1;
  }
  else if (seq=="ZYZ") {
    setting[0] = 2;
    setting[1] = 1;
    setting[2] = 1;
    setting[3] = 1;
  }
  else if (seq=="XYZ") {
    setting[0] = 2;
    setting[1] = 0;
    setting[2] = 1;
    setting[3] = 1;
  }
  else if (seq=="ZXZ") {
    setting[0] = 2;
    setting[1] = 1;
    setting[2] = 0;
    setting[3] = 1;
  }
  else return 0;

  firstAxis = setting[0];
  repetition = setting[1];
  parity = setting[2];
  movingFrame = setting[3];
  i = firstAxis;
  j = nextAxis[i+parity];
  k = nextAxis[i-parity+1];
  MathExtra::quat_to_mat(q, R);

  if (repetition) {
    sy = std::sqrt(R[i][j] * R[i][j] + R[i][k] * R[i][k]);
    singular = sy < 1e-6;

    alpha = std::atan2(R[i][j], R[i][k]);
    beta = std::atan2(sy, R[i][i]);
    gamma = std::atan2(R[j][i], -R[k][i]);

    if (singular) {
      alpha = std::atan2(-R[j][k], R[j][j]);
      beta = std::atan2(sy, R[i][i]);
      gamma = 0.0;
    }
  }
  else{
    sy = std::sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]);
    singular = sy < 1e-6;

    alpha = std::atan2(R[k][j], R[k][k]);
    beta = std::atan2(sy, -R[k][i]);
    gamma = std::atan2(R[j][i], R[i][i]);

    if (singular) {
      alpha = std::atan2(-R[j][k], R[j][j]);
      beta = std::atan2(-R[k][i],sy);
      gamma = 0.0;
    }
  }

  if (parity){
    alpha = - alpha;
    beta = - beta;
    gamma = - gamma;
  }

  if (movingFrame){
    temp = alpha;
    alpha = gamma;
    gamma = temp;
  }
  return 1;
}

/* ----------------------------------------------------------------------
  Convert spherical theta, phi values into a quaternion
  https://github.com/moble/quaternion/blob/master/src/quaternion.c
  https://quaternion.readthedocs.io/en/latest/Package%20API%3A/quaternion/
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


inline void MathSpherharm::times4(const double m[4][4], const double m2[4][4],
                              double ans[4][4])
{
  ans[0][0] = m[0][0]*m2[0][0] + m[0][1]*m2[1][0] + m[0][2]*m2[2][0] + m[0][3]*m2[3][0];
  ans[0][1] = m[0][0]*m2[0][1] + m[0][1]*m2[1][1] + m[0][2]*m2[2][1] + m[0][3]*m2[3][1];
  ans[0][2] = m[0][0]*m2[0][2] + m[0][1]*m2[1][2] + m[0][2]*m2[2][2] + m[0][3]*m2[3][2];
  ans[0][3] = m[0][0]*m2[0][3] + m[0][1]*m2[1][3] + m[0][2]*m2[2][3] + m[0][3]*m2[3][3];

  ans[1][0] = m[1][0]*m2[0][0] + m[1][1]*m2[1][0] + m[1][2]*m2[2][0] + m[1][3]*m2[3][0];
  ans[1][1] = m[1][0]*m2[0][1] + m[1][1]*m2[1][1] + m[1][2]*m2[2][1] + m[1][3]*m2[3][1];
  ans[1][2] = m[1][0]*m2[0][2] + m[1][1]*m2[1][2] + m[1][2]*m2[2][2] + m[1][3]*m2[3][2];
  ans[1][3] = m[1][0]*m2[0][3] + m[1][1]*m2[1][3] + m[1][2]*m2[2][3] + m[1][3]*m2[3][3];

  ans[2][0] = m[2][0]*m2[0][0] + m[2][1]*m2[1][0] + m[2][2]*m2[2][0] + m[2][3]*m2[3][0];
  ans[2][1] = m[2][0]*m2[0][1] + m[2][1]*m2[1][1] + m[2][2]*m2[2][1] + m[2][3]*m2[3][1];
  ans[2][2] = m[2][0]*m2[0][2] + m[2][1]*m2[1][2] + m[2][2]*m2[2][2] + m[2][3]*m2[3][2];
  ans[2][3] = m[2][0]*m2[0][3] + m[2][1]*m2[1][3] + m[2][2]*m2[2][3] + m[2][3]*m2[3][3];

  ans[3][0] = m[3][0]*m2[0][0] + m[3][1]*m2[1][0] + m[3][2]*m2[2][0] + m[3][3]*m2[3][0];
  ans[3][1] = m[3][0]*m2[0][1] + m[3][1]*m2[1][1] + m[3][2]*m2[2][1] + m[3][3]*m2[3][1];
  ans[3][2] = m[3][0]*m2[0][2] + m[3][1]*m2[1][2] + m[3][2]*m2[2][2] + m[3][3]*m2[3][2];
  ans[3][3] = m[3][0]*m2[0][3] + m[3][1]*m2[1][3] + m[3][2]*m2[2][3] + m[3][3]*m2[3][3];
}

inline bool MathSpherharm::invert4(const double m[4][4], double invOut[4][4])
{
  double inv[16], det;
  int i, j;

  inv[0] = m[1][1] * m[2][2] * m[3][3] -
           m[1][1] * m[2][3] * m[3][2] -
           m[2][1] * m[1][2] * m[3][3] +
           m[2][1] * m[1][2] * m[3][2] +
           m[3][1] * m[1][2] * m[2][3] -
           m[3][1] * m[1][2] * m[2][2];

  inv[4] = -m[1][0] * m[2][2] * m[3][3] +
           m[1][0] * m[2][3] * m[3][2] +
           m[2][0] * m[1][2] * m[3][3] -
           m[2][0] * m[1][2] * m[3][2] -
           m[3][0] * m[1][2] * m[2][3] +
           m[3][0] * m[1][2] * m[2][2];

  inv[8] = m[1][0] * m[2][1] * m[3][3] -
           m[1][0] * m[2][3] * m[3][1] -
           m[2][0] * m[1][1] * m[3][3] +
           m[2][0] * m[1][2] * m[3][1] +
           m[3][0] * m[1][1] * m[2][3] -
           m[3][0] * m[1][2] * m[2][1];

  inv[12] = -m[1][0] * m[2][1] * m[3][2] +
            m[1][0] * m[2][2] * m[3][1] +
            m[2][0] * m[1][1] * m[3][2] -
            m[2][0] * m[1][2] * m[3][1] -
            m[3][0] * m[1][1] * m[2][2] +
            m[3][0] * m[1][2] * m[2][1];

  inv[1] = -m[0][1] * m[2][2] * m[3][3] +
           m[0][1] * m[2][3] * m[3][2] +
           m[2][1] * m[0][2] * m[3][3] -
           m[2][1] * m[0][3] * m[3][2] -
           m[3][1] * m[0][2] * m[2][3] +
           m[3][1] * m[0][3] * m[2][2];

  inv[5] = m[0][0] * m[2][2] * m[3][3] -
           m[0][0] * m[2][3] * m[3][2] -
           m[2][0] * m[0][2] * m[3][3] +
           m[2][0] * m[0][3] * m[3][2] +
           m[3][0] * m[0][2] * m[2][3] -
           m[3][0] * m[0][3] * m[2][2];

  inv[9] = -m[0][0] * m[2][1] * m[3][3] +
           m[0][0] * m[2][3] * m[3][1] +
           m[2][0] * m[0][1] * m[3][3] -
           m[2][0] * m[0][3] * m[3][1] -
           m[3][0] * m[0][1] * m[2][3] +
           m[3][0] * m[0][3] * m[2][1];

  inv[13] = m[0][0] * m[2][1] * m[3][2] -
            m[0][0] * m[2][2] * m[3][1] -
            m[2][0] * m[0][1] * m[3][2] +
            m[2][0] * m[0][2] * m[3][1] +
            m[3][0] * m[0][1] * m[2][2] -
            m[3][0] * m[0][2] * m[2][1];

  inv[2] = m[0][1] * m[1][2] * m[3][3] -
           m[0][1] * m[1][2] * m[3][2] -
           m[1][1] * m[0][2] * m[3][3] +
           m[1][1] * m[0][3] * m[3][2] +
           m[3][1] * m[0][2] * m[1][2] -
           m[3][1] * m[0][3] * m[1][2];

  inv[6] = -m[0][0] * m[1][2] * m[3][3] +
           m[0][0] * m[1][2] * m[3][2] +
           m[1][0] * m[0][2] * m[3][3] -
           m[1][0] * m[0][3] * m[3][2] -
           m[3][0] * m[0][2] * m[1][2] +
           m[3][0] * m[0][3] * m[1][2];

  inv[10] = m[0][0] * m[1][1] * m[3][3] -
            m[0][0] * m[1][2] * m[3][1] -
            m[1][0] * m[0][1] * m[3][3] +
            m[1][0] * m[0][3] * m[3][1] +
            m[3][0] * m[0][1] * m[1][2] -
            m[3][0] * m[0][3] * m[1][1];

  inv[14] = -m[0][0] * m[1][1] * m[3][2] +
            m[0][0] * m[1][2] * m[3][1] +
            m[1][0] * m[0][1] * m[3][2] -
            m[1][0] * m[0][2] * m[3][1] -
            m[3][0] * m[0][1] * m[1][2] +
            m[3][0] * m[0][2] * m[1][1];

  inv[3] = -m[0][1] * m[1][2] * m[2][3] +
           m[0][1] * m[1][2] * m[2][2] +
           m[1][1] * m[0][2] * m[2][3] -
           m[1][1] * m[0][3] * m[2][2] -
           m[2][1] * m[0][2] * m[1][2] +
           m[2][1] * m[0][3] * m[1][2];

  inv[7] = m[0][0] * m[1][2] * m[2][3] -
           m[0][0] * m[1][2] * m[2][2] -
           m[1][0] * m[0][2] * m[2][3] +
           m[1][0] * m[0][3] * m[2][2] +
           m[2][0] * m[0][2] * m[1][2] -
           m[2][0] * m[0][3] * m[1][2];

  inv[11] = -m[0][0] * m[1][1] * m[2][3] +
            m[0][0] * m[1][2] * m[2][1] +
            m[1][0] * m[0][1] * m[2][3] -
            m[1][0] * m[0][3] * m[2][1] -
            m[2][0] * m[0][1] * m[1][2] +
            m[2][0] * m[0][3] * m[1][1];

  inv[15] = m[0][0] * m[1][1] * m[2][2] -
            m[0][0] * m[1][2] * m[2][1] -
            m[1][0] * m[0][1] * m[2][2] +
            m[1][0] * m[0][2] * m[2][1] +
            m[2][0] * m[0][1] * m[1][2] -
            m[2][0] * m[0][2] * m[1][1];

  det = m[0][0] * inv[0] + m[0][1] * inv[4] + m[0][2] * inv[8] + m[0][3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 4; j++)
      invOut[i][j] = inv[j+i*4] * det;

  return true;
}

inline void MathSpherharm::matvec4(const double M[4][4], const double v[4], double ans[4])
{
  ans[0] = M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3];
  ans[1] = M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3];
  ans[2] = M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] + M[2][3]*v[3];
  ans[3] = M[3][0]*v[0] + M[3][1]*v[1] + M[3][2]*v[2] + M[3][3]*v[3];
}

#endif
