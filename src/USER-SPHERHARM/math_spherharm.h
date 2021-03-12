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
  inline void quat_to_spherical(double m[4], double &theta, double &phi);
  inline void spherical_to_quat(double theta, double phi, double q[4]);
  inline int quat_to_euler(double q[4], double &alpha, double &beta, double &gamma, const std::string& seq = "ZYX");
  inline int quat_to_euler_test(double q[4], double &alpha, double &beta, double &gamma, const std::string& seq = "ZXZ");

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
#endif
