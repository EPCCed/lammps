/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributead under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <random_park.h>
#include <math_extra.h>
#include <iostream>
#include "atom_vec_spherharm_unittests.h"
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "fix_adapt.h"
#include "error.h"
#include "memory.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;
/* ---------------------------------------------------------------------- */

AtomVecSpherharmUnitTests::AtomVecSpherharmUnitTests(LAMMPS *lmp) : AtomVecSpherharm(lmp)
{
  ellipsoidshape = nullptr;
}

AtomVecSpherharmUnitTests::~AtomVecSpherharmUnitTests()
{
  memory->sfree(ellipsoidshape);
}

/* ----------------------------------------------------------------------
   process sub-style args
------------------------------------------------------------------------- */

void AtomVecSpherharmUnitTests::process_args(int narg, char **arg) {

  AtomVecSpherharm::process_args(narg, arg);

  for (int i=0; i<nshtypes; i++){
    pinertia_byshape[i][0] /=441.0;
    pinertia_byshape[i][1] /=441.0;
    pinertia_byshape[i][2] /=441.0;
  }

  MPI_Bcast(&(pinertia_byshape[0][0]), nshtypes * 3, MPI_DOUBLE, 0, world);

  memory->create(ellipsoidshape, nshtypes, 3, "AtomVecSpherharmUnitTests:ellipsoidshape");
}

void AtomVecSpherharmUnitTests::get_shape(int i, double &shapex, double &shapey, double &shapez)
{
  ellipsoidshape[0][0] = 0.5;
  ellipsoidshape[0][1] = 0.5;
  ellipsoidshape[0][2] = 2.5;

  shapex = ellipsoidshape[shtype[i]][0];
  shapey = ellipsoidshape[shtype[i]][1];
  shapez = ellipsoidshape[shtype[i]][2];
}

void AtomVecSpherharmUnitTests::check_rotations(int sht, int i) {

  int seed = 4;
  double x[3];
  double quattest[4];
  double s, t1, t2, theta1, theta2;
  RanPark *ranpark = new RanPark(lmp, 1);

  double iquat_delta[4];
  double jquat_delta[4];
  double quat_temp[4];
  double irot[3][3], jrot[3][3];
  double ixquadbf[3], jxquadbf[3];
  double ixquadsf[3], jxquadsf[3];
  double xgauss[3], xgaussproj[3];
  double dtemp, phi_proj, theta_proj, finalrad;
  double phi, theta;
  double overlap[3];

  x[0] = x[1] = x[2] = 475.0;
  ranpark->reset(seed, x);
  s = ranpark->uniform();
  t1 = sqrt(1.0 - s);
  t2 = sqrt(s);
  theta1 = 2.0 * MY_PI * ranpark->uniform();
  theta2 = 2.0 * MY_PI * ranpark->uniform();
  quattest[0] = cos(theta2) * t2;
  quattest[1] = sin(theta1) * t1;
  quattest[2] = cos(theta1) * t1;
  quattest[3] = sin(theta2) * t2;

  // Unrolling the initial quat (Unit quaternion - inverse = conjugate)
  MathExtra::qconjugate(pinertia_byshape[sht], quat_temp);
  // Calculating the quat to rotate the particles to new position (q_delta = q_target * q_current^-1)
  MathExtra::quatquat(quattest, quat_temp, iquat_delta);
  MathExtra::qnormalize(iquat_delta);
  // Calculate the rotation matrix for the quaternion for atom i
  MathExtra::quat_to_mat(iquat_delta, irot);

  std::cout << std::endl;
  std::cout << "Rotation matrix" << std::endl;
  std::cout << irot[0][0] << " " << irot[0][1] << " " << irot[0][2] << " " << std::endl;
  std::cout << irot[1][0] << " " << irot[1][1] << " " << irot[1][2] << " " << std::endl;
  std::cout << irot[2][0] << " " << irot[2][1] << " " << irot[2][2] << " " << std::endl;


  // Point of gaussian quadrature in body frame
  ixquadbf[0] = quad_rads_byshape[sht][i] * sin(angles[0][i]) * cos(angles[1][i]);
  ixquadbf[1] = quad_rads_byshape[sht][i] * sin(angles[0][i]) * sin(angles[1][i]);
  ixquadbf[2] = quad_rads_byshape[sht][i] * cos(angles[0][i]);
  // Point of gaussian quadrature in space frame
  MathExtra::matvec(irot, ixquadbf, ixquadsf);

  std::cout << std::endl;
  std::cout << "Body and space (before translate)" << std::endl;
  std::cout << ixquadbf[0] << " " << ixquadbf[1] << " " << ixquadbf[2] << " " << std::endl;
  std::cout << ixquadsf[0] << " " << ixquadsf[1] << " " << ixquadsf[2] << " " << std::endl;



  dtemp = MathExtra::len3(ixquadsf);

  // Translating by current location of particle i's centre
  ixquadsf[0] += x[0];
  ixquadsf[1] += x[1];
  ixquadsf[2] += x[1];

  std::cout << std::endl;
  std::cout << "Space (after translate)" << std::endl;
  std::cout << ixquadsf[0] << " " << ixquadsf[1] << " " << ixquadsf[2] << " " << std::endl;
  std::cout << dtemp << " " << quad_rads_byshape[sht][i] << std::endl;


  // vector distance from gauss point on atom j to COG of atom i (in space frame)
  MathExtra::sub3(ixquadsf, x, xgauss);
  // scalar distance
  dtemp = MathExtra::len3(xgauss);
  // Rotating the projected point into atom i's body frame (rotation matrix transpose = inverse)
  MathExtra::transpose_matvec(irot, xgauss, xgaussproj);
  // Get projected phi and theta angle of gauss point in atom i's body frame
  phi_proj = std::atan2(xgaussproj[1], xgaussproj[0]);
  phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj;
  theta_proj = std::acos(xgaussproj[2] / dtemp);

  std::cout << std::endl;
  std::cout << x[0] << " " << x[1] << " " << x[2] << " " << std::endl;
  std::cout << xgauss[0] << " " << xgauss[1] << " " << xgauss[2] << " " << std::endl;
  std::cout << xgaussproj[0] << " " << xgaussproj[1] << " " << xgaussproj[2] << " " << std::endl;
  std::cout << ixquadbf[0] << " " << ixquadbf[1] << " " << ixquadbf[2] << " " << std::endl;
  std::cout << MathExtra::len3(xgaussproj) << " " << std::endl;
  std::cout << theta_proj << " " << angles[0][i] <<" "<< phi_proj << " " << angles[1][i] << std::endl;


}