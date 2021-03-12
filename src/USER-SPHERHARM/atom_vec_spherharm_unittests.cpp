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
#include <fstream>
#include <iomanip>
#include "atom_vec_spherharm_unittests.h"
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "fix_adapt.h"
#include "error.h"
#include "memory.h"
#include "math_const.h"
#include "math_spherharm.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpherharm;
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

//  check_sphere_normals();
//  check_ellipsoid_normals();
//  get_cog();
//  dump_ply();
//  dump_shapenormals();
//  compare_areas();
  validate_rotation();

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


void AtomVecSpherharmUnitTests::check_sphere_normals() {

  double theta,phi;
  double rad,rad_val,rad_dphi,rad_dtheta;
  double rnorm[3], x[3], diff[3];
  double mag_diff;
  int i,j,n;
  n = 100;

  rad_val = shcoeffs_byshape[0][0] * std::sqrt(1.0 / (4.0 * MY_PI));

  for (i=0; i<n; i++){
    theta = MY_PI*i/(n+1);
    for (j=0; j<n; j++){
      phi = MY_2PI*j/(n+1);
      rad = get_shape_radius_and_normal(0, theta, phi, rnorm);
      x[0] = rad_val*std::sin(theta)*std::cos(phi);
      x[1] = rad_val*std::sin(theta)*std::sin(phi);
      x[2] = rad_val*std::cos(theta);
      MathExtra::norm3(x);
      MathExtra::sub3(x,rnorm,diff);
      mag_diff = MathExtra::len3(diff);
      if (mag_diff > .01) std::cout << "Error" << std::endl;
//      std::cout << std::endl;
      //      std::cout << rnorm[0] << " " << rnorm[1] << " " << rnorm[2] << std::endl;
//      std::cout << x[0] << " " << x[1] << " " << x[2] << std::endl;
    }
  }

}


void AtomVecSpherharmUnitTests::check_ellipsoid_normals() {

  double theta,phi;
  double rad,rad_val,rad_dphi,rad_dtheta;
  double rnorm[3], val_norm[3], x[3], diff[3];
  double mag_diff;
  double ct,cp,st,sp;
  double a,b,c;
  int i,j,n;
  n = 100;

  a = get_shape_radius(0, MY_PI2, MY_PI2);
  b = get_shape_radius(0, MY_PI2, 0);
  c = get_shape_radius(0, 0,0);

  std::cout << "a " << a << " b " << b << " c " << c <<std::endl;

  std::ofstream outfile;
  outfile.open("test_dump/ellipsoid_norm_val.csv");
  if (outfile.is_open()) {
    outfile << "x,y,z,nx,ny,nz,val"<< "\n";
  } else std::cout << "Unable to open file";
  outfile.close();
  outfile.open("test_dump/ellipsoid_norm_val.csv", std::ios_base::app);
  for (i=0; i<=n; i++){
    theta = MY_PI*i/(n+1);
    for (j=0; j<=n; j++){
      phi = MY_2PI*j/(n+1);

      cp = std::cos(phi);
      sp = std::sin(phi);
      ct = std::cos(theta);
      st = std::sin(theta);
      rad_val = a*b*c;
      rad_val /= std::sqrt(c*c*st*st*((b*b*cp*cp)+(a*a*sp*sp))+(a*a*b*b*ct*ct));
      x[0] = rad_val*cp*st;
      x[1] = rad_val*sp*st;
      x[2] = rad_val*ct;
      val_norm[0] = x[0]* 2.0/(a*a);
      val_norm[1] = x[1]*2.0/(b*b);
      val_norm[2] = x[2]*2.0/(c*c);
      MathExtra::norm3(val_norm);

      rad = get_shape_radius_and_normal(0, theta, phi, rnorm);

      MathExtra::sub3(val_norm,rnorm,diff);
//      mag_diff = 100*MathExtra::len3(diff)/MathExtra::len3(x);
      mag_diff = 100*MathExtra::len3(diff);

//      std::cout << std::endl;
//      std::cout << "i " << i << " j " << j << std::endl;
//      std::cout << rad_val << " " << rad << std::endl;
//      std::cout << rnorm[0] << " " << rnorm[1] << " " << rnorm[2] << std::endl;
//      std::cout << val_norm[0] << " " << val_norm[1] << " " << val_norm[2] << std::endl;
//      std::cout << "dev " << mag_diff << std::endl;

      if (mag_diff>10) error->all(FLERR,"Large deviation in unit normal");

      if (outfile.is_open()) {
        outfile << x[0] << "," << x[1] << "," << x[2] << "," << val_norm[0] << "," << val_norm[1] << "," << val_norm[2] << "," << 1 << "\n";
        outfile << x[0] << "," << x[1] << "," << x[2] << "," << rnorm[0] << "," << rnorm[1] << "," << rnorm[2] << "," << 0 << "\n";
      } else std::cout << "Unable to open file";

    }
  }
  outfile.close();
}

void AtomVecSpherharmUnitTests::get_cog() {

  double vol=0.0;
  double cog[3];
  double iang = MY_PI;
  int trap_L = 2*(num_quadrature-1);
  double abscissa[num_quadrature];
  QuadPair p;

  // Get the quadrature weights, and abscissa. Convert abscissa to theta angles
  for (int i = 0; i < num_quadrature; i++) {
    p = GLPair(num_quadrature, i + 1);
    abscissa[i] = p.x();
  }

  cog[0]=cog[1]=cog[2] = 0.0;

  for (int ll = 0; ll <= trap_L; ll++) {
    double phi = MY_2PI * ll / (double(trap_L) + 1.0);
    for (int kk = 0; kk < num_quadrature; kk++) {
      double theta = (iang * 0.5 * abscissa[kk]) + (iang * 0.5);
      double rad = get_shape_radius(0, theta, phi);
      vol += weights[kk] * pow(rad,3) * std::sin(theta);
      cog[0] += weights[kk] * pow(rad, 4) * std::sin(theta) * std::sin(theta) * std::cos(phi);
      cog[1] += weights[kk] * pow(rad, 4) * std::sin(theta) * std::sin(theta) * std::sin(phi);
      cog[2] += weights[kk] * pow(rad, 4) * std::sin(theta) * std::cos(theta);
    }
  }

  vol *= (MY_PI*iang/((double(trap_L)+1.0)))/3.0;
  cog[0] *= (MY_PI*iang/((double(trap_L)+1.0)))/4.0;
  cog[1] *= (MY_PI*iang/((double(trap_L)+1.0)))/4.0;
  cog[2] *= (MY_PI*iang/((double(trap_L)+1.0)))/4.0;

  cog[0] /= vol;
  cog[1] /= vol;
  cog[2] /= vol;

  std::cout << std::endl;
  std::cout << "Total Volume" << std::endl;
  std::cout << vol << std::endl;
  std::cout << std::endl;
  std::cout << "COG" << std::endl;
  std::cout << cog[0] << " " << cog[1] << " " << cog[2] << std::endl;
  std::cout << std::endl;

}

void AtomVecSpherharmUnitTests::dump_ply() {

  double theta, phi, rad_body;
  double ix_sf[3];
  int count, sht;

  count = 0;
  sht = 0;

  std::ofstream outfile;
  outfile.open("test_ply/test.ply");
  if (outfile.is_open()) {
    outfile << "ply" << "\n";
    outfile << "format ascii 1.0" << "\n" << "element vertex " <<
      std::to_string(num_quadrature*num_quadrature) <<
      "\n" << "property float64 x" << "\n" << "property float64 y" <<
      "\n" << "property float64 z" << "\n" << "end_header" << "\n";
  } else std::cout << "Unable to open file";
  for (int i = 0; i < num_quadrature; i++) {
    for (int j = 0; j < num_quadrature; j++) {
      theta = angles[0][count];
      phi = angles[1][count];
      rad_body = quad_rads_byshape[sht][count];
      ix_sf[0] = (rad_body * sin(theta) * cos(phi));
      ix_sf[1] = (rad_body * sin(theta) * sin(phi));
      ix_sf[2] = (rad_body * cos(theta));
      outfile << ix_sf[0] << " " << ix_sf[1] << " " << ix_sf[2] << "\n";
      count++;
    }
  }
  outfile.close();
}

void AtomVecSpherharmUnitTests::dump_shapenormals() {

  std::cout<<"DUMPING NORMALS"<<std::endl;

  double theta, phi, rad_body;
  double ix_sf[3], norm[3];
  int count, sht;

  count = 0;
  sht = 0;

  std::ofstream outfile;
  outfile.open("test_dump/normalcheck.csv");
  if (outfile.is_open()) {
    outfile << "x,y,z,nx,ny,nz" << "\n";
    for (int i = 0; i < num_quadrature; i++) {
      for (int j = 0; j < num_quadrature; j++) {
        theta = angles[0][count];
        phi = angles[1][count];
        rad_body = get_shape_radius_and_normal(sht, theta, phi, norm);
        rad_body = quad_rads_byshape[sht][count];
        ix_sf[0] = (rad_body * sin(theta) * cos(phi));
        ix_sf[1] = (rad_body * sin(theta) * sin(phi));
        ix_sf[2] = (rad_body * cos(theta));
        outfile << std::setprecision(16) << ix_sf[0] << "," << ix_sf[1] << "," << ix_sf[2] <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
        count++;
      }
    }
    outfile.close();
  } else std::cout << "Unable to open file";

}


void AtomVecSpherharmUnitTests::compare_areas() {

  std::cout<<"Comparing Areas"<<std::endl;

  double theta, phi, rad, rp, rt, st, Q;
  double normQ[3], vec_sa[3];
  double surf_area, iang, factor, test_sa, surf_area2;
  double abscissa[num_quadrature];
  int sht, trap_L;

  sht = 0;
  iang = 4.0*MY_PI/5.0;
  trap_L = 2*(num_quadrature-1);
  factor = (MY_PI*iang/((double(trap_L)+1.0)));

  surf_area = 0.0;
  surf_area2 = 0.0;
  MathExtra::zero3(vec_sa);

  QuadPair p;
  // Get the quadrature weights, and abscissa. Convert abscissa to theta angles
  for (int i = 0; i < num_quadrature; i++) {
    p = GLPair(num_quadrature, i + 1);
    abscissa[i] = p.x();
  }

  for (int ll = 0; ll <= trap_L; ll++) {
    phi = MY_2PI * ll / (double(trap_L) + 1.0);
    for (int kk = 0; kk < num_quadrature; kk++) {
      theta = (iang * 0.5 * abscissa[kk]) + (iang * 0.5);
      st = std::sin(theta);
      rad = get_shape_radius_and_gradients(sht, theta, phi, rp, rt);
      get_normal(theta, phi, rad, rp, rt, normQ);
      Q = rad*std::sqrt((rp*rp)+(rt*rt*st*st)+(rad*rad*st*st));
      surf_area += weights[kk] * Q;
      MathExtra::scale3(weights[kk], normQ);
      MathExtra::add3(vec_sa, normQ, vec_sa);
      surf_area2 += MathExtra::len3(normQ);
    }
  }
  surf_area *= factor;
  surf_area2 *= factor;
  MathExtra::scale3(factor, vec_sa);
  test_sa = MathExtra::len3(vec_sa);

  std::cout <<"Surface area, direct calculation -> " << surf_area << std::endl;
  std::cout <<"Surface area, from vector area   -> " << surf_area2 << std::endl;
  std::cout <<"Projected surface area, from vector area   -> " << test_sa << std::endl;
}

void AtomVecSpherharmUnitTests::validate_rotation() {

  double *rotcoeffs;
  double quat[4];
  double rot[3][3], zxzmat[3][3];
  double alpha,beta,gamma;
  double theta, phi, num_quad2;
  double ix_bf[3], ix_sf[3], ix_sf_rot[3];
  double rad_val;
  int n, nloc, loc;
  double P_n_m, x_val, mphi, Pnm_nn;
  std::vector<double> Pnm_m2, Pnm_m1;

  memory->create(rotcoeffs, (maxshexpan+1)*(maxshexpan+2), "validate_rotation:rotcoeffs");

  quat[0]= -0.2;
  quat[1]= -0.3;
  quat[2]= -0.5;
  quat[3]= 0.1;

  std::string seq="ZYZ";
  MathExtra::qnormalize(quat);
  std::cout<<quat[0]<<" "<<quat[1]<<" "<<quat[2]<<" "<<quat[3]<<std::endl;
  MathExtra::quat_to_mat(quat, rot);

//  if (!MathSpherharm::quat_to_euler(quat, alpha, beta, gamma, seq)) error->all(FLERR, "Sequence missing");
//  std::cout<<"alpha, beta, gamma"<<std::endl;
//  std::cout<<alpha<<" "<<beta<<" "<<gamma<<std::endl;
  if (!MathSpherharm::quat_to_euler_test(quat, alpha, beta, gamma, seq)) error->all(FLERR, "Sequence missing");
  std::cout<<"alpha, beta, gamma"<<std::endl;
  std::cout<<alpha<<" "<<beta<<" "<<gamma<<std::endl;

  double c1,c2,c3,s1,s2,s3;
  c1 = std::cos(alpha);
  s1 = std::sin(alpha);
  c2 = std::cos(beta);
  s2 = std::sin(beta);
  c3 = std::cos(gamma);
  s3 = std::sin(gamma);

  std::cout << std::endl;
  std::cout << "Rot mat from quat" << std::endl;
  std::cout << rot[0][0] << " " << rot[0][1] << " " << rot[0][2] << std::endl;
  std::cout << rot[1][0] << " " << rot[1][1] << " " << rot[1][2] << std::endl;
  std::cout << rot[2][0] << " " << rot[2][1] << " " << rot[2][2] << std::endl;
  std::cout << std::endl;

  std::cout << "ZXZ rot mat" << std::endl;
  std::cout <<  c1*c3-c2*s1*s3 << " " << -c1*s3 - c2*c3*s1  << " " << s1*s2 << std::endl;
  std::cout <<  c3*s1+c1*c2*s3 << " " << c1*c2*c3 - s1*s3  << " " << -c1*s2 << std::endl;
  std::cout << s2*s3 << " " << c3*s2 << " " << c2 << std::endl;
  std::cout << std::endl;

  zxzmat[0][0]=c1*c3-c2*s1*s3;
  zxzmat[0][1]=-c1*s3 - c2*c3*s1;
  zxzmat[0][2]=s1*s2;
  zxzmat[1][0]=c3*s1+c1*c2*s3;
  zxzmat[1][1]=c1*c2*c3 - s1*s3;
  zxzmat[1][2]=-c1*s2;
  zxzmat[2][0]=s2*s3;
  zxzmat[2][1]=c3*s2;
  zxzmat[2][2]=c2;

  std::cout << "ZYZ rot mat" << std::endl;
  std::cout << c1*c2*c3-s1*s3 << " " << -c3*s1-c1*c2*s3 << " " << c1*s2 << std::endl;
  std::cout << c1*s3 + c2*c3*s1 << " " << c1*c3-c2*s1*s3 << " " << s1*s2 << std::endl;
  std::cout << -c3*s2 << " " << s2*s3 << " " << c2 << std::endl;
  std::cout << std::endl;

  get_coefficients(0, rotcoeffs);
//  doRotate(0, rotcoeffs, rotcoeffs, alpha, -beta, gamma);
  doRotate(0, rotcoeffs, rotcoeffs, alpha, beta, 0);
  doRotate(0, rotcoeffs, rotcoeffs, 0, 0, gamma);
  num_quad2 = num_quadrature*num_quadrature;

  std::ofstream outfile;
  outfile.open("plys/rottest.ply");
  if (outfile.is_open()) {
    outfile << "ply" << "\n";
    outfile << "format ascii 1.0" << "\n" << "element vertex " <<
            std::to_string(num_quadrature*num_quadrature) <<
            "\n" << "property double x" << "\n" << "property double y" <<
            "\n" << "property double z" << "\n" << "end_header" << "\n";
  } else std::cout << "Unable to open file";
  for (int k = 0; k < num_quad2; k++) {
    theta = angles[0][k];
    phi = angles[1][k];
    rad_val = rotcoeffs[0] * std::sqrt(1.0 / (4.0 * MY_PI));
    Pnm_m2.resize(maxshexpan+1, 0.0);
    Pnm_m1.resize(maxshexpan+1, 0.0);
    x_val = std::cos(theta);
    for (n=1; n<=maxshexpan; n++){
      nloc = n * (n + 1);
      if (n == 1) {
        P_n_m = plegendre(1, 0, x_val);
        Pnm_m2[0] = P_n_m;
        rad_val += rotcoeffs[4] * P_n_m;
        P_n_m = plegendre(1, 1, x_val);
        Pnm_m2[1] = P_n_m;
        mphi = 1.0 * phi;
        rad_val += (rotcoeffs[2] * cos(mphi) - rotcoeffs[3] * sin(mphi)) * 2.0 * P_n_m;
      } else if (n == 2) {
        P_n_m = plegendre(2, 0, x_val);
        Pnm_m1[0] = P_n_m;
        rad_val += rotcoeffs[10] * P_n_m;
        for (int m = 2; m >= 1; m--) {
          P_n_m = plegendre(2, m, x_val);
          Pnm_m1[m] = P_n_m;
          mphi = (double) m * phi;
          rad_val += (rotcoeffs[nloc] * cos(mphi) - rotcoeffs[nloc + 1] * sin(mphi)) * 2.0 * P_n_m;
          nloc += 2;
        }
        Pnm_nn = Pnm_m1[2];
      } else {
        P_n_m = plegendre_recycle(n, 0, x_val, Pnm_m1[0], Pnm_m2[0]);
        Pnm_m2[0] = Pnm_m1[0];
        Pnm_m1[0] = P_n_m;
        loc = (n + 1) * (n + 2) - 2;
        rad_val += rotcoeffs[loc] * P_n_m;
        loc -= 2;
        for (int m = 1; m < n - 1; m++) {
          P_n_m = plegendre_recycle(n, m, x_val, Pnm_m1[m], Pnm_m2[m]);
          Pnm_m2[m] = Pnm_m1[m];
          Pnm_m1[m] = P_n_m;
          mphi = (double) m * phi;
          rad_val += (rotcoeffs[loc] * cos(mphi) - rotcoeffs[loc + 1] * sin(mphi)) * 2.0 * P_n_m;
          loc -= 2;
        }

        P_n_m = x_val * std::sqrt((2.0 * ((double) n - 1.0)) + 3.0) * Pnm_nn;
        Pnm_m2[n - 1] = Pnm_m1[n - 1];
        Pnm_m1[n - 1] = P_n_m;
        mphi = (double) (n - 1) * phi;
        rad_val += (rotcoeffs[loc] * cos(mphi) - rotcoeffs[loc + 1] * sin(mphi)) * 2.0 * P_n_m;
        loc -= 2;

        P_n_m = plegendre_nn(n, x_val, Pnm_nn);
        Pnm_nn = P_n_m;
        Pnm_m1[n] = P_n_m;
        mphi = (double) n * phi;
        rad_val += (rotcoeffs[loc] * cos(mphi) - rotcoeffs[loc + 1] * sin(mphi)) * 2.0 * P_n_m;
      }
    }
    ix_sf_rot[0] = (rad_val * sin(theta) * cos(phi));
    ix_sf_rot[1] = (rad_val * sin(theta) * sin(phi));
    ix_sf_rot[2] = (rad_val * cos(theta));

    outfile << std::setprecision(16) << ix_sf_rot[0] << " " << ix_sf_rot[1] << " " << ix_sf_rot[2] << "\n";

  }
  outfile.close();


  outfile.open("plys/rottest_orig.ply");
  if (outfile.is_open()) {
    outfile << "ply" << "\n";
    outfile << "format ascii 1.0" << "\n" << "element vertex " <<
            std::to_string(num_quadrature*num_quadrature) <<
            "\n" << "property double x" << "\n" << "property double y" <<
            "\n" << "property double z" << "\n" << "end_header" << "\n";
  } else std::cout << "Unable to open file";
  for (int k = 0; k < num_quad2; k++) {
    theta = angles[0][k];
    phi = angles[1][k];
    rad_val = quad_rads_byshape[0][k];
    ix_bf[0] = (rad_val * sin(theta) * cos(phi));
    ix_bf[1] = (rad_val * sin(theta) * sin(phi));
    ix_bf[2] = (rad_val * cos(theta));
//    outfile << std::setprecision(16) << ix_bf[0] << " " << ix_bf[1] << " " << ix_bf[2] << "\n";
//    MathExtra::matvec(zxzmat, ix_bf, ix_sf);
    MathExtra::transpose_matvec(rot, ix_bf, ix_sf);
    outfile << std::setprecision(16) << ix_bf[0] << " " << ix_bf[1] << " " << ix_bf[2] << "\n";


//    std::cout << rad_val << " ";
//    double quat_foo[4], quat_bar[4];
//    MathSpherharm::spherical_to_quat(theta, phi, quat_foo);
//    MathExtra::qconjugate(quat, quat_bar);
//    MathExtra::quatquat(quat_bar, quat_foo, quat);
//    MathSpherharm::quat_to_spherical(quat, theta, phi);


    rad_val = rotcoeffs[0] * std::sqrt(1.0 / (4.0 * MY_PI));
    Pnm_m2.resize(maxshexpan+1, 0.0);
    Pnm_m1.resize(maxshexpan+1, 0.0);
    x_val = std::cos(theta);
    for (n=1; n<=maxshexpan; n++){
      nloc = n * (n + 1);
      if (n == 1) {
        P_n_m = plegendre(1, 0, x_val);
        Pnm_m2[0] = P_n_m;
        rad_val += rotcoeffs[4] * P_n_m;
        P_n_m = plegendre(1, 1, x_val);
        Pnm_m2[1] = P_n_m;
        mphi = 1.0 * phi;
        rad_val += (rotcoeffs[2] * cos(mphi) - rotcoeffs[3] * sin(mphi)) * 2.0 * P_n_m;
      } else if (n == 2) {
        P_n_m = plegendre(2, 0, x_val);
        Pnm_m1[0] = P_n_m;
        rad_val += rotcoeffs[10] * P_n_m;
        for (int m = 2; m >= 1; m--) {
          P_n_m = plegendre(2, m, x_val);
          Pnm_m1[m] = P_n_m;
          mphi = (double) m * phi;
          rad_val += (rotcoeffs[nloc] * cos(mphi) - rotcoeffs[nloc + 1] * sin(mphi)) * 2.0 * P_n_m;
          nloc += 2;
        }
        Pnm_nn = Pnm_m1[2];
      } else {
        P_n_m = plegendre_recycle(n, 0, x_val, Pnm_m1[0], Pnm_m2[0]);
        Pnm_m2[0] = Pnm_m1[0];
        Pnm_m1[0] = P_n_m;
        loc = (n + 1) * (n + 2) - 2;
        rad_val += rotcoeffs[loc] * P_n_m;
        loc -= 2;
        for (int m = 1; m < n - 1; m++) {
          P_n_m = plegendre_recycle(n, m, x_val, Pnm_m1[m], Pnm_m2[m]);
          Pnm_m2[m] = Pnm_m1[m];
          Pnm_m1[m] = P_n_m;
          mphi = (double) m * phi;
          rad_val += (rotcoeffs[loc] * cos(mphi) - rotcoeffs[loc + 1] * sin(mphi)) * 2.0 * P_n_m;
          loc -= 2;
        }

        P_n_m = x_val * std::sqrt((2.0 * ((double) n - 1.0)) + 3.0) * Pnm_nn;
        Pnm_m2[n - 1] = Pnm_m1[n - 1];
        Pnm_m1[n - 1] = P_n_m;
        mphi = (double) (n - 1) * phi;
        rad_val += (rotcoeffs[loc] * cos(mphi) - rotcoeffs[loc + 1] * sin(mphi)) * 2.0 * P_n_m;
        loc -= 2;

        P_n_m = plegendre_nn(n, x_val, Pnm_nn);
        Pnm_nn = P_n_m;
        Pnm_m1[n] = P_n_m;
        mphi = (double) n * phi;
        rad_val += (rotcoeffs[loc] * cos(mphi) - rotcoeffs[loc + 1] * sin(mphi)) * 2.0 * P_n_m;
      }
    }

  }
  outfile.close();


  double inorm[3],inormtemp[3];
  double quat_foo[4], quat_bar[4], quat_temp[4];
  for (int k = 0; k < num_quad2; k++) {
    theta = angles[0][k];
    phi = angles[1][k];

    ///// contact to init
//    rad_val = get_shape_radius_and_normal(theta, phi, inorm, rotcoeffs);
//    std::cout << rad_val << " ";
//
//    MathExtra::copy3(inorm, inormtemp);
//    MathExtra::matvec(rot, inormtemp, inorm);
//    std::cout << inorm[0] << " " << inorm[1] << " " << inorm[2] << " " << std::endl;
//
//    MathSpherharm::spherical_to_quat(theta, phi, quat_foo);
//    MathExtra::quatquat(quat, quat_foo, quat_bar);
//    MathSpherharm::quat_to_spherical(quat_bar, theta, phi);
//    rad_val = get_shape_radius_and_normal(theta, phi, inorm, shcoeffs_byshape[0]);
//    std::cout << rad_val << " ";
//    std::cout << inorm[0] << " " << inorm[1] << " " << inorm[2] << " " << std::endl;

    ///// init to contact
    rad_val = get_shape_radius_and_normal(theta, phi, inorm, shcoeffs_byshape[0]);
    std::cout << rad_val << " ";

    MathExtra::copy3(inorm, inormtemp);
    MathExtra::transpose_matvec(rot, inormtemp, inorm);
    std::cout << inorm[0] << " " << inorm[1] << " " << inorm[2] << " " << std::endl;

    MathSpherharm::spherical_to_quat(theta, phi, quat_bar);
    MathExtra::qconjugate(quat, quat_foo);
    MathExtra::quatquat(quat_foo, quat_bar, quat_temp);
    MathSpherharm::quat_to_spherical(quat_temp, theta, phi);
    rad_val = get_shape_radius_and_normal(theta, phi, inorm, rotcoeffs);
    std::cout << rad_val << " ";
    std::cout << inorm[0] << " " << inorm[1] << " " << inorm[2] << " " << std::endl;
  }


  memory->sfree(rotcoeffs);

}
