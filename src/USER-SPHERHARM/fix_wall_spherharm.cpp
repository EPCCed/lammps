/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Leo Silbert (SNL), Gary Grest (SNL),
                         Dan Bolintineanu (SNL)
------------------------------------------------------------------------- */

#include "fix_wall_spherharm.h"
#include <cmath>
#include <cstring>
#include <math_extra.h>
#include <iomanip>
#include <fstream>
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "force.h"
#include "modify.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "neighbor.h"
#include "math_spherharm.h"
#include "atom_vec_spherharm.h"


using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

// XYZ PLANE need to be 0,1,2

enum{XPLANE=0,YPLANE=1,ZPLANE=2,ZCYLINDER,REGION};
enum{VOLUME_BASED};
enum{NONE,CONSTANT,EQUAL};

#define PI27SQ 266.47931882941264802866    // 27*PI**2
#define THREEROOT3 5.19615242270663202362  // 3*sqrt(3)
#define SIXROOT6 14.69693845669906728801   // 6*sqrt(6)
#define INVROOT6 0.40824829046386307274    // 1/sqrt(6)
#define FOURTHIRDS 1.333333333333333       // 4/3
#define THREEQUARTERS 0.75                 // 3/4
#define TWOPI 6.28318530717959             // 2*PI

#define EPSILON 1e-10
#define BIG 1.0e20
#define EPSILON 1e-10

/* ---------------------------------------------------------------------- */

FixWallSpherharm::FixWallSpherharm(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), idregion(nullptr)
{
  if (narg < 7) error->all(FLERR,"Illegal fix wall/spherharm command");

  if (!atom->spherharm_flag)
    error->all(FLERR,"Fix wall/spherharm requires atom style spherharm");

  num_pole_quad = 30;
  create_attribute = 1;
  restart_peratom = 0;

  // set interaction style
  pairstyle = VOLUME_BASED;

  // wall/particle coefficients

  int iarg;

  kn = utils::numeric(FLERR, arg[3], false, lmp);
  mexpon = utils::numeric(FLERR, arg[4], false, lmp);

  if (kn < 0.0 || mexpon < 1.0)
    error->all(FLERR, "Illegal fix wall/gran command");

  iarg = 5;

  // wallstyle args

  idregion = nullptr;

  if (strcmp(arg[iarg],"xplane") == 0) {
    if (narg < iarg+3) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = XPLANE;
    if (strcmp(arg[iarg+1],"NULL") == 0) lo = -BIG;
    else lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    if (strcmp(arg[iarg+2],"NULL") == 0) hi = BIG;
    else hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    iarg += 3;
  } else if (strcmp(arg[iarg],"yplane") == 0) {
    if (narg < iarg+3) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = YPLANE;
    if (strcmp(arg[iarg+1],"NULL") == 0) lo = -BIG;
    else lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    if (strcmp(arg[iarg+2],"NULL") == 0) hi = BIG;
    else hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    iarg += 3;
  } else if (strcmp(arg[iarg],"zplane") == 0) {
    if (narg < iarg+3) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = ZPLANE;
    if (strcmp(arg[iarg+1],"NULL") == 0) lo = -BIG;
    else lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    if (strcmp(arg[iarg+2],"NULL") == 0) hi = BIG;
    else hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
    iarg += 3;
  } else if (strcmp(arg[iarg],"zcylinder") == 0) {
    if (narg < iarg+2) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = ZCYLINDER;
    lo = hi = 0.0;
    cylradius = utils::numeric(FLERR,arg[iarg+1],false,lmp);
    iarg += 2;
  } else if (strcmp(arg[iarg],"region") == 0) {
    if (narg < iarg+2) error->all(FLERR,"Illegal fix wall/gran command");
    wallstyle = REGION;
    int n = strlen(arg[iarg+1]) + 1;
    idregion = new char[n];
    strcpy(idregion,arg[iarg+1]);
    iarg += 2;
  }

  // optional args

  wiggle = 0;
  wshear = 0;
  peratom_flag = 0;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"wiggle") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix wall/gran command");
      if (strcmp(arg[iarg+1],"x") == 0) axis = 0;
      else if (strcmp(arg[iarg+1],"y") == 0) axis = 1;
      else if (strcmp(arg[iarg+1],"z") == 0) axis = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      amplitude = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      period = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      wiggle = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"shear") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix wall/gran command");
      if (strcmp(arg[iarg+1],"x") == 0) axis = 0;
      else if (strcmp(arg[iarg+1],"y") == 0) axis = 1;
      else if (strcmp(arg[iarg+1],"z") == 0) axis = 2;
      else error->all(FLERR,"Illegal fix wall/gran command");
      vshear = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      wshear = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg],"contacts") == 0) {
      peratom_flag = 1;
      size_peratom_cols = 8;
      peratom_freq = 1;
      iarg += 1;
    } else error->all(FLERR,"Illegal fix wall/gran command");
  }

  if (wallstyle == XPLANE && domain->xperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (wallstyle == YPLANE && domain->yperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (wallstyle == ZPLANE && domain->zperiodic)
    error->all(FLERR,"Cannot use wall in periodic dimension");
  if (wallstyle == ZCYLINDER && (domain->xperiodic || domain->yperiodic))
    error->all(FLERR,"Cannot use wall in periodic dimension");

  if (wiggle && wshear)
    error->all(FLERR,"Cannot wiggle and shear fix wall/gran");
  if (wiggle && wallstyle == ZCYLINDER && axis != 2)
    error->all(FLERR,"Invalid wiggle direction for fix wall/gran");
  if (wshear && wallstyle == XPLANE && axis == 0)
    error->all(FLERR,"Invalid shear direction for fix wall/gran");
  if (wshear && wallstyle == YPLANE && axis == 1)
    error->all(FLERR,"Invalid shear direction for fix wall/gran");
  if (wshear && wallstyle == ZPLANE && axis == 2)
    error->all(FLERR,"Invalid shear direction for fix wall/gran");
  if ((wiggle || wshear) && wallstyle == REGION)
    error->all(FLERR,"Cannot wiggle or shear with fix wall/gran/region");

  // setup oscillations

  if (wiggle) omega = 2.0*MY_PI / period;

  // perform initial allocation of atom-based arrays
  // register with Atom class

  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  if (peratom_flag) {
    clear_stored_contacts();
  }

  time_origin = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

FixWallSpherharm::~FixWallSpherharm()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,Atom::GROW);
  atom->delete_callback(id,Atom::RESTART);

  // delete local storage

  delete [] idregion;

  memory->destroy(weights);
  memory->destroy(abscissa);
}

/* ---------------------------------------------------------------------- */

int FixWallSpherharm::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallSpherharm::init()
{
  dt = update->dt;

  avec = (AtomVecSpherharm *) atom->style_match("spherharm");
  if (!avec) error->all(FLERR,"Pair SH requires atom style shperatom");

  get_quadrature_values(num_pole_quad);
}

/* ---------------------------------------------------------------------- */

void FixWallSpherharm::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallSpherharm::post_force(int /*vflag*/)
{
  double dx,dy,dz,del1,del2,delxy,delr,rsq;
  double vwall[3];
  double gamma, radi;

  // set position of wall to initial settings and velocity to 0.0
  // if wiggle or shear, set wall position and velocity accordingly

  double wlo = lo;
  double whi = hi;
  vwall[0] = vwall[1] = vwall[2] = 0.0;
  if (wiggle) {
    double arg = omega * (update->ntimestep - time_origin) * dt;
    if (wallstyle == axis) {
      wlo = lo + amplitude - amplitude*cos(arg);
      whi = hi + amplitude - amplitude*cos(arg);
    }
    vwall[axis] = amplitude*omega*sin(arg);
  } else if (wshear) vwall[axis] = vshear;

  // loop over all my atoms
  // rsq = distance from wall
  // dx,dy,dz = signed distance from wall
  // for rotating cylinder, reset vwall based on particle position
  // skip atom if not close enough to wall
  //   if wall was set to a null pointer, it's skipped since lo/hi are infinity
  // compute force and torque on atom if close enough to wall
  //   via wall potential matched to pair potential
  // set history if pair potential stores history

  double **x = atom->x;
  double **f = atom->f;
  double **quat = atom->quat;
  double **torque = atom->torque;
  double *max_rad = atom->maxrad_byshape;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *shtype = atom->shtype;
  int ishtype;
  double h, r_i;

  if (peratom_flag) {
    clear_stored_contacts();
  }

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      dx = dy = dz = 0.0;

      ishtype = shtype[i];
      radi = max_rad[ishtype];

      if (wallstyle == XPLANE) {
        del1 = x[i][0] - wlo;
        del2 = whi - x[i][0];
        if (del1 < del2) dx = del1;
        else dx = -del2;
        gamma = std::acos(std::abs(dx) /radi) + (0.5 * MY_PI / 180.0);
      } else if (wallstyle == YPLANE) {
        del1 = x[i][1] - wlo;
        del2 = whi - x[i][1];
        if (del1 < del2) dy = del1;
        else dy = -del2;
        gamma = std::acos(std::abs(dy) /radi) + (0.5 * MY_PI / 180.0);
      } else if (wallstyle == ZPLANE) {
        del1 = x[i][2] - wlo;
        del2 = whi - x[i][2];
        if (del1 < del2) dz = del1;
        else dz = -del2;
        gamma = std::acos(std::abs(dz) /radi) + (0.5 * MY_PI / 180.0);
      } else if (wallstyle == ZCYLINDER) {
        delxy = sqrt(x[i][0]*x[i][0] + x[i][1]*x[i][1]);
        delr = cylradius - delxy;
        if (delr > radi) { // why not abs(delr) here? Just checking positive case is strange
          dz = cylradius;
        } else {
          dx = -delr/delxy * x[i][0];
          dy = -delr/delxy * x[i][1];
          // rwall = -2r_c if inside cylinder, 2r_c outside
          if (delxy < cylradius){ // inside cylinder
            h = 0.5 + (((radi*radi) - (cylradius*cylradius)) / (2.0 * delxy * delxy));
            r_i = std::sqrt((radi*radi) - (h*h*delxy*delxy));
            gamma =  std::asin(r_i/radi) + (0.5 * MY_PI / 180.0);
          }
          else{ // outside cylinder
            gamma = std::acos(delr /radi) + (0.5 * MY_PI / 180.0);
          }
          if (wshear && axis != 2) {
            vwall[0] += vshear * x[i][1]/delxy;
            vwall[1] += -vshear * x[i][0]/delxy;
            vwall[2] = 0.0;
          }
        }
      }

      rsq = dx*dx + dy*dy + dz*dz;

      if (rsq <= radi*radi) {

        // store contact info
        if (peratom_flag) {
          array_atom[i][0] = 1.0;
          array_atom[i][4] = x[i][0] - dx;
          array_atom[i][5] = x[i][1] - dy;
          array_atom[i][6] = x[i][2] - dz;
          array_atom[i][7] = radi;
        }

        // invoke sphere/wall interaction
        double *contact;
        if (peratom_flag)
          contact = array_atom[i];
        else
          contact = nullptr;

        if (pairstyle == VOLUME_BASED)
          vol_based(dx,dy,dz,gamma,ishtype,quat[i],x[i],f[i],torque[i],contact);
      }
    }
  }
}

void FixWallSpherharm::clear_stored_contacts() {
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    for (int m = 0; m < size_peratom_cols; m++) {
      array_atom[i][m] = 0.0;
    }
  }
}


void FixWallSpherharm::vol_based(double dx, double dy, double dz, double iang,
                                 int ishtype, double *quat, double *x, double *f,
                                 double *torque, double *contact) {

  int kk_count;
  double vol_overlap,pn;
  double torsum[3],iforce[3],delvec[3];
  double irot[3][3];
  double iquat_sf_bf[4],iquat_cont[4];
  bool candidates_found;

  delvec[0] = -dx; // vector from particle to wall
  delvec[1] = -dy;
  delvec[2] = -dz;

  kk_count = -1;
  vol_overlap = 0.0;
  MathExtra::zero3(iforce);
  MathExtra::zero3(torsum);

  // Calculate the rotation matrix for the quaternion for atom i
  MathExtra::quat_to_mat(quat, irot);
  // Quaternion to get from space frame to body frame for atom "i"
  MathExtra::qconjugate(quat, iquat_sf_bf);
  MathExtra::qnormalize(iquat_sf_bf);

  // Get the quaternion from north pole of atom "i" to the vector connecting the centre line of atom "i" to the wall
  get_contact_quat(delvec, iquat_cont);

  candidates_found = refine_cap_angle(kk_count, ishtype, iang, iquat_cont, iquat_sf_bf, x, delvec);

  if (kk_count > num_pole_quad) kk_count = num_pole_quad; // don't refine if points on first layer

  if (candidates_found) {
    calc_force_torque(kk_count, ishtype, iang, iquat_cont, iquat_sf_bf, x, irot, vol_overlap, iforce, torsum, delvec);

    if (vol_overlap==0.0) return;

    pn = mexpon * kn * std::pow(vol_overlap, mexpon - 1.0);
    MathExtra::scale3(-pn, iforce);          // F_n = -p_n * S_n (S_n = factor*iforce)
    MathExtra::scale3(-pn, torsum);          // M_n
    MathExtra::add3(f, iforce, f);           // Force and torque on particle a
    MathExtra::add3(torque, torsum, torque);

    if (peratom_flag) {
      contact[1] = iforce[0];
      contact[2] = iforce[1];
      contact[3] = iforce[2];
    }

    static int file_count = 0;
    avec->dump_ply(0,ishtype,file_count,irot,x);
    file_count++;

  }

}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
int FixWallSpherharm::refine_cap_angle(int &kk_count, int ishtype, double iang, double (&iquat_cont)[4],
                             double (&iquat_sf_bf)[4], const double xi[3], const double delvec[3]){

  int kk, ll, n;
  double theta_pole, phi_pole, theta, phi;
  double rad_body, dtemp, cosang;
  double numer, denom;
  double ix_sf[3], gp[3], gp_bf[3], gp_sf[3], wall_normal[3];
  double quat[4];
  double rot_np_bf[3][3], rot_np_sf[3][3];

  MathExtra::normalize3(delvec, wall_normal); // surface point to unit vector
  numer = MathExtra::dot3(delvec, wall_normal);

  MathExtra::quat_to_mat(iquat_cont, rot_np_sf);
  MathExtra::quatquat(iquat_sf_bf, iquat_cont, quat);
  MathExtra::qnormalize(quat);
  MathExtra::quat_to_mat(quat, rot_np_bf);

  n = 2*(num_pole_quad-1);
  cosang = std::cos(iang);

  for (kk = num_pole_quad-1; kk >= 0; kk--) { // start from widest angle to allow early stopping
    theta_pole = std::acos((abscissa[kk]*((1.0-cosang)/2.0)) + ((1.0+cosang)/2.0));
    for (ll = 1; ll <= n+1; ll++) {
      phi_pole = MY_2PI * double(ll-1) / (double(n + 1));

      gp[0] = std::sin(theta_pole)*std::cos(phi_pole); // quadrature point at [0,0,1]
      gp[1] = std::sin(theta_pole)*std::sin(phi_pole);
      gp[2] = std::cos(theta_pole);

      MathExtra::matvec(rot_np_bf, gp, gp_bf); // quadrature point at contact in body frame
      phi = std::atan2(gp_bf[1], gp_bf[0]);
      phi = phi > 0.0 ? phi : MY_2PI + phi;
      theta = std::acos(gp_bf[2]);

      rad_body = avec->get_shape_radius(ishtype, theta, phi);

      MathExtra::matvec(rot_np_sf, gp, gp_sf); // quadrature point at contact in space frame
      phi = std::atan2(gp_sf[1], gp_sf[0]);
      phi = phi > 0.0 ? phi : MY_2PI + phi;
      theta = std::acos(gp_sf[2]);

      ix_sf[0] = (rad_body * sin(theta) * cos(phi));
      ix_sf[1] = (rad_body * sin(theta) * sin(phi));
      ix_sf[2] = (rad_body * cos(theta));

      MathExtra::norm3(ix_sf); // surface point to unit vector
      denom = MathExtra::dot3(ix_sf, wall_normal);
      dtemp = numer/denom;

      // Check for contact
      if (rad_body>dtemp) {
        kk_count = kk+1; // refine the spherical cap angle to this index (+1 as points could exist between indexes)
        return 1;
      }
    }
  }
  return 0;
}
/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
void FixWallSpherharm::calc_force_torque(int kk_count, int ishtype, double iang, double (&iquat_cont)[4],
                                         double (&iquat_sf_bf)[4], const double xi[3], double (&irot)[3][3],
                                         double &vol_overlap, double (&iforce)[3], double (&torsum)[3],
                                         double delvec[3]){

  int kk, ll, n;
  double cosang, fac;
  double theta_pole, phi_pole;
  double theta_bf, phi_bf, theta_sf, phi_sf;
  double rad_body;
  double ix_sf[3], x_testpoint[3];

  double rad_wall, dv, numer, denom;
  double inorm_bf[3], inorm_sf[3], dtor[3];
  double gp[3], gp_bf[3], gp_sf[3];
  double wall_normal[3],line_normal[3];
  double quat[4];
  double rot_np_bf[3][3], rot_np_sf[3][3];

  static int file_count = 0;
  bool first_call = true;

  MathExtra::normalize3(delvec, wall_normal); // surface point to unit vector
  numer = MathExtra::dot3(delvec, wall_normal);

  MathExtra::quat_to_mat(iquat_cont, rot_np_sf);
  MathExtra::quatquat(iquat_sf_bf, iquat_cont, quat);
  MathExtra::qnormalize(quat);
  MathExtra::quat_to_mat(quat, rot_np_bf);

  n = 2*(num_pole_quad-1);
  cosang = std::cos(iang);
  iang = std::acos((abscissa[kk_count]*((1.0-cosang)/2.0)) + ((1.0+cosang)/2.0)); //refine spherical cap angle
  cosang = std::cos(iang);
  fac = ((1.0-cosang)/2.0)*(MY_2PI/double(n+1));

  for (kk = num_pole_quad-1; kk >= 0; kk--) {
    theta_pole = std::acos((abscissa[kk]*((1.0-cosang)/2.0)) + ((1.0+cosang)/2.0));
    for (ll = 1; ll <= n+1; ll++) {
      phi_pole = MY_2PI * double(ll-1) / (double(n + 1));

      gp[0] = std::sin(theta_pole)*std::cos(phi_pole); // quadrature point at [0,0,1]
      gp[1] = std::sin(theta_pole)*std::sin(phi_pole);
      gp[2] = std::cos(theta_pole);

      MathExtra::matvec(rot_np_sf, gp, gp_sf); // quadrature point at contact in space frame
      phi_sf = std::atan2(gp_sf[1], gp_sf[0]);
      phi_sf = phi_sf > 0.0 ? phi_sf : MY_2PI + phi_sf;
      theta_sf = std::acos(gp_sf[2]);

      MathExtra::matvec(rot_np_bf, gp, gp_bf); // quadrature point at contact in body frame
      phi_bf = std::atan2(gp_bf[1], gp_bf[0]);
      phi_bf = phi_bf > 0.0 ? phi_bf : MY_2PI + phi_bf;
      theta_bf = std::acos(gp_bf[2]);

      // Get the radius at the body frame theta and phi value and normal [not unit]
      rad_body = avec->get_shape_radius_and_normal(ishtype, theta_bf, phi_bf, inorm_bf); // inorm is in body frame

      ix_sf[0] = (rad_body * sin(theta_sf) * cos(phi_sf));
      ix_sf[1] = (rad_body * sin(theta_sf) * sin(phi_sf));
      ix_sf[2] = (rad_body * cos(theta_sf));

      MathExtra::normalize3(ix_sf, line_normal);
      denom = MathExtra::dot3(line_normal, wall_normal);
      rad_wall = numer/denom;

//      std::cout<<kk<<" "<<ll<<" "<<" "<<rad_wall<<" "<<" "<<rad_body<<std::endl;

      // Check for contact
      if (rad_body>rad_wall) {

        MathExtra::add3(ix_sf, xi, ix_sf);

        dv = weights[kk] * (std::pow(rad_body, 3) - std::pow(rad_wall, 3));
        vol_overlap += dv;

        MathExtra::scale3(weights[kk]/std::sin(theta_bf), inorm_bf); // w_i * n * Q
        MathExtra::matvec(irot, inorm_bf, inorm_sf);            // w_i * n * Q in space frame
        MathExtra::add3(iforce, inorm_sf, iforce);              // sum(w_i * n * Q)
        MathExtra::sub3(ix_sf, xi, x_testpoint);                // Vector u from centre of "a" to surface point
        MathExtra::cross3(x_testpoint, inorm_sf, dtor);         // u x n_s * Q * w_i
        MathExtra::add3(torsum, dtor, torsum);                  // sum(u x n_s * Q * w_i)

        write_surfpoints_to_file(ix_sf, 1, inorm_sf, file_count, first_call);
        first_call = false;

      } // check_contact
    } // ll (quadrature)
  } // kk (quadrature)
  file_count++;
  vol_overlap*=fac/3.0;
  MathExtra::scale3(fac, iforce);
  MathExtra::scale3(fac, torsum);
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void FixWallSpherharm::get_contact_quat(double (&xvecdist)[3], double (&quat)[4]){
  double vert_unit_vec[3], cross_vec[3], c;
  // North pole unit vector, points generated are with reference to this point
  vert_unit_vec[0] = 0;
  vert_unit_vec[1] = 0;
  vert_unit_vec[2] = 1.0;
  c = MathExtra::dot3(vert_unit_vec, xvecdist);
  MathExtra::cross3(vert_unit_vec, xvecdist, cross_vec);
  quat[1] = cross_vec[0];
  quat[2] = cross_vec[1];
  quat[3] = cross_vec[2];
  quat[0] = sqrt(MathExtra::lensq3(vert_unit_vec) * MathExtra::lensq3(xvecdist)) + c;
  MathExtra::qnormalize(quat);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixWallSpherharm::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 0.0;
  // store contacts
  if (peratom_flag) bytes += nmax*size_peratom_cols*sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixWallSpherharm::grow_arrays(int nmax)
{
  if (peratom_flag) {
    memory->grow(array_atom,nmax,size_peratom_cols,"fix_wall_spherharm:array_atom");
  }
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixWallSpherharm::copy_arrays(int i, int j, int /*delflag*/)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[j][m] = array_atom[i][m];
  }
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixWallSpherharm::set_arrays(int i)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[i][m] = 0;
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixWallSpherharm::pack_exchange(int i, double *buf)
{
  int n = 0;
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      buf[n++] = array_atom[i][m];
  }
  return n;
}

/* ----------------------------------------------------------------------
   unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- */

int FixWallSpherharm::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[nlocal][m] = buf[n++];
  }
  return n;
}

/* ---------------------------------------------------------------------- */

void FixWallSpherharm::reset_dt()
{
  dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixWallSpherharm::get_quadrature_values(int num_quadrature){

  memory->create(weights, num_quadrature, "PairSH:weights");
  memory->create(abscissa, num_quadrature, "PairSH:abscissa");

  MathSpherharm::QuadPair p;
  // Get the quadrature weights, and abscissa.
  for (int i = 0; i < num_quadrature; i++) {
    p = MathSpherharm::GLPair(num_quadrature, i + 1);
    weights[i] = p.weight;
    abscissa[i] = p.x();
  }
}

void FixWallSpherharm::write_surfpoints_to_file(double *x, int cont, double *norm, int file_count, bool first_call){

  std::ofstream outfile;
  if (!first_call){
    outfile.open("test_dump/surfpoint_"+std::to_string(file_count)+".csv", std::ios_base::app);
    if (outfile.is_open()) {
        outfile << std::setprecision(16) << x[0] << "," << x[1] << "," << x[2] << "," << cont <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
      outfile.close();
    } else std::cout << "Unable to open file";
  }
  else {
    outfile.open("test_dump/surfpoint_" + std::to_string(file_count) + ".csv");
    if (outfile.is_open()) {
      outfile << "x,y,z,cont,nx,ny,nz" << "\n";
        outfile << std::setprecision(16) << x[0] << "," << x[1] << "," << x[2] << "," << cont <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
      outfile.close();
    } else std::cout << "Unable to open file";
  }
};

