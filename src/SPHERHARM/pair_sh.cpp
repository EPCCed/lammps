/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_sh.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "math_extra.h"
#include "atom_vec_spherharm.h"
#include "math_spherharm.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EPSILON 1e-10
/* ---------------------------------------------------------------------- */

PairSH::PairSH(LAMMPS *lmp) : Pair(lmp)
{

  no_virial_fdotr_compute = 1;

  // Single steps are for force and energy of a single pairwise interaction between 2 atoms
  // Energy calculation not enabled, as we don't yet have pairwise potential
  single_enable = 0;
  restartinfo = 0; // Not figured out how to do this yet
  writedata = 0; // Ditto
  respa_enable = 0;

  // Flag indicating if lammps types have been matches with SH type.
  matchtypes = 0;
  exponent = -1.0;

  cur_time = 0.0; // for temp file writing
  file_count = 0; // for temp file writing

//  num_pole_quad = 30;
//  radius_tol = 1e-3; // 0.1%

  num_pole_quad = 30;
  radius_tol = 1e-8; // %, for very small overlaps this percentage tolerance must be very fine
}

/* ---------------------------------------------------------------------- */

PairSH::~PairSH()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(normal_coeffs);
    memory->destroy(typetosh);

    memory->destroy(weights);
    memory->destroy(abscissa);
  }
}

/* ---------------------------------------------------------------------- */
void PairSH::compute(int eflag, int vflag)
{
  int i,j,ii,jj;
  int inum,jnum,itype,jtype,ishtype,jshtype;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double fpair,radi,radj,r,rsq,iang;
  double irot[3][3],jrot[3][3];
  double x_testpoint[3],delvec[3];
  double iquat_sf_bf[4],iquat_cont[4];

  ev_init(eflag,vflag);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int *shtype = atom->shtype;
  double **quat = atom->quat;
  int nlocal = atom->nlocal;
  double **torque = atom->torque;
  double *max_rad = atom->maxrad_byshape;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int me,kk_count;
  bool first_call,candidates_found, cont_calc;
  double vol_overlap,factor,pn,fn;
  double torsum[3],xcont[3], iforce[3];
  double irot_cont[3][3];
  int maxshexpan;

  maxshexpan = avec->get_max_expansion();
  file_count++;
  MPI_Comm_rank(world,&me);

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    MathExtra::copy3(x[i], delvec);
    itype = type[i];
    ishtype = shtype[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    radi = max_rad[ishtype];

    // Calculate the rotation matrix for the quaternion for atom i
    MathExtra::quat_to_mat(quat[i], irot);
    // Quaternion to get from space frame to body frame for atom "i"
    MathExtra::qconjugate(quat[i], iquat_sf_bf);
    MathExtra::qnormalize(iquat_sf_bf);

//    std::cout << i << " " << jnum << " " << x[i][0] << " " << x[i][1] << " " << x[i][2] << std::endl;

    for (jj = 0; jj < jnum; jj++) {
      cont_calc = false;
      j = jlist[jj];
      j &= NEIGHMASK;
      MathExtra::copy3(x[i], delvec);
      MathExtra::sub3(delvec, x[j], delvec);
      jshtype = shtype[j];
      radj = max_rad[jshtype];
      rsq = MathExtra::lensq3(delvec);
      r = sqrt(rsq);
      jtype = type[j];

      kk_count = -1;
      first_call = true;
      vol_overlap = 0.0;
      MathExtra::zero3(iforce);
      MathExtra::zero3(torsum);

      //std::cout << std::setprecision(12)<< i << " " << j << " " << r  << " " << x[i][0] << " " << x[i][1] << " " << x[i][2] << " " << x[j][0] << " " << x[j][1] << " " << x[j][2] << std::endl;

//      std::cout << radi << " " << radj << " " << r << std::endl;

      if (r<radi+radj) {

        if (r > radj) { // Particle i's centre is not inside Particle j, can use spherical cap from particle i
//          iang =  std::asin(radj/r) + (0.5 * MY_PI / 180.0); // Adding half a degree to ensure that circumference is populated
//          iang = std::asin(radj / r);

          double h = 0.5 + (radi * radj - radj * radj)/(2.0 * r*r);
          double r_i = std::sqrt(radi*radi - h*h*r*r);
          iang = std::asin(r_i/radi);


//          std::cout << std::setprecision(9) << r << " " << r_i << " " << r_i/radi << " " << iang*180/MY_PI << std::endl;

        }
          // TODO Add condition for particle "j" were the code will swap to particle "j" becoming the primary if "i" is not feasible
        else { // Can't use either spherical cap
          error->all(FLERR, "Error, centre within radius!");
        }

        // Get the quaternion from north pole of atom "i" to the vector connecting the centre line of atom "i" and "j".
        MathExtra::negate3(delvec);
//        get_contact_quat(delvec, iquat_cont);
        MathSpherharm::get_contact_quat(delvec, iquat_cont);
        // Quaternion of north pole to contact for atom "i"
        MathExtra::quat_to_mat(iquat_cont, irot_cont);
        // Calculate the rotation matrix for the quaternion for atom j
        MathExtra::quat_to_mat(quat[j], jrot);
        cur_time += (update->dt) / 1000; // for temp file writing

        // If the max expansion is 0, then we just have sphere-sphere overlap, otherwise confirm overlap
        if (maxshexpan!=0) {
          candidates_found = refine_cap_angle(kk_count, ishtype, jshtype, iang, radj, iquat_cont,
                                              iquat_sf_bf, x[i], x[j], jrot);
          if (kk_count == num_pole_quad) kk_count = num_pole_quad-1;
        }
        else{
          candidates_found = true;
          kk_count = num_pole_quad-1;
        }


        if (candidates_found) {

          // If the max expansion is !=0, then we require quadrature points
//          if (maxshexpan!=0){
            calc_norm_force_torque(kk_count, ishtype, jshtype, iang, radi, radj, iquat_cont, iquat_sf_bf, x[i], x[j],
                                   irot,jrot, vol_overlap, iforce, torsum, factor, first_call, ii, jj);
//          }
//          else{ // simplified case of sphere-sphere overlap
//            sphere_sphere_norm_force_torque(radi, radj, radi+radj-r, x[i], x[j], iforce, torsum, vol_overlap);
//          }

//        if (vol_overlap==0.0) continue;

//          double vol_check = MathSpherharm::get_sphere_overlap_volume(radi, radj, r);
//          std::cout<<vol_overlap<<" "<<vol_check<<" "<<100.0*std::abs(vol_overlap-vol_check)/vol_overlap <<std::endl;

          fpair = normal_coeffs[itype][jtype][0];
          pn = exponent * fpair * std::pow(vol_overlap, exponent - 1.0);
          MathExtra::scale3(-pn, iforce);    // F_n = -p_n * S_n (S_n = factor*iforce)
          MathExtra::scale3(-pn, torsum);    // M_n

          // Force and torque on particle a
          MathExtra::add3(f[i], iforce, f[i]);
          MathExtra::add3(torque[i], torsum, torque[i]);

//          double hf = (4.0/3.0)*69600000000.0*std::sqrt(radi)*std::pow(radi+radj-r,1.5);
//          std::cout << hf << " " << MathExtra::len3(iforce) << " " << 100*(hf-MathExtra::len3(iforce))/hf<<std::endl;

//          std::cout << i << " " << j << " " << r  << " " << x[i][0] << " " << x[i][1] << " " << x[i][2] << " " << x[j][0] << " " << x[j][1] << " " << x[j][2] << std::endl;
//          std::cout << i << " " << j << " " << atom->tag[j] << " " << iforce[0] << " " << iforce[1] << " " << iforce[2] << " " << radi+radj-r << std::endl;

          // N.B on a single proc, N3L is always imposed, regardless of Newton On/Off
          if (force->newton_pair || j < nlocal) {
            cont_calc = true;
            // Force on particle b
            MathExtra::sub3(f[j], iforce, f[j]);
            // Torque on particle b
            fn = MathExtra::len3(iforce);
            MathExtra::cross3(torsum, iforce, xcont);       // M_n x F_n
            MathExtra::scale3(-1.0 / (fn * fn), xcont);  // (M_n x F_n)/|F_n|^2 [Swap direction due to cross, normally x_c X F_n]
            MathExtra::add3(xcont, x[i], xcont);            // x_c global cords
            MathExtra::sub3(xcont, x[j], x_testpoint);      // Vector from centre of "b" to contact point
            MathExtra::cross3(iforce, x_testpoint, torsum); // M_n' = F_n x (x_c - x_b)
            MathExtra::add3(torque[j], torsum, torque[j]);

          } // newton_pair

        if (evflag) {
          ev_tally_spherharm_sphere(i, j, nlocal, force->newton_pair,
                             iforce[0], iforce[1], iforce[2],
                             x[i][0], x[i][1], x[i][2], avec->get_shape_volume(ishtype),
                             x[j][0], x[j][1], x[j][2], avec->get_shape_volume(jshtype),
                             radi, radj);
//          ev_tally_xyz(i,j,nlocal,force->newton_pair,0.0,0.0,iforce[0], iforce[1], iforce[2],-delvec[0],-delvec[1],-delvec[2]);
        }
//        else
//        {
//          std::cout << "false" << std::endl;
//        }
//      avec->dump_ply(i,ishtype,file_count,irot,x[i]);
//      avec->dump_ply(j,jshtype,file_count,jrot,x[j]);

        } // candidates found

//      kk_count = -1;
//      double jang =  std::asin(radi/r) + (0.5 * MY_PI / 180.0);
//      MathExtra::negate3(delvec);
//      double jquat_cont[4], jquat_sf_bf[4];
//      get_contact_quat(delvec, jquat_cont);
//      MathExtra::qconjugate(quat[j], jquat_sf_bf);
//      MathExtra::qnormalize(jquat_sf_bf);
//
//      candidates_found = refine_cap_angle(kk_count,jshtype,ishtype,jang,radi,jquat_cont,jquat_sf_bf,x[j],x[i],irot);
//      if (kk_count == 0) kk_count = 1;
//
//      vol_overlap = 0.0;
//      MathExtra::zero3(iforce);
//      MathExtra::zero3(torsum);
//      if (candidates_found) calc_force_torque(kk_count,jshtype,ishtype,jang,radi,jquat_cont,jquat_sf_bf,x[j],x[i],jrot,
//                    irot,vol_overlap,iforce,torsum,factor,first_call,jj,ii);
//      std::cout<<"Vol j : " << std::setprecision(16) << vol_overlap << std::endl;
//      std::cout<<"A j : " << std::setprecision(16) << iforce[0] << " " << iforce[1] << " " << iforce[2] << " " << MathExtra::len3(iforce) << std::endl;
//      fpair = normal_coeffs[itype][jtype][0];
//      pn  = exponent * fpair * std::pow(vol_overlap, exponent-1.0);
//      MathExtra::scale3(-pn, iforce);    // F_n = -p_n * S_n (S_n = factor*iforce)
//      MathExtra::scale3(-pn, torsum);    // M_n
//      std::cout<<"F j : " << std::setprecision(16) << iforce[0] << " " << iforce[1] << " " << iforce[2] << " " << MathExtra::len3(iforce) << std::endl;

      } // bounding spheres
    } // jj
  } // ii
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSH::allocate()
{
  allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag,n+1,n+1,"pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++)
            setflag[i][j] = 0;

    memory->create(cutsq,n+1,n+1,"pair:cutsq");
    memory->create(cut,n+1,n+1,"pair:cut");
    memory->create(normal_coeffs,n+1,n+1,1,"pair:normal_coeffs");
    memory->create(typetosh,n+1,"pair:typetosh");
}

/* ----------------------------------------------------------------------
   global settings
   JY - Not defining a global cut off, as this must come from the
   atom style, where the maximum particle radius is stored
------------------------------------------------------------------------- */

void PairSH::settings(int narg, char **arg) {
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");

  avec = (AtomVecSpherharm *) atom->style_match("spherharm");
  if (!avec) error->all(FLERR,"Pair SH requires atom style shperatom");

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   JY - Only type pairs are defined here, no other parameters. The global
   cutoff is taken from the atom style here.
------------------------------------------------------------------------- */

void PairSH::coeff(int narg, char **arg)
{

  if (narg != 4)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  double normal_coeffs_one, exponent_in;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
  normal_coeffs_one = utils::numeric(FLERR,arg[2],false,lmp);// kn
  exponent_in = utils::numeric(FLERR,arg[3],false,lmp);// m

  if (exponent==-1){
    exponent=exponent_in;
  }
  else if(exponent!=exponent_in){
    error->all(FLERR,"Exponent must be equal for all type interactions, exponent mixing not developed");
  }

  // Linking the Types to the SH Types, needed for finding the cut per Type
  if (!matchtypes) matchtype();

  int count = 0;
  int shi, shj;
  double *max_rad = atom->maxrad_byshape;

  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      shi = typetosh[i];
      shj = typetosh[j];
      cut[i][j] = max_rad[shi]+max_rad[shj];
      setflag[i][j] = 1;
      normal_coeffs[i][j][0] = normal_coeffs_one;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   JY - Each type can only use one Spherical Harmonic Particle type. This
   method associates a SH particle type with the atom->types. Required for
   finding the cut[i][j] between types which is then used in the neighbour
   searching.
------------------------------------------------------------------------- */
void PairSH::matchtype()
{

  matchtypes = 1;

  int nlocal = atom->nlocal;
  int *shtype = atom->shtype;
  int *type = atom->type;

  for (int i = 0; i <= atom->ntypes; i++) {
    typetosh[i] = -1;
  }

  for (int i = 0; i < nlocal; i++) {
    if (typetosh[type[i]]==-1) {
      typetosh[type[i]] = shtype[i];
    }
    else if(typetosh[type[i]] != shtype[i]){
      error->all(FLERR,"Types must have same Spherical Harmonic particle type");
    }
  }

  // Possibility that atoms on different processors may have associated different
  // SH particle types with atom->types. This will not be caught here and the maximum
  // will be taken.
  MPI_Allreduce(MPI_IN_PLACE,typetosh,atom->ntypes+1,MPI_INT,MPI_MAX,world);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSH::init_style()
{
  neighbor->request(this,instance_me);
  get_quadrature_values(num_pole_quad);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   JY - Need to set up for different types, although both types must use the
   spherical harmonic atom style. Maximum radius of type pair is used for cut.
   The only mixing would be in the coefficients used in the contact model,
   i.e stiffness, but this will need to be explored later
   These coefficients wouldn't even be mixed if using F_i = K_i*V*n_i (bad model)
------------------------------------------------------------------------- */

double PairSH::init_one(int i, int j)
{
  int shi, shj;
  double *max_rad = atom->maxrad_byshape;

  // No epsilon and no sigma used for the spherical harmonic atom style
  if (setflag[i][j] == 0) {
    shi = typetosh[i];
    shj = typetosh[j];
    cut[i][j] = max_rad[shi]+max_rad[shj];
  }

  // TODO - Just use the first coefficient for the pair, no mixing
  normal_coeffs[i][j][0] = normal_coeffs[j][i][0] = normal_coeffs[i][i][0];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   JY - Calculates the quaternion required to rotate points generated
   on the (north) pole of an atom back to the vector between two atom centres.
   https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another

    TODO - Need to comapre this against generating the quaterion from the
    spherical coordinates (theta,phi) of the contact line from the COG of
    each particle in space frame
 ------------------------------------------------------------------------- */
void PairSH::get_contact_quat(double (&xvecdist)[3], double (&quat)[4]){

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


void PairSH::get_quadrature_values(int num_quadrature) {

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

int PairSH::refine_cap_angle(int &kk_count, int ishtype, int jshtype, double iang,  double radj,
                             double (&iquat_cont)[4], double (&iquat_sf_bf)[4], const double xi[3],
                             const double xj[3], double (&jrot)[3][3]){

  int kk, ll, n;
  double theta_pole, phi_pole, theta, phi, theta_proj, phi_proj;
  double rad_body, dtemp, finalrad, cosang;
  double ix_sf[3], x_testpoint[3], x_projtestpoint[3];
  double gp[3], gp_bf[3], gp_sf[3];
  double quat[4];
  double rot_np_bf[3][3], rot_np_sf[3][3];

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

      ix_sf[0] = (rad_body * sin(theta) * cos(phi)) + xi[0];
      ix_sf[1] = (rad_body * sin(theta) * sin(phi)) + xi[1];
      ix_sf[2] = (rad_body * cos(theta)) + xi[2];
      // vector distance from COG of atom j (in space frame) to test point on atom i
      MathExtra::sub3(ix_sf, xj, x_testpoint);
      // scalar distance
      dtemp = MathExtra::len3(x_testpoint);

//      std::cout << std::setprecision(9)<< dtemp << " " << radj << " " << iang*180/MY_PI << std::endl;
//      std::cout << std::setprecision(9)<< xi[0] << " "<< xi[1] << " "<< xi[2] << " "<< xj[0] << " "<< xj[1] << " "<< xj[2]  << std::endl;
//      std::cout << std::setprecision(9)<< ix_sf[0] << " "<< ix_sf[1] << " "<< ix_sf[2] << " "<< xj[0] << " "<< xj[1] << " "<< xj[2]  << std::endl;


      if (dtemp > radj) continue;
      // Rotating the projected point into atom j's body frame (rotation matrix transpose = inverse)
      MathExtra::transpose_matvec(jrot, x_testpoint, x_projtestpoint);
      // Get projected phi and theta angle of gauss point in atom i's body frame
      phi_proj = std::atan2(x_projtestpoint[1], x_projtestpoint[0]);
      phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
      theta_proj = std::acos(x_projtestpoint[2] / dtemp);

      // Check for contact
      if (avec->check_contact(jshtype, phi_proj, theta_proj, dtemp, finalrad)) {
        kk_count = kk+1; // refine the spherical cap angle to this index (+1 as points could exist between indexes)
        return 1;
      }
    }
  }
  return 0;
}

void PairSH::calc_norm_force_torque(int kk_count, int ishtype, int jshtype, double iang, double radi, double radj,
                               double (&iquat_cont)[4], double (&iquat_sf_bf)[4], const double xi[3],
                               const double xj[3], double (&irot)[3][3],  double (&jrot)[3][3],
                               double &vol_overlap, double (&iforce)[3], double (&torsum)[3],
                               double &factor, bool &first_call, int ii, int jj){

  int kk, ll, n;
  double cosang, fac, radtol, st;
  double theta_pole, phi_pole, theta_proj, phi_proj;
  double theta_bf, phi_bf, theta_sf, phi_sf;
  double rad_body, dtemp, finalrad;
  double ix_sf[3], x_testpoint[3], x_projtestpoint[3];

  double rad_sample, dv;
  double inorm_bf[3], inorm_sf[3], dtor[3];
  double gp[3], gp_bf[3], gp_sf[3];
  double quat[4];
  double rot_np_bf[3][3], rot_np_sf[3][3];

  MathExtra::quat_to_mat(iquat_cont, rot_np_sf);
  MathExtra::quatquat(iquat_sf_bf, iquat_cont, quat);
  MathExtra::qnormalize(quat);
  MathExtra::quat_to_mat(quat, rot_np_bf);

  radtol = radius_tol*radi/100.0; // fraction of max radius
  n = 2*(num_pole_quad-1);
  cosang = std::cos(iang);
  // Refine spherical cap angle
  // The 1.0/abscissa[0] term is to ensure that quadrature points populate the outer "shell" or else they would be offset
  // by whatever the difference between 1 and abscissa[0] is.
  iang = std::acos((1.0/abscissa[0])*(abscissa[kk_count]*((1.0-cosang)/2.0)) + ((1.0+cosang)/2.0));
  cosang = std::cos(iang);
  fac = ((1.0-cosang)/2.0)*(MY_2PI/double(n+1));

  for (kk = num_pole_quad-1; kk >= 0; kk--) {
//  for (kk = 0; kk < num_pole_quad; kk++) {
    theta_pole = std::acos((abscissa[kk]*((1.0-cosang)/2.0)) + ((1.0+cosang)/2.0));
    for (ll = 1; ll <= n+1; ll++) {
      phi_pole = MY_2PI * double(ll-1) / (double(n + 1));

      st = std::sin(theta_pole);
      gp[0] = st*std::cos(phi_pole); // quadrature point at [0,0,1]
      gp[1] = st*std::sin(phi_pole);
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

      st = rad_body*std::sin(theta_sf);
      ix_sf[0] = (st * cos(phi_sf)) + xi[0]; // Global coordinates of quadrature point
      ix_sf[1] = (st * sin(phi_sf)) + xi[1];
      ix_sf[2] = (rad_body * cos(theta_sf)) + xi[2];
      // vector distance from COG of atom j (in space frame) to quadrature point on atom i
      MathExtra::sub3(ix_sf, xj, x_testpoint);
      // scalar distance
      dtemp = MathExtra::len3(x_testpoint);
      if (dtemp > radj) continue;
      // Rotating the projected point into atom j's body frame (rotation matrix transpose = inverse)
      MathExtra::transpose_matvec(jrot, x_testpoint, x_projtestpoint);
      // Get projected phi and theta angle of gauss point in atom j's body frame
      phi_proj = std::atan2(x_projtestpoint[1], x_projtestpoint[0]);
      phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
      theta_proj = std::acos(x_projtestpoint[2] / dtemp);

      // Check for contact between the quadrature point on atom i and the surface of atom j (at the angular coordinates
      // as projected from the quadrature point onto the surface of particle j
      if (avec->check_contact(jshtype, phi_proj, theta_proj, dtemp, finalrad)) {

        // Get the intersection between the straight line from the quadrature point on atom i to the intersection with
        // the surface of atom j.
//        rad_sample = find_intersection_by_bisection(rad_body, radtol, theta_sf, phi_sf, xi, xj, radj, jshtype,jrot);
//        rad_sample = find_intersection_by_bisection_gradient(rad_body, radtol, theta_sf, phi_sf, xi, xj, radi, radj,
//                                                             jshtype,jrot);
        rad_sample = find_intersection_by_newton(ix_sf, xi, xj, theta_proj, phi_proj, rad_body, radtol, jshtype,
         jrot);
        //std::cout<<"Done"<<std::endl;

        dv = weights[kk] * (std::pow(rad_body, 3) - std::pow(rad_sample, 3));
        vol_overlap += dv;

        MathExtra::scale3(weights[kk]/std::sin(theta_bf), inorm_bf); // w_i * n * Q
        MathExtra::matvec(irot, inorm_bf, inorm_sf);            // w_i * n * Q in space frame
        MathExtra::add3(iforce, inorm_sf, iforce);              // sum(w_i * n * Q)
        MathExtra::sub3(ix_sf, xi, x_testpoint);                // Vector u from centre of "a" to surface point
        MathExtra::cross3(x_testpoint, inorm_sf, dtor);         // u x n_s * Q * w_i
        MathExtra::add3(torsum, dtor, torsum);                  // sum(u x n_s * Q * w_i)

        ///////////
//        double zero_norm[3];
//        MathExtra::zero3(zero_norm);
//        if (file_count % 1 == 0) {
//          if ((first_call) & (ii == 0) & (jj == 0)) {
//            first_call = false;
//            write_surfpoints_to_file(ix_sf, false, 1, 1, inorm_sf);
//            write_surfpoints_to_file(jx_sf, true, 0, 0, zero_norm);
//          } else if (ii == 0 & jj == 0) {
//            write_surfpoints_to_file(ix_sf, true, 1, 1, inorm_sf);
//            write_surfpoints_to_file(jx_sf, true, 0, 0, zero_norm);
//          }
//        }
        ///////////

      } // check_contact
    } // ll (quadrature)
  } // kk (quadrature)
  vol_overlap*=fac/3.0;
  MathExtra::scale3(fac, iforce);
  MathExtra::scale3(fac, torsum);
}

void PairSH::calc_tang_force_torque(double mu, int ishtype, int jshtype, double const (&normforce)[3], double const (&vr)[3],
                                    double const (&omegaa)[3], double const (&omegab)[3], double const (&cp)[3],
                                    double const (&rot_sf_bf_a)[3][3], double const (&rot_sf_bf_b)[3][3],
                                    double (&tforce)[3]){

  double fn, phi_bf, theta_bf, rad_body, vrdn;
  double cp_bf[3], n[3], vt[3], rw[3], rwn[3], vtr[3];

  fn = MathExtra::len3(normforce);
  n[0] = normforce[0]/fn;
  n[1] = normforce[1]/fn;
  n[2] = normforce[2]/fn;
  vrdn = MathExtra::dot3(vr, n);
  vt[0] = vr[0] - vrdn*n[0];
  vt[1] = vr[1] - vrdn*n[1];
  vt[2] = vr[2] - vrdn*n[2];

  MathExtra::matvec(rot_sf_bf_b, cp, cp_bf); // Contact point conversion from space frame to body frame for particle "a".
  phi_bf = std::atan2(cp_bf[1], cp_bf[0]); // Body frame contact point to phi and theta
  phi_bf = phi_bf > 0.0 ? phi_bf : MY_2PI + phi_bf;
  theta_bf = std::acos(cp_bf[2]);
  rad_body = avec->get_shape_radius(ishtype, theta_bf, phi_bf); // particle "a"'s radius
  rw[0] = omegaa[0]*rad_body;
  rw[1] = omegaa[1]*rad_body;
  rw[2] = omegaa[2]*rad_body;

  MathExtra::matvec(rot_sf_bf_b, cp, cp_bf); // Contact point conversion from space frame to body frame for particle "b".
  phi_bf = std::atan2(cp_bf[1], cp_bf[0]); // Body frame contact point to phi and theta
  phi_bf = phi_bf > 0.0 ? phi_bf : MY_2PI + phi_bf;
  theta_bf = std::acos(cp_bf[2]);
  rad_body = avec->get_shape_radius(jshtype, theta_bf, phi_bf); // particle "b"'s radius
  rw[0] += omegab[0]*rad_body;
  rw[1] += omegab[1]*rad_body;
  rw[2] += omegab[2]*rad_body;

  MathExtra::cross3(rw, n, rwn);
  MathExtra::sub3(vt, rwn, vtr);
  MathExtra::norm3(vtr);

  tforce[0] = mu*fn*vtr[0];
  tforce[1] = mu*fn*vtr[1];
  tforce[2] = mu*fn*vtr[2];
}

int PairSH::write_surfpoints_to_file(double *x, bool append_file, int cont, int ifnorm, double *norm) const{

  std::ofstream outfile;
  if (append_file){
//    outfile.open("test_dump/surfpoint_"+std::to_string(cur_time)+".csv", std::ios_base::app);
    outfile.open("test_dump/surfpoint_"+std::to_string(file_count)+".csv", std::ios_base::app);
    if (outfile.is_open()) {
      if (ifnorm) {
        outfile << std::setprecision(16) << x[0] << "," << x[1] << "," << x[2] << "," << cont <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
      }
      else{
        outfile << std::setprecision(16) << x[0] << "," << x[1] << "," << x[2] << "," << cont <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
      }
      outfile.close();
    } else std::cout << "Unable to open file";
  }
  else {
//    cur_time += update->dt;
//    outfile.open("test_dump/surfpoint_" + std::to_string(cur_time) + ".csv");
    outfile.open("test_dump/surfpoint_" + std::to_string(file_count) + ".csv");
    if (outfile.is_open()) {
      outfile << "x,y,z,cont,nx,ny,nz" << "\n";
      if (ifnorm) {
        outfile << std::setprecision(16) << x[0] << "," << x[1] << "," << x[2] << "," << cont <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
      }
      else{
        outfile << std::setprecision(16) << x[0] << "," << x[1] << "," << x[2] << "," << cont <<
                "," << norm[0] << "," << norm[1] << "," << norm[2] << "\n";
      }
      outfile.close();
    } else std::cout << "Unable to open file";
  }
  return 0;
};

int PairSH::write_ellipsoid(double *xi, double *xj, double irotmat[3][3], double jrotmat[3][3]) const{

  double sa[3][3];
  double rotmatinv[3][3];
  double tempmat[3][3];
  double icurmat[3][3];
  double jcurmat[3][3];

  MathExtra::zeromat3(sa);
  sa[0][0] = 21.0;
  sa[1][1] = 21.0;
  sa[2][2] = 105.0;

  MathExtra::times3(irotmat, sa, tempmat);
  MathExtra::invert3(irotmat, rotmatinv);
  MathExtra::times3(tempmat, rotmatinv, icurmat);

  MathExtra::times3(jrotmat, sa, tempmat);
  MathExtra::invert3(jrotmat, rotmatinv);
  MathExtra::times3(tempmat, rotmatinv, jcurmat);

  std::ofstream outfile;
//  outfile.open("test_dump/ellipsoidpos_" + std::to_string(cur_time) + ".vtk");
  outfile.open("test_dump/ellipsoidpos_" + std::to_string(file_count) + ".vtk");
  if (outfile.is_open()) {
    outfile << "# vtk DataFile Version 3.0" << "\n";
    outfile << "vtk output" << "\n";
    outfile << "ASCII" << "\n";
    outfile << "DATASET POLYDATA" << "\n";
    outfile << "POINTS 2 float" << "\n";
    outfile << xi[0] << " " << xi[1] << " " << xi[2] << "\n";
    outfile << "\n";
    outfile << xj[0] << " " << xj[1] << " " << xj[2] << "\n";
    outfile << "\n";
    outfile << "POINT_DATA 2" << "\n";
    outfile << "TENSORS tensorF float" << "\n";
    outfile << icurmat[0][0] << " " << icurmat[0][1] << " " << icurmat[0][2] << "\n";
    outfile << icurmat[1][0] << " " << icurmat[1][1] << " " << icurmat[1][2] << "\n";
    outfile << icurmat[2][0] << " " << icurmat[2][1] << " " << icurmat[2][2] << "\n";
    outfile << "\n";
    outfile << jcurmat[0][0] << " " << jcurmat[0][1] << " " << jcurmat[0][2] << "\n";
    outfile << jcurmat[1][0] << " " << jcurmat[1][1] << " " << jcurmat[1][2] << "\n";
    outfile << jcurmat[2][0] << " " << jcurmat[2][1] << " " << jcurmat[2][2] << "\n";
    outfile << "\n";
    outfile.close();
  } else std::cout << "Unable to open file";
  return 0;
};

void PairSH::get_contact_point_bounds(double rada, double radb, double xi[3], double xj[3], double (&linenorm)[3],
                                     double (&lineorigin)[3], double &lambdamin, double &lambdamax){
  int num_interacts;
  double sol1, sol2;
  std::vector<double> vec;

  //Intersection of the contact line with particle A's bounding sphere
  num_interacts = MathSpherharm::line_sphere_intersection(rada, xi, linenorm, lineorigin, sol1, sol2);
  if (num_interacts==1) error->all(FLERR, "Error, Contact line does not intersect with bounding sphere");
  vec.push_back(sol1);
  vec.push_back(sol2);

  //Intersection of the contact line with particle B's bounding sphere
  num_interacts = MathSpherharm::line_sphere_intersection(radb, xj, linenorm, lineorigin, sol1, sol2);
  if (num_interacts==1) error->all(FLERR, "Error, Contact line does not intersect with bounding sphere");
  vec.push_back(sol1);
  vec.push_back(sol2);
  std::sort(vec.begin(),vec.end());

  // Inner bounds are the middle values in the sorted list, don't care about outer intersections
  lambdamin = vec[1];
  lambdamin = vec[2];
}

void PairSH::sphere_sphere_norm_force_torque(double ri, double rj, double delta, const double x1[3],
                                             const double x2[3], double (&iforce)[3], double (&torsum)[3],
                                             double &voloverlap){

  double dr, d, wc2, Sn;
  double xi, yi, zi;
  double xj, yj, zj;
  double dist;

  xi = x1[0];
  yi = x1[1];
  zi = x1[2];
  xj = x2[0];
  yj = x2[1];
  zj = x2[2];

  dr = ri -rj;
  d = ri+rj-delta;
  wc2 = 2.0*delta*((ri+rj)-(dr*dr/d))-((delta*delta)*(1.0+(dr/d)*(dr/d)));
  Sn = MY_PI*wc2/4.0;

  //distance between the center i and j
  dist = sqrt((xj-xi)*(xj-xi)+(yj-yi)*(yj-yi)+(zj-zi)*(zj-zi));
  //normal to the contact plane
  iforce[0] = (xj-xi) / dist;
  iforce[1] = (yj-yi) / dist;
  iforce[2] = (zj-zi) / dist;

  MathExtra::scale3(Sn, iforce);
  voloverlap = MathSpherharm::get_sphere_overlap_volume(ri, rj, d);
}

double PairSH::find_intersection_by_bisection(double rad_body, double radtol, double theta_sf, double phi_sf,
                                            const double xi[3], const double xj[3], double radj, int jshtype,
                                            double (&jrot)[3][3]){

  double upper_bound, lower_bound, finalrad, rad_sample;
  double st, phi_proj, theta_proj, dtemp;
  double jx_sf[3], x_testpoint[3], x_projtestpoint[3];

  upper_bound = rad_body;
  lower_bound = 0.0;
  rad_sample = (upper_bound + lower_bound) / 2.0;
  while (upper_bound - lower_bound > radtol) {
    st = rad_sample*std::sin(theta_sf);
    jx_sf[0] = (st * cos(phi_sf)) + xi[0]; // Global coordinates of point
    jx_sf[1] = (st * sin(phi_sf)) + xi[1];
    jx_sf[2] = (rad_sample * cos(theta_sf)) + xi[2];
    // vector distance from COG of atom j (in space frame) to test point on atom i
    MathExtra::sub3(jx_sf, xj, x_testpoint);
    // scalar distance
    dtemp = MathExtra::len3(x_testpoint);
    if (dtemp > radj) {
      lower_bound = rad_sample;  // sampled radius outside of particle j, increase the lower bound
    } else {
      // Rotating the projected point into atom j's body frame (rotation matrix transpose = inverse)
      MathExtra::transpose_matvec(jrot, x_testpoint, x_projtestpoint);
      // Get projected phi and theta angle of gauss point in atom i's body frame
      phi_proj = std::atan2(x_projtestpoint[1], x_projtestpoint[0]);
      phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
      theta_proj = std::acos(x_projtestpoint[2] / dtemp);
      if (avec->check_contact(jshtype, phi_proj, theta_proj, dtemp, finalrad)) {
        upper_bound = rad_sample; // sampled radius inside of particle j, decrease the upper bound
      } else {
        lower_bound = rad_sample;  // sampled radius outside of particle j, increase the lower bound
      }
    }
    rad_sample = (upper_bound + lower_bound) / 2.0;
  }
  return rad_sample;
}

/* ----------------------------------------------------------------------
   Finds the intersection between the line projected from the quadrature point
   on the surface of atom i to the centre of mass of atom i and the surface of
   atom j. Initially a bisection method is used to reduce the size of the line
   segment. Next, the intersection of the upper and lower bounds of the line
   segment and the line formed by their intersections with atom j's surface is
   used as an approximation. (Note that these may not actually intersect and
   a skewness check is required)
 ------------------------------------------------------------------------- */
double PairSH::find_intersection_by_bisection_gradient(double rad_body, double radtol, double theta_sf, double phi_sf,
                                              const double xi[3], const double xj[3], double radi, double radj, int jshtype,
                                              double (&jrot)[3][3]){

  double upper_bound, lower_bound, finalrad, rad_sample;
  double st, phi_proj, theta_proj, dtemp;
  double p1rad, p2rad;
  double jx_sf[3], x_testpoint[3], x_projtestpoint[3];
  int num_bisects = 3;

  upper_bound = rad_body;
  lower_bound = 0.0;
  rad_sample = (upper_bound + lower_bound) / 2.0;


  st = upper_bound*std::sin(theta_sf);
  jx_sf[0] = (st * cos(phi_sf)) + xi[0]; // Global coordinates of point
  jx_sf[1] = (st * sin(phi_sf)) + xi[1];
  jx_sf[2] = (upper_bound * cos(theta_sf)) + xi[2];
  MathExtra::sub3(jx_sf, xj, x_testpoint);
  dtemp = MathExtra::len3(x_testpoint);
  // theta and phi from COG of atom j (in space frame) to p1
  phi_proj = std::atan2(x_testpoint[1], x_testpoint[0]);
  phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
  theta_proj = std::acos(x_testpoint[2] / dtemp);
  // Get the radius of atom j associated with lower bound
  p2rad = avec->get_shape_radius(jshtype, theta_proj, phi_proj);

  //collinearity check
//  MathExtra::sub3(jx_sf, xi, x_testpoint);
//  p1rad = MathExtra::len3(x_testpoint);
//  MathExtra::sub3(xi, xj, x_testpoint);
//  st = MathExtra::len3(x_testpoint);
////  std::cout<<2.0*std::max({dtemp, p1rad, st})-dtemp-p1rad-st<<std::endl;
//  if(std::abs(2.0*std::max({dtemp, p1rad, st})-dtemp-p1rad-st)<1.0e-5*radius_tol) return rad_body-p2rad;
//  if(skew_lines_check(xi, jx_sf, xj, jx_sf)) return rad_body-p2rad;

  // Bisect for num_bisects times prior to using the gradient approach
  for (int i=0; i<num_bisects; i++) {
    st = rad_sample*std::sin(theta_sf);
    jx_sf[0] = (st * cos(phi_sf)) + xi[0]; // Global coordinates of point
    jx_sf[1] = (st * sin(phi_sf)) + xi[1];
    jx_sf[2] = (rad_sample * cos(theta_sf)) + xi[2];
    // vector distance from COG of atom j (in space frame) to test point on atom i
    MathExtra::sub3(jx_sf, xj, x_testpoint);
    // scalar distance
    dtemp = MathExtra::len3(x_testpoint);
    if (dtemp > radj & rad_sample>lower_bound) {
      lower_bound = rad_sample;  // sampled radius outside of particle j, increase the lower bound
    } else {
      // Rotating the projected point into atom j's body frame (rotation matrix transpose = inverse)
      MathExtra::transpose_matvec(jrot, x_testpoint, x_projtestpoint);
      // Get projected phi and theta angle of gauss point in atom i's body frame
      phi_proj = std::atan2(x_projtestpoint[1], x_projtestpoint[0]);
      phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
      theta_proj = std::acos(x_projtestpoint[2] / dtemp);
      if (avec->check_contact(jshtype, phi_proj, theta_proj, dtemp, finalrad)) {
        upper_bound = rad_sample; // sampled radius inside of particle j, decrease the upper bound
        p2rad = finalrad; // point inside, store the radius of atom j associated with upper bound
      } else {
        lower_bound = rad_sample;  // sampled radius outside of particle j, increase the lower bound
      }
    }
    rad_sample = (upper_bound + lower_bound) / 2.0;
  }

  double p1[3], p2[3], p3[3], p4[3];
  double d1[3], d2[3];

  // Use the line connecting subsequent samples to guess where the intersection with the surface is
  while (std::abs(upper_bound - lower_bound) > radtol) {

    // point 1 (line 1 - atom i centre to quadrature point)
    st = lower_bound*std::sin(theta_sf);
    p1[0] = (st * cos(phi_sf)) + xi[0]; // Global coordinates of point
    p1[1] = (st * sin(phi_sf)) + xi[1];
    p1[2] = (lower_bound * cos(theta_sf)) + xi[2];

    // point 2 (line 1 - atom i centre to quadrature point)
    st = upper_bound*std::sin(theta_sf);
    p2[0] = (st * cos(phi_sf)) + xi[0]; // Global coordinates of point
    p2[1] = (st * sin(phi_sf)) + xi[1];
    p2[2] = (upper_bound * cos(theta_sf)) + xi[2];

    // TODO - store phi and theta rather than recalculating them here
    // vector distance from COG of atom j (in space frame) to p1
    MathExtra::sub3(p1, xj, x_testpoint);
    dtemp = MathExtra::len3(x_testpoint);
    // theta and phi from COG of atom j (in space frame) to p1
    phi_proj = std::atan2(x_testpoint[1], x_testpoint[0]);
    phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
    theta_proj = std::acos(x_testpoint[2] / dtemp);
    // Get the radius of atom j associated with lower bound
    p1rad = avec->get_shape_radius(jshtype, theta_proj, phi_proj);
    // point 1 (line 2 - atom j centre to surface (as projected by quadrature point))
    st = p1rad*std::sin(theta_proj);
    p3[0] = (st * cos(phi_proj)) + xj[0]; // Global coordinates of point
    p3[1] = (st * sin(phi_proj)) + xj[1];
    p3[2] = (p1rad * cos(theta_proj)) + xj[2];

    // TODO - store phi and theta rather than recalculating them here
    // vector distance from COG of atom j (in space frame) to p1
    MathExtra::sub3(p2, xj, x_testpoint);
    dtemp = MathExtra::len3(x_testpoint);
    // theta and phi from COG of atom j (in space frame) to p1
    phi_proj = std::atan2(x_testpoint[1], x_testpoint[0]);
    phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
    theta_proj = std::acos(x_testpoint[2] / dtemp);
    // point 1 (line 2 - atom j centre to surface (as projected by quadrature point))
    st = p2rad*std::sin(theta_proj);
    p4[0] = (st * cos(phi_proj)) + xj[0]; // Global coordinates of point
    p4[1] = (st * sin(phi_proj)) + xj[1];
    p4[2] = (p2rad * cos(theta_proj)) + xj[2];

    MathExtra::sub3(p2,p1, d1); //p2-p1=d1; p2=d1+p1;
    MathExtra::norm3(d1);
    MathExtra::sub3(p4,p3, d2);
    MathExtra::norm3(d2);
    rad_sample = calc_nearest_point(p1, p3, d1, d2, xi, jx_sf);

    // vector distance from COG of atom j (in space frame) to test point on atom i
    MathExtra::sub3(jx_sf, xj, x_testpoint);
    // scalar distance
    dtemp = MathExtra::len3(x_testpoint);

//    std::cout<<std::setprecision(11)<<p1[0]<<" "<<p1[1]<<" "<<p1[2]<<" "<<std::endl;
//    std::cout<<std::setprecision(11)<<p2[0]<<" "<<p2[1]<<" "<<p2[2]<<" "<<std::endl;
//    std::cout<<std::setprecision(11)<<p3[0]<<" "<<p3[1]<<" "<<p3[2]<<" "<<std::endl;
//    std::cout<<std::setprecision(11)<<p4[0]<<" "<<p4[1]<<" "<<p4[2]<<" "<<std::endl;
//    std::cout<<std::setprecision(11)<<jx_sf[0]<<" "<<jx_sf[1]<<" "<<jx_sf[2]<<" "<<std::endl;
//    std::cout<<std::setprecision(15)<<lower_bound<<" "<< upper_bound<<" "<< rad_sample<< " " <<radj<<" "<<p2rad<<" "<<upper_bound - lower_bound <<" "<< dtemp<< std::endl;
//    std::cout<<std::setprecision(15)<<((dtemp >= radj) & (rad_sample>lower_bound))<< " " << std::abs(dtemp-radj) << " " << (std::abs(dtemp-radj)<radtol) << " " << (rad_sample>lower_bound)<<std::endl<<std::endl;

    if (rad_sample<lower_bound) error->all(FLERR, "Error, centre within radius!");

    if ((std::abs(dtemp-radj)<radtol) & (rad_sample>lower_bound)) {
      lower_bound = rad_sample;  // sampled radius outside of particle j, increase the lower bound
    } else {
//      std::cout << "TRUE"<<std::endl;
      // Rotating the projected point into atom j's body frame (rotation matrix transpose = inverse)
      MathExtra::transpose_matvec(jrot, x_testpoint, x_projtestpoint);
      // Get projected phi and theta angle of gauss point in atom i's body frame
      phi_proj = std::atan2(x_projtestpoint[1], x_projtestpoint[0]);
      phi_proj = phi_proj > 0.0 ? phi_proj : MY_2PI + phi_proj; // move atan2 range from 0 to 2pi
      theta_proj = std::acos(x_projtestpoint[2] / dtemp);
      if (avec->check_contact(jshtype, phi_proj, theta_proj, dtemp, finalrad)) {
        upper_bound = rad_sample; // sampled radius inside of particle j, decrease the upper bound
        p2rad = finalrad; // point inside, store the radius of atom j associated with upper bound
      } else if (rad_sample>lower_bound) {
        lower_bound = rad_sample;  // sampled radius outside of particle j, increase the lower bound
      }
    }
    rad_sample = (upper_bound + lower_bound) / 2.0;

//    if (std::abs(dtemp-radj)<radtol){
//      std::cout << std::abs(upper_bound - lower_bound) << " " <<radtol<<" "<<(std::abs(upper_bound - lower_bound) > radtol)<<std::endl;
//    }
  }

  std::cout << std::setprecision(12)<<"dtemp " <<dtemp << std::endl;

  //  error->all(FLERR, "Error, centre within radius!");
  return rad_sample;
}

/* ----------------------------------------------------------------------
   Takes two lines and checks whether they are skew. Each line is defined
   by two points (a,b) and (c,d). The volume associated with these points,
   if they were the vertices of a tetrahedron, is calculated. If this volume
   is below a threshold then the points are determined to be coplanar and
   the lines are not skew.
   Algorithm taken from https://en.wikipedia.org/wiki/Skew_lines
 ------------------------------------------------------------------------- */
bool PairSH::skew_lines_check(const double a[3], const double b[3], const double c[3], const double d[3]){

  double m00, m01, m02;
  double m10, m11, m12;
  double m20, m21, m22;
  double det;
  double skew_vol_tol = 1e-20;

  m00 = a[0] - b[0];
  m01 = a[1] - b[1];
  m02 = a[2] - b[2];

  m10 = b[0] - c[0];
  m11 = b[1] - c[1];
  m12 = b[2] - c[2];

  m20 = c[0] - d[0];
  m21 = c[1] - d[1];
  m22 = c[2] - d[2];

  det = m00*m11*m22 - m00*m12*m21 -
        m10*m01*m22 + m10*m02*m21 +
        m20*m01*m12 - m20*m02*m11;

  det = std::abs(det)/6.0;

  std::cout<<"det" << det<<std::endl;

  if (det>skew_vol_tol) return false;
  return true;
}

/* ----------------------------------------------------------------------
   Takes the origins and unit normals of two skew lines and finds the point on
   the first line that is nearest to the seconds line. Returns the length of
   this first line from its start to this nearest point.
   Algorithm taken from https://en.wikipedia.org/wiki/Skew_lines
 ------------------------------------------------------------------------- */
double PairSH::calc_nearest_point(const double p1[3], const double p2[3], const double d1[3], const double d2[3],
                                  const double (xi)[3], double (&intersect)[3]){

  double n[3], n2[3];
  double dot;

  MathExtra::cross3(d1, d2, n);
  MathExtra::cross3(d2, n, n2);
  MathExtra::sub3(p2, p1, intersect);
  dot = MathExtra::dot3(intersect, n2);
  dot /= MathExtra::dot3(d1, n2);
  intersect[0] = p1[0] + dot * d1[0];
  intersect[1] = p1[1] + dot * d1[1];
  intersect[2] = p1[2] + dot * d1[2];
  MathExtra::sub3(intersect, xi, n); // reusing n
  return(MathExtra::len3(n));
}


/* ----------------------------------------------------------------------
    Find the intersection between two surfaces where the ray cast from the
    centre of particle "A" to it's surface with some fixed angle meets the
    ray cast from the centre of particle "B" to it's surface
    Problem is phrased as:
     |xb - xa|                        |cos(phi)sin(theta)|    | a |
     |yb - ya| + r(N,theta,phi) * R * |sin(phi)sin(theta)| - t| b | = 0
     |zb - za|                        |   cos(theta)     |    | c |
    where, (a,b,c) is the normal vector from the centre of particle "A" to
    its surface and "t" is the length along this normal vector to the intersection,
    (xa,ya,za) is the centre of Particle "A", R is the rotation matrix rotating the
    particle "B" from its local frame to space frame, and "r" is the radius of particle
    "B" at the given theta and phi value.
    Solved using Newton's Method and Guassian Elimination.
 ------------------------------------------------------------------------- */
double PairSH::find_intersection_by_newton(const double ix_sf[3], const double xi[3], const double xj[3],
                                           double theta_n, double phi_n, double t_n, const double radtol,
                                           const int sht, const double jrot[3][3]){

  double J[3][3], Ji[3][3], vec[3], temp[3], centre_diff[3];
  double a, b, c;
  double rp, rt, r, tmax;
  double ct, st, cp, sp, cpct, spst, spct, cpst;
  double Am[3][4];
  double alpha=0.5; // not described in paper, used to prevent "back and forth" infinite action, slows convergence

  // Vector between particle centres
  centre_diff[0] = xi[0] - xj[0];
  centre_diff[1] = xi[1] - xj[1];
  centre_diff[2] = xi[2] - xj[2];

  // Vector from centre of Particle A to the given surface point
  temp[0] = ix_sf[0] - xi[0];
  temp[1] = ix_sf[1] - xi[1];
  temp[2] = ix_sf[2] - xi[2];
//  std::cout << "a " << std::setprecision(14) << std::endl << temp[0]  << " " << temp[1]  << " " << temp[2]  << std::endl;

  // Unit vector describing the direction from the centre of Particle A to the given surface point
  MathExtra::norm3(temp);
  a = temp[0];
  b = temp[1];
  c = temp[2];

  // Testing with the analytical line/ellipsoid intersection
//  double t_analytical = -1.0;
//  double ea = 105.0/2.0;
//  double eb = 21.0/2.0;
//  double ec = 21.0/2.0;
//  double elipse_a[3] = {std::sqrt(ea), 0.0, 0.0};
//  double elipse_b[3] = {0.0, std::sqrt(eb), 0.0};
//  double elipse_c[3] = {0.0, 0.0, std::sqrt(ec)};
//  std::cout<< "b " << std::setprecision(14) << std::endl << a << " " << b << " " << c << std::endl;
//
//  std::cout << xj[0] << " " <<xj[1] << "  " << xj[2] << std::endl;
//  std::cout << elipse_a[0] << " " <<elipse_a[1] << "  " << elipse_a[2] << std::endl;
//  std::cout << elipse_b[0] << " " <<elipse_b[1] << "  " << elipse_b[2] << std::endl;
//  std::cout << elipse_c[0] << " " <<elipse_c[1] << "  " << elipse_c[2] << std::endl;
//  std::cout << xi[0] << " " <<xi[1] << "  " << xi[2] << std::endl;
//  std::cout << temp[0] << " " <<temp[1] << "  " << temp[2] << std::endl;
//  if(MathSpherharm::line_ellipsoid_intersection(xj, elipse_a, elipse_b, elipse_c, xi, temp, t_analytical)){
//    error->all(FLERR, "Error, couldn't find ellipsoid-ray intersection");
//  }
//  error->all(FLERR, "Error, couldn't find ellipsoid-ray intersection");

//  double t_analytical = -1.0;
//  double ec = 2.0;
//  double eb = 1.0;
//  double ea = 10.0;
//  double elipsoid_centre[3] = {0.0, 0.0, 0.0};
//  double line_centre[3] = {0.0, 6.0, 6.0};
//  double line_normal[3] = {0.0, -1.0, -1.0};
//  double elipse_a[3] = {std::sqrt(ea), 0.0, 0.0};
//  double elipse_b[3] = {0.0, std::sqrt(eb), 0.0};
//  double elipse_c[3] = {0.0, 0.0, std::sqrt(ec)};
//  MathSpherharm::line_ellipsoid_intersection(elipsoid_centre,elipse_a, elipse_b,elipse_c,
//                                             line_centre, line_normal, t_analytical);
//  error->all(FLERR, "Error, couldn't find ellipsoid-ray intersection");

  // Testing for a sphere
//    double t_analytical;
//    double sol1, sol2;
//    MathSpherharm::line_sphere_intersection(t_n, xj, temp, xi, sol1, sol2);
//    t_analytical = sol1 < sol2 ? sol1 : sol2;

//  double t_analytical=0.0;
//  std::cout<< "x " << theta_n << " " << phi_n << " " << t_n << std::endl <<std::endl;

  tmax = t_n;
  r = 2*radtol;

  // Not radius, just recycling variable for the while condition
  while (r>radtol){

    // Calculating repetitive terms
    ct = cos(theta_n);
    cp = cos(phi_n);
    st = sin(theta_n);
    sp = sin(phi_n);
    cpct = cp*ct;
    spst = sp*st;
    spct = sp*ct;
    cpst = cp*st;

    // Get the radius and gradients for the current iteration of theta and theta.
    r = avec->get_shape_radius_and_gradients(sht, theta_n, phi_n, rp, rt);

    // TODO - Compare Gaussian Elimination to direct matrix inversion method
    // Calculating the Jacobian of the three functions (i.e the three-dimensional coordiantes) and storing this in
    // the augmented matrix
    Am[0][0] = rt*(jrot[0][0]*cpst + jrot[0][1]*spst - jrot[0][2]*ct) + r*(jrot[0][0]*cpct + jrot[0][1]*spct - jrot[0][2]*st);
    Am[1][0] = rt*(jrot[1][0]*cpst + jrot[1][1]*spst - jrot[1][2]*ct) + r*(jrot[1][0]*cpct + jrot[1][1]*spct - jrot[1][2]*st);
    Am[2][0] = rt*(jrot[2][0]*cpst + jrot[2][1]*spst - jrot[2][2]*ct) + r*(jrot[2][0]*cpct + jrot[2][1]*spct - jrot[2][2]*st);
    Am[0][1] = rp*(jrot[0][0]*cpst + jrot[0][1]*spst - jrot[0][2]*ct) + r*(-jrot[0][0]*spst + jrot[0][1]*cpst);
    Am[1][1] = rp*(jrot[1][0]*cpst + jrot[1][1]*spst - jrot[1][2]*ct) + r*(-jrot[1][0]*spst + jrot[1][1]*cpst);
    Am[2][1] = rp*(jrot[2][0]*cpst + jrot[2][1]*spst - jrot[2][2]*ct) + r*(-jrot[2][0]*spst + jrot[2][1]*cpst);
    Am[0][2] = -a;
    Am[1][2] = -b;
    Am[2][2] = -c;

    // Calculating the value of the three functions
    temp[0] = r * cpst;
    temp[1] = r * spst;
    temp[2] = r * ct;
    MathExtra::matvec(jrot, temp, vec); // from Particle B's body frame back to space frame
    vec[0] -= (t_n * a + centre_diff[0]);
    vec[1] -= (t_n * b + centre_diff[1]);
    vec[2] -= (t_n * c + centre_diff[2]);
    temp[0] = vec[0];
    temp[1] = vec[1];
    temp[2] = vec[2];

    // Finishing the augmented matrix to be solved using Gaussian elimination
    Am[0][3] = -vec[0];
    Am[1][3] = -vec[1];
    Am[2][3] = -vec[2];

    // Solving for (x_{n+1} - x_{n}) using Gaussian Elimination.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    for (int i=0; i<3; i++) {
      // Search for maximum in this column
      double maxEl = std::abs(Am[i][i]);
      int maxRow = i;
      for (int k=i+1; k<3; k++) {
        if (std::abs(Am[k][i]) > maxEl) {
          maxEl = std::abs(Am[k][i]);
          maxRow = k;
        }
      }

      // Swap maximum row with current row (column by column)
      for (int k=i; k<3+1;k++) {
        double tmp = Am[maxRow][k];
        Am[maxRow][k] = Am[i][k];
        Am[i][k] = tmp;
      }

      // Make all rows below this one 0 in current column
      for (int k=i+1; k<3; k++) {
        double z = -Am[k][i]/Am[i][i];
        for (int j=i; j<3+1; j++) {
          if (i==j) {
            Am[k][j] = 0;
          } else {
            Am[k][j] += z * Am[i][j];
          }
        }
      }
    }

    // Solve equation Ax=b for an upper triangular matrix A
    for (int i=3-1; i>=0; i--) {
      vec[i] = Am[i][3]/Am[i][i];
      for (int k=i-1;k>=0; k--) {
        Am[k][3] -= Am[k][i] * vec[i];
      }
    }

    // Updating the function variables.
    theta_n += alpha*vec[0];
    phi_n += alpha*vec[1];
    t_n += alpha*vec[2];

    // Not radius, just recycling variable for the while condition. Using the magnitude of the error vector to
    // determine when the Newton's Method has converged. We want each term in this vector to be as close to zero as
    // possible
    r = MathExtra::len3(vec);

    if (update->ntimestep>=67) {
      std::cout << "dx " << vec[0] << " " << vec[1] << " " << vec[2] << std::endl;
      std::cout << "x " << theta_n << " " << phi_n << " " << t_n << " " << std::endl;
      std::cout<< "f " << temp[0] << " " << temp[1] << " " << temp[2] << std::endl<<std::endl;
    }

  }

//  std::cout<< "f " <<std::endl << "theta    phi   t"<<std::endl;
//  std::cout<< "g " << theta_n << " " << phi_n << " " << t_n << std::endl;
//  exit(0);

  return t_n;
}
