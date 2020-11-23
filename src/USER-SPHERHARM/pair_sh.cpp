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
#include <cstring>
#include <iostream>
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

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairSH::PairSH(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  writedata = 1;
  centroidstressflag = 1;

  //  JY Added

  // Single steps are for force and energy of a single pairwise interaction between 2 atoms
  // Energy calculation not enabled, as we don't yet have pairwise potential
  single_enable = 0;
  restartinfo = 0; // Not figured out how to do this yet
  writedata = 0; // Ditto
  respa_enable = 0;

  // Flag indicating if lammps types have been matches with SH type.
  matchtypes = 0;
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
  }
}

/* ---------------------------------------------------------------------- */
void PairSH::compute(int eflag, int vflag)
{
  int i,j,ii,jj,ll,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double radi,radj,radsum,r,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int ishtype, jshtype;
  double irot[3][3], jrot[3][3];
  double ixquadbf[3],jxquadbf[3];
  double ixquadsf[3],jxquadsf[3];
  double xgauss[3], xgaussproj[3];
  double dtemp, phi_proj, theta_proj, finalrad;
  double phi, theta;
  double overlap[3];

  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *shtype = atom->shtype;
  double **quat = atom->quat;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  double iquat_delta[4];
  double jquat_delta[4];
  double quat_temp[4];

  int num_quad2;
  double **quad_rads = avec->get_quadrature_rads(num_quad2);
  double **angles = avec->get_quadrature_angs();
  double **quatinit = avec->get_quat_init();
  double *max_rad = avec->get_max_rads();

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    ishtype = shtype[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    radi = max_rad[ishtype];

    // Unrolling the initial quat (Unit quaternion - inverse = conjugate)
    MathExtra::qconjugate(quatinit[ishtype],quat_temp);
    // Calculating the quat to rotate the particles to new position (q_delta = q_target * q_current^-1)
    MathExtra::quatquat(quat[i],quat_temp,iquat_delta);
    MathExtra::qnormalize(iquat_delta);
    // Calculate the rotation matrix for the quaternion for atom i
    MathExtra::quat_to_mat(iquat_delta,irot);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      jshtype = shtype[j];
      radj = max_rad[jshtype];
      radsum = radi + radj;
      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      jtype = type[j];

      if (r < radsum) {
        std::cout << "ij: " <<i<<j<< " r: "<< r  << "  radsum: " << radsum << " " << nlocal << std::endl;
        // TODO require a check here to make sure that the COG of particle is not inside of other particle. Comparing
        // max radii does not work as not a correct test for asymmetrical particles. Check should also look at minimum
        // radius. There will then be a zone in which the COG of a particle *could* be inside another, but this can't be
        // proven until the SH expansion has been tested.
        //if (r < MAX(radi, radj)){
        //  error->all(FLERR,"Error, centre within radius!");
        //}

        // Unrolling the initial quat (Unit quaternion - inverse = conjugate)
        MathExtra::qconjugate(quatinit[jshtype],quat_temp);
        // Calculating the quat to rotate the particles to new position (q_delta = q_target * q_current^-1)(space frame)
        MathExtra::quatquat(quat[j],quat_temp,jquat_delta);
        MathExtra::qnormalize(jquat_delta);
        // Calculate the rotation matrix for the quaternion for atom j
        MathExtra::quat_to_mat(jquat_delta,jrot);

        // Compare all points of guassian quadrature on atom j against the projected point on atom i
        for (ll = 0; ll < num_quad2; ll++) {
          // Point of gaussian quadrature in body frame
          jxquadbf[0] = quad_rads[jshtype][ll] * sin(angles[0][ll]) * cos(angles[1][ll]);
          jxquadbf[1] = quad_rads[jshtype][ll] * sin(angles[0][ll]) * sin(angles[1][ll]);
          jxquadbf[2] = quad_rads[jshtype][ll] * cos(angles[0][ll]);
          // Point of gaussian quadrature in space frame
          MathExtra::matvec(jrot, jxquadbf, jxquadsf);
          // Translating by current location of particle j's centre
          jxquadsf[0] += x[j][0];
          jxquadsf[1] += x[j][1];
          jxquadsf[2] += x[j][2];
          // vector distance from gauss point on atom j to COG of atom i (in space frame)
          MathExtra::sub3(jxquadsf, x[i], xgauss);
          // scalar distance
          dtemp = MathExtra::len3(xgauss);
          // Rotating the projected point into atom i's body frame (rotation matrix transpose = inverse)
          MathExtra::transpose_matvec(irot, xgauss, xgaussproj);
          // Get projected phi and theta angle of gauss point in atom i's body frame
          phi_proj = std::atan2(xgaussproj[1], xgaussproj[0]);
          theta_proj = std::acos(xgaussproj[2] / dtemp);
          // Check for contact
          if (avec->check_contact(ishtype, phi_proj, theta_proj, dtemp, finalrad)) {
            // Get the phi and thea angles as project from the gauss point in the space frame for both particle i and j
            phi = std::atan2(xgauss[1], xgauss[0]);
            theta = std::acos(xgauss[2] / dtemp);
            // Get the space frame vector of the radius
            ixquadsf[0] = finalrad * sin(theta) * cos(phi);
            ixquadsf[1] = finalrad * sin(theta) * sin(phi);
            ixquadsf[2] = finalrad * cos(theta);
            // Translating by current location of particle i's centre
            ixquadsf[0] += xtmp;
            ixquadsf[1] += ytmp;
            ixquadsf[2] += ztmp;
            // Getting the overlap vector in the space frame, acting towards the COG of particle i
            MathExtra::sub3(ixquadsf, jxquadsf, overlap);
            // Using a F=-kx force for testing.
            fpair = normal_coeffs[itype][jtype][0];
            f[i][0] += overlap[0] * fpair;
            f[i][1] += overlap[1] * fpair;
            f[i][2] += overlap[2] * fpair;
          }
        }

        if (newton_pair || j < nlocal) {
          // Compare all points of guassian quadrature on atom i against the projected point on atom j
          for (ll=0; ll<num_quad2; ll++) {
            // Point of gaussian quadrature in body frame
            ixquadbf[0] = quad_rads[ishtype][ll] * sin(angles[0][ll]) * cos(angles[1][ll]);
            ixquadbf[1] = quad_rads[ishtype][ll] * sin(angles[0][ll]) * sin(angles[1][ll]);
            ixquadbf[2] = quad_rads[ishtype][ll] * cos(angles[0][ll]);
            // Point of gaussian quadrature in space frame
            MathExtra::matvec(irot, ixquadbf, ixquadsf);
            // Translating by current location of particle i's centre
            ixquadsf[0] += xtmp;
            ixquadsf[1] += ytmp;
            ixquadsf[2] += ztmp;
            // vector distance from COG of atom j (in space frame) to gauss point on atom i
            MathExtra::sub3(ixquadsf, x[j], xgauss);
            // scalar distance
            dtemp = MathExtra::len3(xgauss);
            // Rotating the projected point into atom j's body frame (rotation matrix transpose = inverse)
            MathExtra::transpose_matvec(jrot, xgauss, xgaussproj);
            // Get projected phi and theta angle of gauss point in atom j's body frame
            phi_proj = std::atan2(xgaussproj[1], xgaussproj[0]);
            theta_proj = std::acos(xgaussproj[2] / dtemp);

            // Check for contact
            if (avec->check_contact(jshtype, phi_proj, theta_proj, dtemp, finalrad)) {
              // Get the phi and thea angles as project from the gauss point in the space frame for both particle i and j
              phi = std::atan2(xgauss[1], xgauss[0]);
              theta = std::acos(xgauss[2] / dtemp);
              // Get the space frame vector of the radius
              jxquadsf[0] = finalrad * sin(theta) * cos(phi);
              jxquadsf[1] = finalrad * sin(theta) * sin(phi);
              jxquadsf[2] = finalrad * cos(theta);
              // Translating by current location of particle j's centre
              jxquadsf[0] += x[j][0];
              jxquadsf[1] += x[j][1];
              jxquadsf[2] += x[j][2];
              // Getting the overlap vector in the space frame, acting towards the COG of particle j
              MathExtra::sub3(jxquadsf, ixquadsf, overlap);
              // Using a F=-kx force for testing.
              fpair = normal_coeffs[itype][jtype][0];
              f[j][0] += overlap[0] * fpair;
              f[j][1] += overlap[1] * fpair;
              f[j][2] += overlap[2] * fpair;
           }
          }
        }
      }
    }
  }

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
  if (narg != 3)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  double normal_coeffs_one;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
  normal_coeffs_one = utils::numeric(FLERR,arg[2],false,lmp);// kn

  // Linking the Types to the SH Types, needed for finding the cut per Type
  if (!matchtypes) matchtype();

  int count = 0;
  int shi, shj;
  double *max_rad = avec->get_max_rads();

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

//void PairSH::init_style()
//{
//  neighbor->request(this,instance_me);
//}

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
  double *max_rad = avec->get_max_rads();

  // No epsilon and no sigma used for the spherical harmonic atom style
  if (setflag[i][j] == 0) {
    shi = typetosh[i];
    shj = typetosh[j];
    cut[i][j] = max_rad[shi]+max_rad[shj];
  }

  // TO FIX - Just use the first coefficient for the pair, no mixing
  normal_coeffs[i][j][0] = normal_coeffs[j][i][0] = normal_coeffs[i][i][0];

  return cut[i][j];
}