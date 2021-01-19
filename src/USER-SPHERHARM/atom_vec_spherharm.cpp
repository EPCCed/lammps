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

#include "atom_vec_spherharm.h"
#include <iostream>
#include <potential_file_reader.h>
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "fix_adapt.h"
#include "math_const.h"
#include "error.h"
#include "memory.h"
#include "utils.h"
#include "math_extra.h"
#include "math_eigen.h"
#include "math_spherharm.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpherharm;
/* ---------------------------------------------------------------------- */

AtomVecSpherharm::AtomVecSpherharm(LAMMPS *lmp) : AtomVec(lmp)
{
  shcoeffs_byshape=pinertia_byshape=quatinit_byshape=nullptr;
  expfacts_byshape=quad_rads_byshape=angles=nullptr;
  quat=angmom=omega=nullptr;
  maxrad_byshape=weights=nullptr;
  shtype=nullptr;
  num_quadrature=nshtypes=0;
  maxshexpan = 20;

  verbose_out = true;

  mass_type = 1;  // per-type mass arrays
  molecular = 0;  // 0 = atomic

  atom->spherharm_flag = 1;
  atom->radius_flag = atom->rmass_flag = 0;  // Particles don't store radius, per-type masses
  atom->omega_flag = atom->torque_flag = atom -> angmom_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "omega torque shtype angmom quat ";
  fields_copy = (char *) "omega shtype angmom quat";
  fields_comm = (char *) "quat";
  fields_comm_vel = (char *) "omega angmom quat";
  fields_reverse = (char *) "torque";
  fields_border = (char *) "shtype";
  fields_border_vel = (char *) "omega angmom shtype";
  fields_exchange = (char *) "omega shtype angmom";
  fields_restart = (char *) "omega shtype angmom";
  fields_create = (char *) "omega shtype angmom quat";
  fields_data_atom = (char *) "id type x shtype quat";
  fields_data_vel = (char *) "id v omega angmom";
}

AtomVecSpherharm::~AtomVecSpherharm()
{
  memory->sfree(angles);
  memory->sfree(weights);
  memory->sfree(shcoeffs_byshape);
  memory->sfree(expfacts_byshape);
  memory->sfree(quad_rads_byshape);

}

/* ----------------------------------------------------------------------
   process sub-style args
------------------------------------------------------------------------- */

void AtomVecSpherharm::process_args(int narg, char **arg) {

  int num_quad2, numcoeffs, me;
  MPI_Comm_rank(world,&me);

  if (narg < 1) error->all(FLERR, "llegal atom_style atom_style spherharm command");

  num_quadrature = utils::inumeric(FLERR, arg[0], true, lmp);
  nshtypes = narg - 1;
  atom -> nshtypes = nshtypes;

  num_quad2 = num_quadrature*num_quadrature;
//  numcoeffs = 2*((maxshexpan*maxshexpan)-1);
  numcoeffs = (maxshexpan+1)*(maxshexpan+2);

  // Local memory allocation, must be deleted in class destructor
  memory->create(angles, 2, num_quad2, "AtomVecSpherharm:angles");
  memory->create(weights, num_quadrature, "AtomVecSpherharm:weights");
  memory->create(quad_rads_byshape, nshtypes, num_quad2, "AtomVecSpherharm:quad_rads_byshape");
  memory->create(shcoeffs_byshape, nshtypes, numcoeffs, "AtomVecSpherharm:shcoeff");
  memory->create(expfacts_byshape, nshtypes, maxshexpan + 1, "AtomVecSpherharm:expfacts_byshape");

  // Atom memory allocation, must be deleted in atom class destructor
  memory->create(atom->pinertia_byshape, nshtypes, 3, "AtomVecSpherharm:pinertia");
  memory->create(atom->quatinit_byshape, nshtypes, 4, "AtomVecSpherharm:orient");
  memory->create(atom->maxrad_byshape, nshtypes, "AtomVecSpherharm:maxrad_byshape");
//  memory->create(atom->extentpoints_byshape, nshtypes, num_quad2, 3, "AtomVecSpherharm:extentpoints_byshape");

  // Directing the local pointers to the memory just allocated in the atom class
  pinertia_byshape = atom->pinertia_byshape;
  quatinit_byshape = atom->quatinit_byshape;
  maxrad_byshape = atom->maxrad_byshape;
//  extentpoints_byshape = atom->extentpoints_byshape;


  for (int type=0; type<nshtypes; type++) {
    maxrad_byshape[type] = 0.0;
    for (int i=0; i<numcoeffs; i++) {
      shcoeffs_byshape[type][i] = 0.0;
    }
  }

  if (me==0){
    for (int i = 1; i < narg; i++) {
      std::cout<< arg[i] << std::endl;
      read_sh_coeffs(arg[i], i - 1);
    }

    for (int type=0; type<nshtypes; type++) {
      for (int i=0; i<numcoeffs; i++) {
        std::cout << "coeff " << i << " "<< shcoeffs_byshape[type][i] << std::endl;
      }
    }


    get_quadrature_values();
    getI();
    //    calcexpansionfactors();
    calcexpansionfactors_gauss();
//    gen_extent_box_points(num_quadrature);
  }

  MPI_Bcast(&(angles[0][0]), 2 * num_quad2, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(weights[0]), num_quadrature, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(quad_rads_byshape[0][0]), nshtypes * num_quad2, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(pinertia_byshape[0][0]), nshtypes * 3, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(quatinit_byshape[0][0]), nshtypes * 4, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(shcoeffs_byshape[0][0]), nshtypes * numcoeffs, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(expfacts_byshape[0][0]), nshtypes * maxshexpan + 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(maxrad_byshape[0]), nshtypes, MPI_DOUBLE, 0, world);
//  MPI_Bcast(&(extentpoints_byshape[0]), nshtypes*num_quad2*3, MPI_DOUBLE, 0, world);

  // delay setting up of fields until now
  setup_fields();

}

void AtomVecSpherharm::init()
{
  AtomVec::init();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()

   Not growing per-shape values as new atoms do not add new shapes. Shapes
   are defined once, in the prcoess args method.
------------------------------------------------------------------------- */

void AtomVecSpherharm::grow_pointers() {
  omega = atom->omega;
  shtype = atom->shtype;
  angmom = atom->angmom;
  quat = atom->quat;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecSpherharm::create_atom_post(int ilocal)
{
  shtype[ilocal] = -1;
  quat[ilocal][0] = 1.0;
  quat[ilocal][1] = 0.0;
  quat[ilocal][2] = 0.0;
  quat[ilocal][3] = 0.0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecSpherharm::data_atom_post(int ilocal)
{
  // reading and writing for this atom style has not been considered yet
  omega[ilocal][0] = 0.0;
  omega[ilocal][1] = 0.0;
  omega[ilocal][2] = 0.0;
}

/* ----------------------------------------------------------------------
   modify values for AtomVec::pack_data() to pack
------------------------------------------------------------------------- */

void AtomVecSpherharm::pack_data_pre(int ilocal)
{
  // reading and writing for this atom style has not been considered yet
}

/* ----------------------------------------------------------------------
   unmodify values packed by AtomVec::pack_data()
------------------------------------------------------------------------- */

void AtomVecSpherharm::pack_data_post(int ilocal)
{
  // reading and writing for this atom style has not been considered yet

}

/* ----------------------------------------------------------------------
 Calculate the inertia of all SH particle types
------------------------------------------------------------------------- */
void AtomVecSpherharm::getI() {

  using std::cout;
  using std::endl;
  using std::sin;
  using std::cos;
  using std::pow;
  using std::sqrt;
  using std::fabs;

  std::vector<double> itensor;
  double i11, i22, i33, i12, i23, i13;
  double theta, phi, st, ct, sp, cp, r, vol, fact;
  double factor = (0.5 * MY_PI * MY_PI);
  int count;

  int ierror;
  double inertia[3];
  double tensor[3][3], evectors[3][3];
  double cross[3];
  double ex[3];
  double ey[3];
  double ez[3];

  static const double EPSILON   = 1.0e-7;

  for (int sht = 0; sht < nshtypes; sht++) {

    itensor.clear();
    count=0;
    i11 = i22 = i33 = i12 = i23 = i13 = fact = vol = 0.0;

    for (int i = 0; i < num_quadrature; i++) {
      for (int j = 0; j < num_quadrature; j++) {
        theta = angles[0][count];
        phi = angles[1][count];
        st = sin(theta);
        ct = cos(theta);
        sp = sin(phi);
        cp = cos(phi);
        r = quad_rads_byshape[sht][count];
        fact = 0.2 * weights[i] * weights[j] * pow(r, 5) * st;
        vol += (weights[i] * weights[j] * pow(r, 3) * st / 3.0);
        i11 += (fact * (1.0 - pow(cp * st, 2)));
        i22 += (fact * (1.0 - pow(sp * st, 2)));
        i33 += (fact * (1.0 - pow(ct, 2)));
        i12 -= (fact * cp * sp * st * st);
        i13 -= (fact * cp * ct * st);
        i23 -= (fact * sp * ct * st);
        count++;
      }
    }

    double vol2=0.0;
    double iang = MY_PI;
    int trap_L = 2*(num_quadrature-1);
    double abscissa[num_quadrature];
    QuadPair p;
    // Get the quadrature weights, and abscissa. Convert abscissa to theta angles
    for (int i = 0; i < num_quadrature; i++) {
      p = GLPair(num_quadrature, i + 1);
      abscissa[i] = p.x();
    }
    for (int ll = 0; ll <= trap_L; ll++) {
      double phi_pole = MY_2PI * ll / (double(trap_L) + 1.0);
      for (int kk = 0; kk < num_quadrature; kk++) {
        double theta_pole = (iang * 0.5 * abscissa[kk]) + (iang * 0.5);
        vol2 += weights[kk] * pow(get_shape_radius(sht, theta_pole, phi_pole),3) * std::sin(theta_pole);
      }
    }
    vol2 *= (MY_PI*iang/((double(trap_L)+1.0)))/3.0;

    vol *= factor;
    i11 *= factor;
    i22 *= factor;
    i33 *= factor;
    i12 *= factor;
    i13 *= factor;
    i23 *= factor;
    if (vol > 0.0) {
      i11 /= vol;
      i22 /= vol;
      i33 /= vol;
      i12 /= vol;
      i13 /= vol;
      i23 /= vol;
      itensor.push_back(i11);
      itensor.push_back(i22);
      itensor.push_back(i33);
      itensor.push_back(i12);
      itensor.push_back(i13);
      itensor.push_back(i23);
    } else {
      error->all(FLERR,"Divide by vol = 0 in getI");
    }

    tensor[0][0] = itensor[0];
    tensor[1][1] = itensor[1];
    tensor[2][2] = itensor[2];
    tensor[1][2] = tensor[2][1] = itensor[5];
    tensor[0][2] = tensor[2][0] = itensor[4];
    tensor[0][1] = tensor[1][0] = itensor[3];

    if (verbose_out) {
      std::cout << std::endl;
      std::cout << "Total Volume" << std::endl;
      std::cout << vol << std::endl;
      std::cout << std::endl;
      std::cout << "Total Volume Method 2" << std::endl;
      std::cout << vol2 << std::endl;
      std::cout << std::endl;
      std::cout << "Iniertia tensor" << std::endl;
      std::cout << itensor[0] << " " << itensor[1] << " " << itensor[2] << " " << itensor[3] << " " << itensor[4] << " "
                << itensor[5] << " " << std::endl;
    }

    ierror = MathEigen::jacobi3(tensor,inertia,evectors);
    if (ierror) error->all(FLERR,"Insufficient Jacobi rotations for rigid body");
    ex[0] = evectors[0][0];
    ex[1] = evectors[1][0];
    ex[2] = evectors[2][0];
    ey[0] = evectors[0][1];
    ey[1] = evectors[1][1];
    ey[2] = evectors[2][1];
    ez[0] = evectors[0][2];
    ez[1] = evectors[1][2];
    ez[2] = evectors[2][2];

    // if any principal moment < scaled EPSILON, set to 0.0
    double max;
    max = MAX(inertia[0],inertia[1]);
    max = MAX(max,inertia[2]);

    if (inertia[0] < EPSILON*max) inertia[0] = 0.0;
    if (inertia[1] < EPSILON*max) inertia[1] = 0.0;
    if (inertia[2] < EPSILON*max) inertia[2] = 0.0;

    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd vector if needed
    MathExtra::cross3(ex,ey,cross);
    if (MathExtra::dot3(cross,ez) < 0.0) MathExtra::negate3(ez);

    if (verbose_out) {
      std::cout << std::endl;
      std::cout << "Iniertia tensor eigenvectors" << std::endl;
      std::cout << ex[0] << " " << ex[1] << " " << ex[2] << " " << std::endl;
      std::cout << ey[0] << " " << ey[1] << " " << ey[2] << " " << std::endl;
      std::cout << ez[0] << " " << ez[1] << " " << ez[2] << " " << std::endl;

      std::cout << std::endl;
      std::cout << "Iniertia tensor eigenvalues (principle inertia)" << std::endl;
      std::cout << inertia[0] << " " << inertia[1] << " " << inertia[2] << " " << std::endl;
    }

    // create initial quaternion
    MathExtra::exyz_to_q(ex, ey, ez, quatinit_byshape[sht]);

    if (verbose_out) {
      std::cout << std::endl;
      std::cout << "Initial Quaternion (Defined by Iniertia)" << std::endl;
      std::cout << quatinit_byshape[sht][0] << " " << quatinit_byshape[sht][1] << " "
                << quatinit_byshape[sht][2] << " " << quatinit_byshape[sht][3] << " " << quatinit_byshape[sht][4]
                << std::endl;

      MathExtra::q_to_exyz(quatinit_byshape[sht], ex, ey, ez);
      std::cout << std::endl;
      std::cout << "Initial Quaternion (Defined by Iniertia) converted back to eigenvectors" << std::endl;
      std::cout << ex[0] << " " << ex[1] << " " << ex[2] << " " << std::endl;
      std::cout << ey[0] << " " << ey[1] << " " << ey[2] << " " << std::endl;
      std::cout << ez[0] << " " << ez[1] << " " << ez[2] << " " << std::endl;
    }

    pinertia_byshape[sht][0] = inertia[0];
    pinertia_byshape[sht][1] = inertia[1];
    pinertia_byshape[sht][2] = inertia[2];
  }
}

/* ----------------------------------------------------------------------
  Calculate the radi at the points of quadrature using the Spherical Harmonic
  expansion
------------------------------------------------------------------------- */
void AtomVecSpherharm::get_quadrature_values() {

  // Fixed properties
  double theta, phi;
  int num_quad2, count;
  double abscissa[num_quadrature];
  QuadPair p;

  // Get the quadrature weights, and abscissa. Convert abscissa to theta angles
  for (int i = 0; i < num_quadrature; i++) {
    p = GLPair(num_quadrature, i + 1);
    weights[i] = p.weight;
    abscissa[i] = p.x();
  }

  count=0;
  for (int i = 0; i < num_quadrature; i++) {
    for (int j = 0; j < num_quadrature; j++) {
      angles[0][count] = 0.5 * MY_PI * (abscissa[i] + 1.0);
      angles[1][count] = MY_PI * (abscissa[j] + 1.0);
      count++;
    }
  }

  num_quad2 = num_quadrature*num_quadrature;
  for (int sht = 0; sht < nshtypes; sht++) {
    for (int k = 0; k < num_quad2; k++) {
      theta = angles[0][k];
      phi = angles[1][k];
      quad_rads_byshape[sht][k] = get_shape_radius(sht, theta, phi);
    }
  }

  std::cout<< "RADIUS ADSADSAD " << get_shape_radius(0, MY_PI2, 0)<< std::endl;

}

/* ----------------------------------------------------------------------
  Calculate the expansion factors for all SH particles using a grid of points
  (clustering at poles, spreading at the equator)
------------------------------------------------------------------------- */
void AtomVecSpherharm::calcexpansionfactors()
{

//  double safety_factor = 1.01;
  double safety_factor = 1.00;
  double theta, phi, factor;
  double rmax;
  double x_val;
  double mphi;
  double P_n_m;
  int num_quad2 = num_quadrature*num_quadrature;
  int nloc, loc;
  std::vector<double> r_n, r_npo;
  r_n.resize(num_quad2, 0.0);
  r_npo.resize(num_quad2, 0.0);

  std::vector<double> ratios, expfactors;
  ratios.resize(num_quad2, 0.0);
  expfactors.resize(maxshexpan + 1, 0.0);
  expfactors[maxshexpan] = 1.0;
  rmax = 0;

  int k;
  for (int n = 0; n <= maxshexpan; n++) { // For each harmonic n
    nloc = n*(n+1);
    k = 0;
    for (int i = 0; i < num_quadrature; i++) { // For each theta value (k corresponds to angle pair)
      theta = ((double)(i) * MY_PI) / ((double)(num_quadrature));
      if (i == 0) theta = 0.001 * MY_PI;
      if (i == num_quadrature - 1) theta = 0.999 * MY_PI;
      x_val = cos(theta);
      for (int j = 0; j < num_quadrature; j++) { // For each phi value (k corresponds to angle pair)
        phi = (2.0 * MY_PI * (double)(j)) / ((double)((num_quadrature)));
        loc = nloc;
        P_n_m = plegendre(n, 0, x_val);
        r_n[k] += shcoeffs_byshape[0][(n + 1) * (n + 2) - 2] * P_n_m;
        for (int m = n; m > 0; m--) { // For each m in current harmonic n
          mphi = (double) m * phi;
          P_n_m = plegendre(n, m, x_val);
          r_n[k] += (shcoeffs_byshape[0][loc] * cos(mphi) - shcoeffs_byshape[0][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
          loc+=2;
        }

        if (i==0 && j==0){
          std::cout << n << " " << r_n[k] << " " << shcoeffs_byshape[0][(n + 1) * (n + 2) - 2] << std::endl;
        }

        if (r_n[k] > rmax) { //
          rmax = r_n[k];
        }
        if (n <= maxshexpan - 1) {
          r_npo[k] = r_n[k];
          n++;
          loc = n*(n+1);
          P_n_m = plegendre(n, 0, x_val);
          r_npo[k] += shcoeffs_byshape[0][(n + 1) * (n + 2) - 2] * P_n_m;
          for (int m = n; m > 0; m--) {
            mphi = (double) m * phi;
            P_n_m = plegendre(n, m, x_val);
            r_npo[k] += (shcoeffs_byshape[0][loc] * cos(mphi) - shcoeffs_byshape[0][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
            loc+=2;
          }
          n--;
          ratios[k] = r_npo[k] / r_n[k];
        }
        k++;
      }
    }
    if (n <= maxshexpan - 1) {
      double max_val = 0;
      for (int ii = 0; ii<k; ii++){
        if (ratios[ii]>max_val){
          max_val = ratios[ii];
        }
      }
      expfactors[n] = max_val;
      if (expfactors[n] < 1.0) {
        expfactors[n] = 1.0;
      }
    }
  }

  factor = expfactors[maxshexpan];
  for (int n = maxshexpan - 1; n >= 0; n--) {
    factor *= expfactors[n] * safety_factor;
    expfactors[n] = factor;
    expfacts_byshape[0][n] = factor;  // NEED TO FIX THE INDEX HERE
  }
  expfacts_byshape[0][maxshexpan] = 1.0; // NEED TO FIX THE INDEX HERE
  rmax *= safety_factor;

  if (verbose_out) {
    std::cout << "R_max for all harmonics " << rmax << std::endl;
    std::cout << "0th harmonic expansion factor " << expfacts_byshape[0][0] << std::endl;
    std::cout << "0th harmonic sphere radius " << shcoeffs_byshape[0][0] * std::sqrt(1.0 / (4.0 * MY_PI)) << std::endl;
    std::cout << "expanded 0th harmonic sphere radius "
              << expfacts_byshape[0][0] * double(shcoeffs_byshape[0][0]) * std::sqrt(1.0 / (4.0 * MY_PI)) << std::endl;
  }

  for (int n = 0; n <= maxshexpan; n++) {
    std::cout << expfacts_byshape[0][n] << std::endl;
  }

}

/* ----------------------------------------------------------------------
  Calculate the expansion factors for all particles using the points of Gaussian quadrature
  (clustering at poles, spreading at the equator)
------------------------------------------------------------------------- */
void AtomVecSpherharm::calcexpansionfactors_gauss()
{

  double safety_factor = 1.00;
  double theta, phi, factor;
  double x_val, mphi;
  double P_n_m;
  int nloc, loc, k;
  int num_quad2 = num_quadrature*num_quadrature;
  std::vector<double> r_n, r_npo;
  std::vector<double> ratios, expfactors;
  r_n.resize(num_quad2, 0.0);
  r_npo.resize(num_quad2, 0.0);
  ratios.resize(num_quad2, 0.0);
  expfactors.resize(maxshexpan + 1, 0.0);
  expfactors[maxshexpan] = 1.0;


  for (int sht = 0; sht < nshtypes; sht++) {

    std::fill(r_n.begin(), r_n.end(), 0.0);

    for (int n = 0; n <= maxshexpan; n++) { // For each harmonic n
      nloc = n * (n + 1);
      k = 0;
      for (int i = 0; i < num_quadrature; i++) { // For each theta value (k corresponds to angle pair)
        for (int j = 0; j < num_quadrature; j++) { // For each phi value (k corresponds to angle pair)
          theta = angles[0][k];
          phi = angles[1][k];
          x_val = cos(theta);
          loc = nloc;
          P_n_m = plegendre(n, 0, x_val);
          r_n[k] += shcoeffs_byshape[sht][(n + 1) * (n + 2) - 2] * P_n_m;
          for (int m = n; m > 0; m--) { // For each m in current harmonic n
            mphi = (double) m * phi;
            P_n_m = plegendre(n, m, x_val);
            r_n[k] += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
            loc += 2;
          }
          if (n <= maxshexpan - 1) { // Get the ratios of radii between subsequent harmonics (except the final two)
            r_npo[k] = r_n[k];
            n++;
            loc = n * (n + 1);
            P_n_m = plegendre(n, 0, x_val);
            r_npo[k] += shcoeffs_byshape[sht][(n + 1) * (n + 2) - 2] * P_n_m;
            for (int m = n; m > 0; m--) {
              mphi = (double) m * phi;
              P_n_m = plegendre(n, m, x_val);
              r_npo[k] += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
              loc += 2;
            }
            n--;
            ratios[k] = r_npo[k] / r_n[k];
          }
          else { // Get the maximum radius at the final harmonic
            if (r_n[k] > maxrad_byshape[sht]) {
              maxrad_byshape[sht] = r_n[k];
            }
          }
          k++;
        }
      }
      if (n <= maxshexpan - 1) {
        double max_val = 0;
        for (int ii = 0; ii < k; ii++) {
          if (ratios[ii] > max_val) {
            max_val = ratios[ii];
          }
        }
        expfactors[n] = max_val;
        if (expfactors[n] < 1.0) {
          expfactors[n] = 1.0;
        }
      }
    }

    factor = expfactors[maxshexpan];
    for (int n = maxshexpan - 1; n >= 0; n--) {
      factor *= expfactors[n] * safety_factor;
      expfactors[n] = factor;
      expfacts_byshape[sht][n] = factor;
    }
    expfacts_byshape[sht][maxshexpan] = 1.0;

    if (verbose_out) {
      std::cout << "R_max for final harmonic " << maxrad_byshape[sht] << std::endl;
      std::cout << "0th harmonic expansion factor " << expfacts_byshape[sht][0] << std::endl;
      std::cout << "0th harmonic sphere radius " << shcoeffs_byshape[sht][0] * std::sqrt(1.0 / (4.0 * MY_PI))
                << std::endl;
      std::cout << "expanded 0th harmonic sphere radius "
                << expfacts_byshape[0][0] * double(shcoeffs_byshape[sht][0]) * std::sqrt(1.0 / (4.0 * MY_PI))
                << std::endl;

      for (int n = 0; n <= maxshexpan; n++) {
        std::cout << expfacts_byshape[0][n] << std::endl;
      }
    }

    maxrad_byshape[sht] *= safety_factor;
  }
}

/* ----------------------------------------------------------------------
  Given a shape, a spherical coordinate (value of theta and phi), and an input distance,
  check whether the radius for that shape and spherical coordinate is greater than
  the input distance. If yes, there is contact and return 1 (also set the
  value of "finalrad" to the radius for the shape and spherical coordinate). If not,
  return 0.

  Note that contact is checked at progressive harmonics. The radius at each harmonic
  is expanded using the pre-calculated expansion factors. If at any harmonic, the radius
  is less than the input distance, the the radius will be less than the input distance
  for all subsequent harmonics and the algorithm can be stopped and return 0.
------------------------------------------------------------------------- */
int AtomVecSpherharm::check_contact(int sht, double phi_proj, double theta_proj, double outerdist, double &finalrad) {

  double rad_val = shcoeffs_byshape[sht][0] * std::sqrt(1.0 / (4.0 * MY_PI));
  double sh_dist = expfacts_byshape[sht][0] * rad_val;

  if (outerdist > sh_dist) {
    return 0;
  }

  int n, nloc, loc;
  double P_n_m, x_val, mphi, Pnm_nn;
  std::vector<double> Pnm_m2, Pnm_m1;

  Pnm_m2.resize(maxshexpan+1, 0.0);
  Pnm_m1.resize(maxshexpan+1, 0.0);
  n = 1;
  x_val = std::cos(theta_proj);
  while (n<=maxshexpan) {
    nloc = n * (n + 1);
    if (n == 1) {
      P_n_m = plegendre(1, 0, x_val);
      Pnm_m2[0] = P_n_m;
      rad_val += shcoeffs_byshape[sht][4] * P_n_m;
      P_n_m = plegendre(1, 1, x_val);
      Pnm_m2[1] = P_n_m;
      mphi = 1.0 * phi_proj;
      rad_val += (shcoeffs_byshape[sht][2] * cos(mphi) - shcoeffs_byshape[sht][3] * sin(mphi)) * 2.0 * P_n_m;
    } else if (n == 2) {
      P_n_m = plegendre(2, 0, x_val);
      Pnm_m1[0] = P_n_m;
      rad_val += shcoeffs_byshape[sht][10] * P_n_m;
      for (int m = 2; m >= 1; m--) {
        P_n_m = plegendre(2, m, x_val);
        Pnm_m1[m] = P_n_m;
        mphi = (double) m * phi_proj;
        rad_val += (shcoeffs_byshape[sht][nloc] * cos(mphi) - shcoeffs_byshape[sht][nloc + 1] * sin(mphi)) * 2.0 * P_n_m;
        nloc += 2;
      }
      Pnm_nn = Pnm_m1[2];
    } else {
      P_n_m = plegendre_recycle(n, 0, x_val, Pnm_m1[0], Pnm_m2[0]);
      Pnm_m2[0] = Pnm_m1[0];
      Pnm_m1[0] = P_n_m;
      loc = (n + 1) * (n + 2) - 2;
      rad_val += shcoeffs_byshape[sht][loc] * P_n_m;
      loc -= 2;
      for (int m = 1; m < n - 1; m++) {
        P_n_m = plegendre_recycle(n, m, x_val, Pnm_m1[m], Pnm_m2[m]);
        Pnm_m2[m] = Pnm_m1[m];
        Pnm_m1[m] = P_n_m;
        mphi = (double) m * phi_proj;
        rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
        loc -= 2;
      }

      P_n_m = x_val * std::sqrt((2.0 * ((double) n - 1.0)) + 3.0) * Pnm_nn;
      Pnm_m2[n - 1] = Pnm_m1[n - 1];
      Pnm_m1[n - 1] = P_n_m;
      mphi = (double) (n - 1) * phi_proj;
      rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
      loc -= 2;

      P_n_m = plegendre_nn(n, x_val, Pnm_nn);
      Pnm_nn = P_n_m;
      Pnm_m1[n] = P_n_m;
      mphi = (double) n * phi_proj;
      rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
    }

    sh_dist = expfacts_byshape[sht][n] * (rad_val);

    if (outerdist > sh_dist) {
      return 0;
    } else {
      if (++n > maxshexpan) {
        if (outerdist <= sh_dist) {
          finalrad = rad_val;
          return 1;
        } else return 0;
      }
    }
  }
  return 0;
}

/* ----------------------------------------------------------------------
  Given a shape and a spherical coordinate (value of theta and phi), return
  the radius at the maximum degree of spherical harmonic expansion.
------------------------------------------------------------------------- */
double AtomVecSpherharm::get_shape_radius(int sht, double theta, double phi) {

  double rad_val = shcoeffs_byshape[sht][0] * std::sqrt(1.0 / (4.0 * MY_PI));

  int n, nloc, loc;
  double P_n_m, x_val, mphi, Pnm_nn;
  std::vector<double> Pnm_m2, Pnm_m1;

  Pnm_m2.resize(maxshexpan+1, 0.0);
  Pnm_m1.resize(maxshexpan+1, 0.0);
  x_val = std::cos(theta);
  for (n=1; n<=maxshexpan; n++){
    nloc = n * (n + 1);
    if (n == 1) {
      P_n_m = plegendre(1, 0, x_val);
      Pnm_m2[0] = P_n_m;
      rad_val += shcoeffs_byshape[sht][4] * P_n_m;
      P_n_m = plegendre(1, 1, x_val);
      Pnm_m2[1] = P_n_m;
      mphi = 1.0 * phi;
      rad_val += (shcoeffs_byshape[sht][2] * cos(mphi) - shcoeffs_byshape[sht][3] * sin(mphi)) * 2.0 * P_n_m;
    } else if (n == 2) {
      P_n_m = plegendre(2, 0, x_val);
      Pnm_m1[0] = P_n_m;
      rad_val += shcoeffs_byshape[sht][10] * P_n_m;
      for (int m = 2; m >= 1; m--) {
        P_n_m = plegendre(2, m, x_val);
        Pnm_m1[m] = P_n_m;
        mphi = (double) m * phi;
        rad_val += (shcoeffs_byshape[sht][nloc] * cos(mphi) - shcoeffs_byshape[sht][nloc + 1] * sin(mphi)) * 2.0 * P_n_m;
        nloc += 2;
      }
      Pnm_nn = Pnm_m1[2];
    } else {
      P_n_m = plegendre_recycle(n, 0, x_val, Pnm_m1[0], Pnm_m2[0]);
      Pnm_m2[0] = Pnm_m1[0];
      Pnm_m1[0] = P_n_m;
      loc = (n + 1) * (n + 2) - 2;
      rad_val += shcoeffs_byshape[sht][loc] * P_n_m;
      loc -= 2;
      for (int m = 1; m < n - 1; m++) {
        P_n_m = plegendre_recycle(n, m, x_val, Pnm_m1[m], Pnm_m2[m]);
        Pnm_m2[m] = Pnm_m1[m];
        Pnm_m1[m] = P_n_m;
        mphi = (double) m * phi;
        rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
        loc -= 2;
      }

      P_n_m = x_val * std::sqrt((2.0 * ((double) n - 1.0)) + 3.0) * Pnm_nn;
      Pnm_m2[n - 1] = Pnm_m1[n - 1];
      Pnm_m1[n - 1] = P_n_m;
      mphi = (double) (n - 1) * phi;
      rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
      loc -= 2;

      P_n_m = plegendre_nn(n, x_val, Pnm_nn);
      Pnm_nn = P_n_m;
      Pnm_m1[n] = P_n_m;
      mphi = (double) n * phi;
      rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
    }
  }
  return rad_val;
}

/* ----------------------------------------------------------------------
  Given a shape and a spherical coordinate (value of theta and phi), return
  the radius at the maximum degree of spherical harmonic expansion.
------------------------------------------------------------------------- */
double AtomVecSpherharm::get_shape_radius_and_normal(int sht, double theta, double phi, double rnorm[3]) {

  int n, nloc, loc;
  double P_n_m, x_val, mphi, Pnm_nn, fnm;
  std::vector<double> Pnm_m2, Pnm_m1;

  fnm = std::sqrt(1.0 / MY_4PI);
  double rad_val = shcoeffs_byshape[sht][0] * fnm;
  double rad_dphi, rad_dtheta;
  rad_dphi = rad_dtheta = 0.0;

  Pnm_m2.resize(maxshexpan+1, 0.0);
  Pnm_m1.resize(maxshexpan+1, 0.0);

  if (sin(theta) == 0.0) theta += 1.0e-5; // otherwise dividing by sin(theta) for gradients will not work
  if (sin(phi) == 0.0) phi += 1.0e-5; // To be consistent...
  x_val = std::cos(theta);

  for (n=1; n<=maxshexpan; n++){
    nloc = n * (n + 1);

    // n=1
    if (n == 1) {
      // n=1, m=0
      P_n_m = plegendre(1, 0, x_val);
      Pnm_m2[0] = P_n_m;
      rad_val += shcoeffs_byshape[sht][4] * P_n_m;
      fnm = std::sqrt(3.0 / MY_4PI);
      rad_dtheta -= (shcoeffs_byshape[sht][4]*fnm / std::sin(theta))*((2.0*x_val*plgndr(1,0,x_val))-(2.0*plgndr(2, 0, x_val)));
      // n=1, m=1
      P_n_m = plegendre(1, 1, x_val);
      Pnm_m2[1] = P_n_m;
      mphi = 1.0 * phi;
      rad_val += (shcoeffs_byshape[sht][2] * cos(mphi) - shcoeffs_byshape[sht][3] * sin(mphi)) * 2.0 * P_n_m;
      rad_dphi -= (shcoeffs_byshape[sht][2] * sin(mphi) + shcoeffs_byshape[sht][3] * cos(mphi)) * 2.0 * P_n_m;
      fnm = std::sqrt(3.0 / (2.0 * MY_4PI));
      rad_dtheta += 2.0*(fnm/std::sin(theta))*((2.0*x_val*plgndr(1,1,x_val))-(plgndr(2, 1, x_val)))*
              ((shcoeffs_byshape[sht][3]*sin(mphi))-(shcoeffs_byshape[sht][2] * cos(mphi))) ;
      // n = 2
    } else if (n == 2) {
      // n=2, m=0
      P_n_m = plegendre(2, 0, x_val);
      Pnm_m1[0] = P_n_m;
      rad_val += shcoeffs_byshape[sht][10] * P_n_m;
      fnm = std::sqrt(5.0 / MY_4PI);
      rad_dtheta -= (shcoeffs_byshape[sht][10]*fnm / std::sin(theta))*((3.0*x_val*plgndr(2,0,x_val))-(3.0*plgndr(3, 0, x_val)));
      // n=2 2>=m>0
      for (int m = 2; m >= 1; m--) {
        P_n_m = plegendre(2, m, x_val);
        Pnm_m1[m] = P_n_m;
        mphi = (double) m * phi;
        rad_val += (shcoeffs_byshape[sht][nloc] * cos(mphi) - shcoeffs_byshape[sht][nloc + 1] * sin(mphi)) * 2.0 * P_n_m;
        rad_dphi -= (shcoeffs_byshape[sht][nloc + 1] * sin(mphi) + shcoeffs_byshape[sht][nloc] * cos(mphi)) * 2.0 * P_n_m * (double) m;
        fnm = std::sqrt((2.0*double(n)+1.0)*factorial(n-m)/(MY_4PI*factorial(n+m)));
        rad_dtheta += 2.0*(fnm/std::sin(theta))*((double(n+1)*x_val*plgndr(n,m,x_val))-(double(n-m+1)*plgndr(n+1, m, x_val)))*
                ((shcoeffs_byshape[sht][nloc+1]*sin(mphi))-(shcoeffs_byshape[sht][nloc] * cos(mphi)));
        nloc += 2;
      }
      Pnm_nn = Pnm_m1[2];

    // 2 < n > n-1
    } else {
      P_n_m = plegendre_recycle(n, 0, x_val, Pnm_m1[0], Pnm_m2[0]);
      Pnm_m2[0] = Pnm_m1[0];
      Pnm_m1[0] = P_n_m;
      loc = (n + 1) * (n + 2) - 2;
      rad_val += shcoeffs_byshape[sht][loc] * P_n_m;
      fnm = std::sqrt((2.0*double(n)+1.0)/(MY_4PI));
      rad_dtheta -= (shcoeffs_byshape[sht][loc]*fnm / std::sin(theta))*
              ((double(n+1)*x_val*plgndr(n,0,x_val))-(double(n+1)*plgndr(n+1, 0, x_val)));

      loc -= 2;
      for (int m = 1; m < n - 1; m++) {
        P_n_m = plegendre_recycle(n, m, x_val, Pnm_m1[m], Pnm_m2[m]);
        Pnm_m2[m] = Pnm_m1[m];
        Pnm_m1[m] = P_n_m;
        mphi = (double) m * phi;
        rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
        rad_dphi -= (shcoeffs_byshape[sht][nloc + 1] * sin(mphi) + shcoeffs_byshape[sht][nloc] * cos(mphi)) * 2.0 * P_n_m * (double) m;
        fnm = std::sqrt((2.0*double(n)+1.0)*factorial(n-m)/(MY_4PI*factorial(n+m)));
        rad_dtheta += 2.0*(fnm/std::sin(theta))*((double(n+1)*x_val*plgndr(n,m,x_val))-(double(n-m+1)*plgndr(n+1, m, x_val)))*
                      ((shcoeffs_byshape[sht][nloc+1]*sin(mphi))-(shcoeffs_byshape[sht][nloc] * cos(mphi)));
        loc -= 2;
      }

      // m = n-1
      P_n_m = x_val * std::sqrt((2.0 * ((double) n - 1.0)) + 3.0) * Pnm_nn;
      Pnm_m2[n - 1] = Pnm_m1[n - 1];
      Pnm_m1[n - 1] = P_n_m;
      mphi = (double) (n - 1) * phi;
      rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
      rad_dphi -= (shcoeffs_byshape[sht][nloc + 1] * sin(mphi) + shcoeffs_byshape[sht][nloc] * cos(mphi)) * 2.0 * P_n_m * (double) (n-1);
      fnm = std::sqrt((2.0*double(n)+1.0)/(MY_4PI*factorial(2*n-1)));
      rad_dtheta += 2.0*(fnm/std::sin(theta))*((double(n+1)*x_val*plgndr(n,n-1,x_val))-(2.0*plgndr(n+1, n-1, x_val)))*
                    ((shcoeffs_byshape[sht][nloc+1]*sin(mphi))-(shcoeffs_byshape[sht][nloc] * cos(mphi)));
      loc -= 2;

      // m = n
      P_n_m = plegendre_nn(n, x_val, Pnm_nn);
      Pnm_nn = P_n_m;
      Pnm_m1[n] = P_n_m;
      mphi = (double) n * phi;
      rad_val += (shcoeffs_byshape[sht][loc] * cos(mphi) - shcoeffs_byshape[sht][loc + 1] * sin(mphi)) * 2.0 * P_n_m;
      rad_dphi -= (shcoeffs_byshape[sht][nloc + 1] * sin(mphi) + shcoeffs_byshape[sht][nloc] * cos(mphi)) * 2.0 * P_n_m * (double) n;
      fnm = std::sqrt((2.0*double(n)+1.0)/(MY_4PI*factorial(2*n)));
      rad_dtheta += 2.0*(fnm/std::sin(theta))*((double(n+1)*x_val*plgndr(n,n,x_val))-(plgndr(n+1, n, x_val)))*
                    ((shcoeffs_byshape[sht][nloc+1]*sin(mphi))-(shcoeffs_byshape[sht][nloc] * cos(mphi)));
    }
  }

//  std::cout << "theta " << theta << " phi " << phi << " drdt " << rad_dtheta << " drdp " << rad_dphi << std::endl;
  get_normal(theta, phi, rad_val, rad_dphi, rad_dtheta, rnorm);

  return rad_val;
}

/* ----------------------------------------------------------------------
  Get the unit surface normal for a specified theta and phi value
------------------------------------------------------------------------- */
void AtomVecSpherharm::get_normal(double theta, double phi, double r, double rp, double rt, double rnorm[3]) {

  double st, sp, ct, cp;
  double denom;
  double sfac;

  st = std::sin(theta);
  ct = std::cos(theta);
  sp = std::sin(phi);
  cp = std::cos(phi);

  //sfac = r * std::sqrt((rp*rp)+(rt*rt*st*st)+(r*r*st*st));

  rnorm[0] = r * ((cp*r*st*st) + (sp*rp) - (cp*ct*st*rt));
  rnorm[1] = r * ((r*sp*st*st) - (cp*rp) - (ct*sp*st*rt));
  rnorm[2] = r * st * ((ct*r) + (st*rt));

  denom = sqrt((rnorm[0]*rnorm[0]) +
               (rnorm[1]*rnorm[1]) +
               (rnorm[2]*rnorm[2]));

  if (denom > 0.0) {
    rnorm[0] /= denom;
    rnorm[1] /= denom;
    rnorm[2] /= denom;
  } else {
    error->all(FLERR,"Zero Unit normal vector");
  }
}

/* ----------------------------------------------------------------------
  Reading in the shape coefficients as listed by the user in the input file
  and read by process args. Uses the LAMMPS in-built PotentialFileReader for
  reading the file. Note that files may list all coefficients, whilst we only
  want the coefficients for which m>=0. Coefficients for m<0 will be skipped.
------------------------------------------------------------------------- */
void AtomVecSpherharm::read_sh_coeffs(char *file, int shapenum){
  
  char * line;
  int nn, mm, entry;
  double a_real, a_imag;
  int NPARAMS_PER_LINE = 4;

  PotentialFileReader reader(lmp, file, "atom_vec_spherharm:coeffs input file");
  reader.ignore_comments(true);

  while((line = reader.next_line(NPARAMS_PER_LINE))) {
    try {
      ValueTokenizer values(line);

      nn = values.next_int();
      mm = values.next_int();

      if(nn>maxshexpan){
        break;
      }

      if (mm>=0){
        a_real = values.next_double();
        a_imag = values.next_double();
        entry = nn*(nn+1)+2*(nn-mm);
        shcoeffs_byshape[shapenum][entry] = a_real;
        shcoeffs_byshape[shapenum][++entry] = a_imag;
      }

    } catch (TokenizerException & e) {
      error->one(FLERR, e.what());
    }
  }

}

void AtomVecSpherharm::gen_extent_box_points(int n_ext){

  double theta, phi, rad;
  int count;

  for (int sht = 0; sht < nshtypes; sht++) {
    count = 0;
    for (int i = 0; i < n_ext; i++) {
      for (int j = 0; j < n_ext; j++) {
        theta = std::acos((((double)(i) * 2.0) / ((double)(n_ext)))-1.0);
        phi = (2.0 * MY_PI * (double)(j)) / ((double)((n_ext)));
        rad = get_shape_radius(sht, theta, phi);
        extentpoints_byshape[sht][count][0] = (rad * sin(theta) * cos(phi));
        extentpoints_byshape[sht][count][1] = (rad * sin(theta) * sin(phi));
        extentpoints_byshape[sht][count][2] = (rad * cos(theta));
        count++;
      }
    }
  }

}
