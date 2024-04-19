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
/* ------------------------------------------------------------------------
   Contributing authors: James Young (UoE)
                         Mohammad Imaran (UoE)
                         Kevin Hanley (UoE)

   Please cite the related publication:
   TBC
------------------------------------------------------------------------- */

#include "atom_vec_shdem.h"
#include "atom.h"
#include "error.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_extra.h"
#include "math_shdem.h"
#include "memory.h"
#include "potential_file_reader.h"


using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSHDEM;

#define EPSILON 1e-20
/* ---------------------------------------------------------------------- */

AtomVecSHDEM::AtomVecSHDEM(LAMMPS *lmp) : AtomVec(lmp)
{
  shcoeffs_byshape = pinertia_byshape = quatinit_byshape = nullptr;
  expfacts_byshape = quad_rads_byshape = angles = nullptr;
  quat = angmom = omega = nullptr;
  maxrad_byshape = weights = nullptr;
  shtype = nullptr;
  num_quadrature = nshtypes = 0;
  vol_byshape = nullptr;
  maxshexpan = -1;


  mass_type = 0;    // not per-type mass arrays
  molecular = 0;    // 0 = atomic

  atom->shdem_flag = atom->rmass_flag = 1;
  atom->radius_flag = 0;    // Particles don't store radius
  atom->omega_flag = atom->torque_flag = atom->angmom_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = {"omega","torque", "shtype", "angmom", "quat", "rmass"};
  fields_copy =  {"omega", "shtype", "angmom", "quat", "rmass"};
  fields_comm =  {"quat"};
  fields_comm_vel = {"omega", "angmom", "quat"};
  fields_reverse =  {"torque"};
  fields_border = {"shtype","rmass"};
  fields_border_vel =  {"omega", "angmom", "shtype", "rmass", "quat"};
  fields_exchange = {"omega", "shtype", "angmom", "rmass"};
  fields_restart = {"omega", "shtype", "angmom", "rmass"};
  fields_create =  {"omega", "shtype", "angmom", "quat","rmass"};
  fields_data_atom =  {"id", "type", "shtype", "rmass", "quat", "x"};
  fields_data_vel = {"id", "v", "omega", "angmom"};
}

AtomVecSHDEM::~AtomVecSHDEM()
{
  memory->sfree(angles);
  memory->sfree(weights);
  memory->sfree(shcoeffs_byshape);
  memory->sfree(expfacts_byshape);
  memory->sfree(quad_rads_byshape);
  memory->sfree(vol_byshape);

}

/* ----------------------------------------------------------------------
   process sub-style args
------------------------------------------------------------------------- */

void AtomVecSHDEM::process_args(int narg, char **arg)
{

  int num_quad2, numcoeffs, me;
  MPI_Comm_rank(world, &me);

  if (narg < 3) error->all(FLERR, "llegal atom_style shdem command");

  maxshexpan = utils::inumeric(FLERR, arg[0], true, lmp);    // Maximum degree of the SH expansion
  num_quadrature =
      utils::inumeric(FLERR, arg[1], true, lmp);    // Order of the numerical quadrature
  nshtypes = narg - 2;                              // Number of SH types
  atom->nshtypes = nshtypes;                        // Setting the atom property

  num_quad2 = num_quadrature * num_quadrature;
  // Coefficient storage is not duplicated, i.e negative "m" values are not stored due to their relationship to the
  // positive "m" values: a_{n,-m} = (-1)^m a_{n,m}*, where * denotes the complex conjugate. For more information on
  // the coefficients see Spherical harmonic-based random fields for aggregates used in concrete by Grigoriu et. al.
  numcoeffs = (maxshexpan + 1) * (maxshexpan + 2);

  // Memory allocation local to atom_vec, must be deleted in class destructor
  memory->create(angles, 2, num_quad2, "AtomVecSHDEM:angles");
  memory->create(weights, num_quadrature, "AtomVecSHDEM:weights");
  memory->create(quad_rads_byshape, nshtypes, num_quad2, "AtomVecSHDEM:quad_rads_byshape");
  memory->create(shcoeffs_byshape, nshtypes, numcoeffs, "AtomVecSHDEM:shcoeff");
  memory->create(expfacts_byshape, nshtypes, maxshexpan + 1, "AtomVecSHDEM:expfacts_byshape");
  memory->create(vol_byshape, nshtypes, "AtomVecSHDEM:vol_byshape");

  // Atom memory allocation, must be deleted in atom class destructor
  memory->create(atom->pinertia_byshape, nshtypes, 3, "AtomVecSHDEM:pinertia");
  memory->create(atom->quatinit_byshape, nshtypes, 4, "AtomVecSHDEM:orient");
  memory->create(atom->maxrad_byshape, nshtypes, "AtomVecSHDEM:maxrad_byshape");

  // Directing the local pointers to the memory just allocated in the atom class
  pinertia_byshape = atom->pinertia_byshape;
  quatinit_byshape = atom->quatinit_byshape;
  maxrad_byshape = atom->maxrad_byshape;

  // Pre-allocating arrays to zero for all types and coefficients
  for (int type = 0; type < nshtypes; type++) {
    maxrad_byshape[type] = 0.0;
    for (int i = 0; i < numcoeffs; i++) { shcoeffs_byshape[type][i] = 0.0; }
  }

  if (me == 0) {    // Only want the 0th processor to read in the coefficients
    for (int i = 2; i < narg; i++) {    // Can list a number of files storing coefficients, each will be read in turn
      read_sh_coeffs(arg[i], i - 2);    // method for coefficient reading
    }


    // Get the weights, abscissa (theta and phi values) and radius for each quadrature point (radius is per-shape)
    get_quadrature_values();
    // Get the principal moment of inertia for each shape and the initial quaternion, this is referenced by all
    // particles that make use of that shape (they have a current quaternion that describes their current orientation
    // from the reference). Also gets the volume of each shape.
    getI();

    // Calculate the expansion factors as described by "A hierarchical, spherical harmonic-based approach to simulate
    // abradable, irregularly shaped particles in DEM" by Capozza and Hanley. The inverse approach is adopted here.

    calcexpansionfactors_gauss();
  }

  MPI_Bcast(&(angles[0][0]), 2 * num_quad2, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(weights[0]), num_quadrature, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(quad_rads_byshape[0][0]), nshtypes * num_quad2, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(pinertia_byshape[0][0]), nshtypes * 3, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(quatinit_byshape[0][0]), nshtypes * 4, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(shcoeffs_byshape[0][0]), nshtypes * numcoeffs, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(expfacts_byshape[0][0]), nshtypes * maxshexpan + 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(maxrad_byshape[0]), nshtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&(vol_byshape[0]), nshtypes, MPI_DOUBLE, 0, world);

  setup_fields();
}

void AtomVecSHDEM::init()
{
  AtomVec::init();
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()

   Not growing per-shape values as new atoms do not add new shapes. Shapes
   are defined once, in the process args method.
------------------------------------------------------------------------- */

void AtomVecSHDEM::grow_pointers()
{
  omega = atom->omega;
  shtype = atom->shtype;
  angmom = atom->angmom;
  quat = atom->quat;
  rmass = atom->rmass;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecSHDEM::create_atom_post(int ilocal)
{
  shtype[ilocal] = 0;
  quat[ilocal][0] = 1.0;
  quat[ilocal][1] = 0.0;
  quat[ilocal][2] = 0.0;
  quat[ilocal][3] = 0.0;
  rmass[ilocal] = 1.0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecSHDEM::data_atom_post(int ilocal)
{
  omega[ilocal][0] = 0.0;
  omega[ilocal][1] = 0.0;
  omega[ilocal][2] = 0.0;
  angmom[ilocal][0] = 0.0;
  angmom[ilocal][1] = 0.0;
  angmom[ilocal][2] = 0.0;
  shtype[ilocal] -= 1;
  rmass[ilocal] *= vol_byshape[shtype[ilocal]];
}

/* ----------------------------------------------------------------------
   modify values for AtomVec::pack_data() to pack
------------------------------------------------------------------------- */

void AtomVecSHDEM::pack_data_pre(int ilocal)
{
  // Convert mass  to density
  rmass_one = rmass[ilocal];
  rmass[ilocal] = rmass_one / vol_byshape[shtype[ilocal]];
  shtype[ilocal] += 1;    // not using 0-based indexing in the read files
}

/* ----------------------------------------------------------------------
   unmodify values packed by AtomVec::pack_data()
------------------------------------------------------------------------- */

void AtomVecSHDEM::pack_data_post(int ilocal)
{
  //density back to mass
  rmass[ilocal] = rmass_one;
  shtype[ilocal] -= 1;
}

/* ----------------------------------------------------------------------
 Calculate the inertia of all SH particle types. This code as adapted from the TSQUARE package.
 See "Three-dimensional mathematical analysis of particle shape using X-ray tomography and spherical harmonics:
 Application to aggregates used in concrete" by Garboczi.
------------------------------------------------------------------------- */
void AtomVecSHDEM::getI()
{

  using std::cos;
  using std::cout;
  using std::endl;
  using std::fabs;
  using std::pow;
  using std::sin;
  using std::sqrt;

  std::vector<double> itensor;
  double i11, i22, i33, i12, i23, i13;
  double theta, phi, st, ct, sp, cp, r, fact;
  double factor = (0.5 * MY_PI * MY_PI);
  int count;

  int ierror;
  double inertia[3];
  double tensor[3][3], evectors[3][3];
  double cross[3];
  double ex[3];
  double ey[3];
  double ez[3];

  for (int sht = 0; sht < nshtypes; sht++) {

    vol_byshape[sht] = 0.0;
    itensor.clear();
    count = 0;
    i11 = i22 = i33 = i12 = i23 = i13 = 0.0;

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
        vol_byshape[sht] += (weights[i] * weights[j] * pow(r, 3) * st / 3.0);
        i11 += (fact * (1.0 - pow(cp * st, 2)));
        i22 += (fact * (1.0 - pow(sp * st, 2)));
        i33 += (fact * (1.0 - pow(ct, 2)));
        i12 -= (fact * cp * sp * st * st);
        i13 -= (fact * cp * ct * st);
        i23 -= (fact * sp * ct * st);
        count++;
      }
    }


    vol_byshape[sht] *= factor;
    i11 *= factor;
    i22 *= factor;
    i33 *= factor;
    i12 *= factor;
    i13 *= factor;
    i23 *= factor;
    if (vol_byshape[sht] > 0.0) {
      i11 /= vol_byshape[sht];
      i22 /= vol_byshape[sht];
      i33 /= vol_byshape[sht];
      i12 /= vol_byshape[sht];
      i13 /= vol_byshape[sht];
      i23 /= vol_byshape[sht];
      itensor.push_back(i11);
      itensor.push_back(i22);
      itensor.push_back(i33);
      itensor.push_back(i12);
      itensor.push_back(i13);
      itensor.push_back(i23);
    } else {
      error->all(FLERR, "Divide by vol = 0 in getI");
    }

    tensor[0][0] = itensor[0];
    tensor[1][1] = itensor[1];
    tensor[2][2] = itensor[2];
    tensor[1][2] = tensor[2][1] = itensor[5];
    tensor[0][2] = tensor[2][0] = itensor[4];
    tensor[0][1] = tensor[1][0] = itensor[3];


    ierror = MathEigen::jacobi3(tensor, inertia, evectors);
    if (ierror) error->all(FLERR, "Insufficient Jacobi rotations for rigid body");
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
    max = MAX(inertia[0], inertia[1]);
    max = MAX(max, inertia[2]);

    if (inertia[0] < EPSILON * max) inertia[0] = 0.0;
    if (inertia[1] < EPSILON * max) inertia[1] = 0.0;
    if (inertia[2] < EPSILON * max) inertia[2] = 0.0;

    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd vector if needed
    MathExtra::cross3(ex, ey, cross);
    if (MathExtra::dot3(cross, ez) < 0.0) MathExtra::negate3(ez);


    // create initial quaternion
    MathExtra::exyz_to_q(ex, ey, ez, quatinit_byshape[sht]);

    pinertia_byshape[sht][0] = inertia[0];
    pinertia_byshape[sht][1] = inertia[1];
    pinertia_byshape[sht][2] = inertia[2];
  }
}

/* ----------------------------------------------------------------------
  Calculate the radi at the points of quadrature using the Spherical Harmonic
  expansion
------------------------------------------------------------------------- */
void AtomVecSHDEM::get_quadrature_values()
{

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

  count = 0;
  for (int i = 0; i < num_quadrature; i++) {
    for (int j = 0; j < num_quadrature; j++) {
      angles[0][count] = 0.5 * MY_PI * (abscissa[i] + 1.0);
      angles[1][count] = MY_PI * (abscissa[j] + 1.0);
      count++;
    }
  }

  num_quad2 = num_quadrature * num_quadrature;
  for (int sht = 0; sht < nshtypes; sht++) {
    for (int k = 0; k < num_quad2; k++) {
      theta = angles[0][k];
      phi = angles[1][k];
      quad_rads_byshape[sht][k] = get_shape_radius(sht, theta, phi);
    }
  }
}

/* ----------------------------------------------------------------------
  Calculate the expansion factors for all particles using the points of Gaussian quadrature
  (clustering at poles, spreading at the equator)
------------------------------------------------------------------------- */

void AtomVecSHDEM::calcexpansionfactors_gauss()
{

  double theta, phi, factor;
  int num_quad2 = num_quadrature * num_quadrature;

  std::vector<double> r_n, r_npo;
  std::vector<double> ratios, expfactors;
  r_n.resize(num_quad2, 0.0);
  r_npo.resize(num_quad2, 0.0);
  ratios.resize(num_quad2, 0.0);
  expfactors.resize(maxshexpan + 1, 0.0);
  expfactors[maxshexpan] = 1.0;

  for (int sht = 0; sht < nshtypes; sht++) {

    std::fill(r_n.begin(), r_n.end(), 0.0);

    for (int n = 0; n <= maxshexpan; n++) {    // For each harmonic n
      int k = 0;
      for (int i = 0; i < num_quadrature; i++) {
        for (int j = 0; j < num_quadrature; j++) {
	  int m = 0;
	  int indexa = n*(n+1) + 2*(n-m);

	  double areal = shcoeffs_byshape[sht][indexa];
	  double aimag = 0.0;

          theta = angles[0][k];
          phi   = angles[1][k];

          r_n[k] += areal*std::sph_legendre(n, m, theta);

          for (m = n; m > 0; m--) {    // For each m in current harmonic n
            double mphi = 1.0*m*phi;
	    double ynm = std::sph_legendre(n, m, theta);

	    indexa = n*(n+1) + 2*(n-m);
	    areal = shcoeffs_byshape[sht][indexa];
	    aimag = shcoeffs_byshape[sht][indexa+1];
            r_n[k] += 2.0*(areal*cos(mphi) - aimag*sin(mphi))*ynm;
          }

	  if (n == maxshexpan) {
	    // Get the maximum radius at the final harmonic
            if (r_n[k] > maxrad_byshape[sht]) maxrad_byshape[sht] = r_n[k];
	  }
	  else {
	    // Get the ratios of radii between subsequent harmonics
	    int np1 = n + 1;
	    m = 0;
            r_npo[k] = r_n[k];

	    indexa = np1*(np1 + 1) + 2*(np1 - m);
	    areal = shcoeffs_byshape[sht][indexa];
            r_npo[k] += areal*std::sph_legendre(np1, m, theta);

            for (m = np1; m > 0; m--) {
              double mphi = 1.0*m*phi;
	      double ynm = std::sph_legendre(np1, m, theta);

	      indexa = np1*(np1 + 1) + 2*(np1 - m);
	      areal = shcoeffs_byshape[sht][indexa];
	      aimag = shcoeffs_byshape[sht][indexa+1];
              r_npo[k] += 2.0*(areal*cos(mphi) - aimag*sin(mphi))*ynm;
            }
            ratios[k] = r_npo[k] / r_n[k];
          }
	  // Next angle ...
          k++;
        }
      }

      if (n <= maxshexpan - 1) {
        double max_val = 0;
        for (int ii = 0; ii < k; ii++) {
          if (ratios[ii] > max_val) max_val = ratios[ii];
        }
        expfactors[n] = max_val;
        if (expfactors[n] < 1.0) expfactors[n] = 1.0;
      }
    }

    factor = expfactors[maxshexpan];
    for (int n = maxshexpan - 1; n >= 0; n--) {
      factor *= expfactors[n];
      expfactors[n] = factor;
      expfacts_byshape[sht][n] = factor;
    }
    expfacts_byshape[sht][maxshexpan] = 1.0;
  }

  return;
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

int AtomVecSHDEM::check_contact(int sht, double phi, double theta,
				double outerdist,
				double &radius) {
  int n = 0;
  int indexa = 0;
  double sh_dist = 0.0;

  radius  = shcoeffs_byshape[sht][indexa]*std::sph_legendre(0, 0, theta);
  sh_dist = expfacts_byshape[sht][n]*radius;

  // If the input distance > the 0th harmonic radius,
  // then it is greater than the radius for all subsequent harmonics

  if (outerdist > sh_dist) return 0;

  // Edge case for spheres when the maximum harmonic is 0
  if (maxshexpan == 0 && outerdist <= sh_dist) return 1;


  for (n = 1; n <= maxshexpan; n++) {

    // n, m = 0 contribution
    int m = 0;
    indexa = n*(n + 1) + 2*(n - m);
    radius += shcoeffs_byshape[sht][indexa]*std::sph_legendre(n, m, theta);

    // n, +/- m contribution (... is two terms ...)
    for (m = 1; m <= n; m++) {
      indexa = n*(n + 1) + 2*(n - m);
      {
	double areal = shcoeffs_byshape[sht][indexa];
	double aimag = shcoeffs_byshape[sht][indexa + 1];
	double ynm   = std::sph_legendre(n, m, theta);
	radius += 2.0*(areal*cos(1.0*m*phi) - aimag*sin(1.0*m*phi))*ynm;
      }
    }

    sh_dist = expfacts_byshape[sht][n]*radius;

    if (outerdist > sh_dist) break;
  }

  if (outerdist <= sh_dist) return 1;

  return 0;
}

/* ----------------------------------------------------------------------
  Given a shape and a spherical coordinate (value of theta and phi), return
  the radius at the maximum degree of spherical harmonic expansion.
------------------------------------------------------------------------- */

double AtomVecSHDEM::get_shape_radius(int sht, double theta, double phi)
{

  int indexa = 0;
  double radius = 0.0;
  double a00 = shcoeffs_byshape[sht][indexa];

  radius += a00*std::sph_legendre(0, 0, theta);

  for (int n = 1; n <= maxshexpan; n++) {

    // n, m = 0 contribution
    int m = 0;
    indexa = n*(n + 1) + 2*(n - m);
    radius += shcoeffs_byshape[sht][indexa]*std::sph_legendre(n, m, theta);

    // n, +/- m contribution (... is two terms ...)
    for (m = 1; m <= n; m++) {
      indexa = n*(n + 1) + 2*(n - m);
      {
	double areal = shcoeffs_byshape[sht][indexa];
	double aimag = shcoeffs_byshape[sht][indexa + 1];
	double ynm   = std::sph_legendre(n, m, theta);
	radius += 2.0*(areal*cos(1.0*m*phi) - aimag*sin(1.0*m*phi))*ynm;
      }
    }
  }

  return radius;
}

/* ----------------------------------------------------------------------
  Given a shape and a spherical coordinate (value of theta and phi), return
  the radius at the maximum degree of spherical harmonic expansion.
------------------------------------------------------------------------- */
double AtomVecSHDEM::get_shape_radius_and_normal(int sht, double theta, double phi, double rnorm[3]) {

  double r = 0.0;
  double r_dtheta = 0.0;
  double r_dphi = 0.0;

  r = get_shape_radius_and_gradients(sht, theta, phi, r_dphi, r_dtheta);

  get_normal(theta, phi, r, r_dphi, r_dtheta, rnorm);

  return r;
}

/* ----------------------------------------------------------------------
  Given a shape and a spherical coordinate (value of theta and phi), return
  the radius at the maximum degree of spherical harmonic expansion and its
  gradients in theta and phi.
------------------------------------------------------------------------- */

double AtomVecSHDEM::get_shape_radius_and_gradients(int sht, double theta,
						    double phi,
						    double &r_dphi,
						    double &r_dtheta) {
  int indexa = 0;
  double radius = 0.0;

  double x = cos(theta);
  double y = sin(theta);
  double cottheta = x/y;
  double a00 = shcoeffs_byshape[sht][indexa];

  r_dtheta = 0.0;  // derivative w.r.t. theta
  r_dphi = 0.0;    // derivative w.r.t. phi

  radius += a00*std::sph_legendre(0, 0, theta);

  if (y == 0.0) cottheta = x/EPSILON;

  for (int n = 1; n <= maxshexpan; n++) {

    // n, m = 0 contribution
    int m = 0;
    indexa = n*(n + 1) + 2*(n - m);
    {
      double an0 = shcoeffs_byshape[sht][indexa];
      double ynm = std::sph_legendre(n, m, theta);
      double ynmp1 = std::sph_legendre(n, m + 1, theta);

      radius += an0*ynm;
      r_dtheta += an0*(cottheta*m*ynm + sqrt(1.0*(n-m)*(n+m+1))*ynmp1);
    }

    // n, +/- m contribution ...
    // Only non-zero m contribute to r_dphi
    for (m = 1; m <= n; m++) {
      indexa = n*(n + 1) + 2*(n - m);
      {
	double sinmphi = std::sin(1.0*m*phi);
	double cosmphi = std::cos(1.0*m*phi);
	double areal = shcoeffs_byshape[sht][indexa];
	double aimag = shcoeffs_byshape[sht][indexa + 1];
	double ynm   = std::sph_legendre(n, m, theta);
	double ynmp1 = std::sph_legendre(n, m + 1, theta);

	radius += 2.0*(areal*cosmphi - aimag*sinmphi)*ynm;

	r_dtheta -= 2.0*(cottheta*m*ynm + sqrt(1.0*(n-m)*(n+m+1))*ynmp1)
	  *(-areal*cosmphi + aimag*sinmphi);
	r_dphi   -= 2.0*m*(areal*sinmphi + aimag*cosmphi)*ynm;
      }
    }
  }

  return radius;
}


/* ----------------------------------------------------------------------
  Get the [NOT UNIT] surface normal for a specified theta and phi value
------------------------------------------------------------------------- */
void AtomVecSHDEM::get_normal(double theta, double phi, double r, double rp, double rt,
                                  double rnorm[3])
{

  double st, sp, ct, cp;

  st = std::sin(theta);
  ct = std::cos(theta);
  sp = std::sin(phi);
  cp = std::cos(phi);


  rnorm[0] = r * ((cp * r * st * st) + (sp * rp) - (cp * ct * st * rt));
  rnorm[1] = r * ((r * sp * st * st) - (cp * rp) - (ct * sp * st * rt));
  rnorm[2] = r * st * ((ct * r) + (st * rt));
}

/* ----------------------------------------------------------------------
  Reading in the shape coefficients as listed by the user in the input file
  and read by process args. Uses the LAMMPS in-built PotentialFileReader for
  reading the file. Note that files may list all coefficients, whilst we only
  want the coefficients for which m>=0. Coefficients for m<0 will be skipped.
------------------------------------------------------------------------- */
void AtomVecSHDEM::read_sh_coeffs(const char *file, int shapenum)
{

  char *line;
  int nn, mm, entry;
  double a_real, a_imag;
  int NPARAMS_PER_LINE = 4;

  PotentialFileReader reader(lmp, file, "atom_vec_shdem:coeffs input file");
  reader.ignore_comments(true);

  while ((line = reader.next_line(NPARAMS_PER_LINE))) {
    try {
      ValueTokenizer values(line);

      nn = values.next_int();
      mm = values.next_int();

      if (nn > maxshexpan) { break; }

      if (mm >= 0) {
        a_real = values.next_double();
        a_imag = values.next_double();
        entry = nn * (nn + 1) + 2 * (nn - mm);
        shcoeffs_byshape[shapenum][entry] = a_real;
        shcoeffs_byshape[shapenum][++entry] = a_imag;
      }

    } catch (TokenizerException &e) {
      error->one(FLERR, e.what());
    }
  }
}

double AtomVecSHDEM::get_shape_volume(int sht)
{
  return vol_byshape[sht];
}

int AtomVecSHDEM::get_max_expansion() const
{
  return maxshexpan;
}
