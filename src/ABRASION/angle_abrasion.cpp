/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "angle_abrasion.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neighbor.h"
#include "math_extra.h"
#include "atom_vec_angle.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

/* ---------------------------------------------------------------------- */

AngleAbrasion::AngleAbrasion(LAMMPS *_lmp) : Angle(_lmp)
{
  born_matrix_enable = 1;
}

/* ---------------------------------------------------------------------- */

AngleAbrasion::~AngleAbrasion()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
  }
}

/* ---------------------------------------------------------------------- */

void AngleAbrasion::compute(int eflag, int vflag)
{
  int i1, i2, i3, n, type;
  double delx1, dely1, delz1, delx2, dely2, delz2;
  double eangle, f1[3], f3[3];
  double rsq1, rsq2, r1, r2, c, a, a11, a12, a22;
  double axbi, axbj, axbk, area;
  double centroid[3], se1[3], se2[3], se3[3];
  double st[3], dots[3], abs[3];
  double sub_area;
  double norm1, norm2, norm3, length;
  double n1, n2, n3;

  eangle = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (int i = 0; i < nlocal; i++) {
    (atom->normal)[i][0] = 0.0;
    (atom->normal)[i][1] = 0.0;
    (atom->normal)[i][2] = 0.0;
    (atom->area)[i] = 0.0;
  }
  norm1 = 0.0;
  norm2 = 0.0;
  norm3 = 0.0;
  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // 1st bond

    delx1 = x[i1][0] - x[i2][0];
    dely1 = x[i1][1] - x[i2][1];
    delz1 = x[i1][2] - x[i2][2];

    rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3][0] - x[i2][0];
    dely2 = x[i3][1] - x[i2][1];
    delz2 = x[i3][2] - x[i2][2];

    rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
    r2 = sqrt(rsq2);

    //**** start of inserted code

    // angle

    std::cout << "****1****" << std::endl;

    std::cout << "Angle " << n << std::endl;
    std::cout << "atom1 " << x[i1][0] << " " << x[i1][1] << " " << x[i1][2] << std::endl;
    std::cout << "atom2 " << x[i2][0] << " " << x[i2][1] << " " << x[i2][2] << std::endl;
    std::cout << "atom3 " << x[i3][0] << " " << x[i3][1] << " " << x[i3][2] << std::endl;

    // cross product
    // a x b = (a2.b3 - a3.b2)i + (a3.b1 - a1.b3)j + (a1.b2 - a2.b1)k
    // b is the first bond whilst a is the second bond.

    axbi = dely2*delz1 - delz2*dely1;
    axbj = delz2*delx1 - delx2*delz1;
    axbk = delx2*dely1 - dely2*delx1;

    // area of facet

    area = sqrt(axbi*axbi + axbj*axbj + axbk*axbk); // actually 2*area

    n1 = axbi/area;
    n2 = axbj/area;
    n3 = axbk/area;

    std::cout << "Normal to facet: " << n1 << " " << n2 << " " << n3 << std::endl;
    std::cout << "Area of facet: " << area/2 << std::endl;

    // Centroid of the vertices making the current triangle
    centroid[0] = (x[i1][0] + x[i2][0] + x[i3][0])/3.0;
    centroid[1] = (x[i1][1] + x[i2][1] + x[i3][1])/3.0;
    centroid[2] = (x[i1][2] + x[i2][2] + x[i3][2])/3.0;
    std::cout << "Centroid: " << centroid[0] << " " << centroid[1]
	      << " " << centroid[2] << std::endl;

    // Sub-edge 1
    se1[0] = x[i1][0] - centroid[0];
    se1[1] = x[i1][1] - centroid[1];
    se1[2] = x[i1][2] - centroid[2];

    // Sub-edge 2
    se2[0] = x[i2][0] - centroid[0];
    se2[1] = x[i2][1] - centroid[1];
    se2[2] = x[i2][2] - centroid[2];

    // Sub-edge 3
    se3[0] = x[i3][0] - centroid[0];
    se3[1] = x[i3][1] - centroid[1];
    se3[2] = x[i3][2] - centroid[2];

    // dots between sub-edges 1-2 2-3 3-1
    dots[0] = MathExtra::dot3(se1, se2);
    dots[1] = MathExtra::dot3(se2, se3);
    dots[2] = MathExtra::dot3(se3, se1);

    // absolute length of sub-edges
    abs[0] = MathExtra::len3(se1);
    abs[1] = MathExtra::len3(se2);
    abs[2] = MathExtra::len3(se3);

    // sin of the angle between sub-edges (from centroid to vertices)
    // sin(theta) = sqrt(1 - cos(theta)^2), cos(theta) = dots / abs
    st[0] = std::sqrt(1.0 - std::pow(dots[0]/(abs[0]*abs[1]),2));
    st[1] = std::sqrt(1.0 - std::pow(dots[1]/(abs[1]*abs[2]),2));
    st[2] = std::sqrt(1.0 - std::pow(dots[2]/(abs[2]*abs[0]),2));

    // Half of each sub-triangle associated with each vertex
    // A = 0.5 * se1 * se2 * st
    std::cout << "i1: " << i1 << std::endl;
    std::cout << "rmass: " << atom->rmass[i1] << std::endl;
    std::cout << "radius: " << (atom->radius)[i1] << std::endl;

    sub_area = 0.25 * abs[0] * abs[1] * st[0];
    (atom->area)[i1] += sub_area;
    (atom->normal)[i1][0] += n1*sub_area;
    (atom->normal)[i1][1] += n2*sub_area;
    (atom->normal)[i1][2] += n3*sub_area;
    (atom->area)[i2] += sub_area;
    (atom->normal)[i2][0] += n1*sub_area;
    (atom->normal)[i2][1] += n2*sub_area;
    (atom->normal)[i2][2] += n3*sub_area;

    sub_area = 0.25 * abs[1] * abs[2] * st[1];
    (atom->area)[i2] += sub_area;
    (atom->normal)[i2][0] += n1*sub_area;
    (atom->normal)[i2][1] += n2*sub_area;
    (atom->normal)[i2][2] += n3*sub_area;
    (atom->area)[i3] += sub_area;
    (atom->normal)[i3][0] += n1*sub_area;
    (atom->normal)[i3][1] += n2*sub_area;
    (atom->normal)[i3][2] += n3*sub_area;

    sub_area = 0.25 * abs[2] * abs[0] * st[2];
    (atom->area)[i3] += sub_area;
    (atom->normal)[i3][0] += n1*sub_area;
    (atom->normal)[i3][1] += n2*sub_area;
    (atom->normal)[i3][2] += n3*sub_area;
    (atom->area)[i1] += sub_area;
    (atom->normal)[i1][0] += n1*sub_area;
    (atom->normal)[i1][1] += n2*sub_area;
    (atom->normal)[i1][2] += n3*sub_area;

    std::cout << "****2****" << std::endl;

    //**** end of inserted code

    // c = cosine of angle

    c = delx1 * delx2 + dely1 * dely2 + delz1 * delz2;
    c /= r1 * r2;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    // force & energy

    if (eflag) eangle = k[type] * (1.0 + c);

    a = k[type];
    a11 = a * c / rsq1;
    a12 = -a / (r1 * r2);
    a22 = a * c / rsq2;

    f1[0] = a11 * delx1 + a12 * delx2;
    f1[1] = a11 * dely1 + a12 * dely2;
    f1[2] = a11 * delz1 + a12 * delz2;
    f3[0] = a22 * delx2 + a12 * delx1;
    f3[1] = a22 * dely2 + a12 * dely1;
    f3[2] = a22 * delz2 + a12 * delz1;

    // apply force to each of 3 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= f1[0] + f3[0];
      f[i2][1] -= f1[1] + f3[1];
      f[i2][2] -= f1[2] + f3[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (evflag)
      ev_tally(i1, i2, i3, nlocal, newton_bond, eangle, f1, f3, delx1, dely1, delz1, delx2, dely2,
               delz2);
  }

  // ******** start of 2nd lot of inserted code
  for (int i = 0; i < nlocal; i++) {
    (atom->area)[i] = (atom->area)[i]/3.0;
    std::cout << "Constructed area asscociated with atom " << i << " is " << (atom->area)[i]  << std::endl;

    norm1 = (atom->normal)[i][0];
    norm2 = (atom->normal)[i][1];
    norm3 = (atom->normal)[i][2];
    length = sqrt(norm1*norm1 + norm2*norm2 + norm3*norm3);
    (atom->normal)[i][0] = norm1/length;
    (atom->normal)[i][1] = norm2/length;
    (atom->normal)[i][2] = norm3/length;
    std::cout << "Normal to constructed area is: " << norm1/length
	      << " " << norm2/length << " " << norm3/length <<std::endl;
  }
  // ******** end of 2nd lot of inserted code
}

/* ---------------------------------------------------------------------- */

void AngleAbrasion::allocate()
{
  allocated = 1;
  const int np1 = atom->nangletypes + 1;

  memory->create(k, np1, "angle:k");
  memory->create(setflag, np1, "angle:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void AngleAbrasion::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Incorrect args for angle coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nangletypes, ilo, ihi, error);

  double k_one = utils::numeric(FLERR, arg[1], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for angle coefficients");
}

/* ---------------------------------------------------------------------- */

double AngleAbrasion::equilibrium_angle(int /*i*/)
{
  return MY_PI;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void AngleAbrasion::write_restart(FILE *fp)
{
  fwrite(&k[1], sizeof(double), atom->nangletypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void AngleAbrasion::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0)
    utils::sfread(FLERR, &k[1], sizeof(double), atom->nangletypes, fp, nullptr, error);
  MPI_Bcast(&k[1], atom->nangletypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nangletypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void AngleAbrasion::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nangletypes; i++) fprintf(fp, "%d %g\n", i, k[i]);
}

/* ---------------------------------------------------------------------- */

double AngleAbrasion::single(int type, int i1, int i2, int i3)
{
  double **x = atom->x;

  double delx1 = x[i1][0] - x[i2][0];
  double dely1 = x[i1][1] - x[i2][1];
  double delz1 = x[i1][2] - x[i2][2];
  domain->minimum_image(delx1, dely1, delz1);
  double r1 = sqrt(delx1 * delx1 + dely1 * dely1 + delz1 * delz1);

  double delx2 = x[i3][0] - x[i2][0];
  double dely2 = x[i3][1] - x[i2][1];
  double delz2 = x[i3][2] - x[i2][2];
  domain->minimum_image(delx2, dely2, delz2);
  double r2 = sqrt(delx2 * delx2 + dely2 * dely2 + delz2 * delz2);

  double c = delx1 * delx2 + dely1 * dely2 + delz1 * delz2;
  c /= r1 * r2;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  return k[type] * (1.0 + c);
}

/* ---------------------------------------------------------------------- */

void AngleAbrasion::born_matrix(int type, int /*i1*/, int /*i2*/, int /*i3*/, double &du, double &du2)
{
  du2 = 0;
  du = k[type];
}

/* ----------------------------------------------------------------------
   return ptr to internal members upon request
------------------------------------------------------------------------ */

void *AngleAbrasion::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "k") == 0) return (void *) k;
  return nullptr;
}
