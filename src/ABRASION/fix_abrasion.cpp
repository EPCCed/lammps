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
   Contributing author: James Young (Edinburgh), adapted from Kevin Hanley (Imperial)
------------------------------------------------------------------------- */

#include "fix_abrasion.h"
#include "neighbor.h"
#include <cstdlib>
#include "atom.h"
#include "update.h"
#include "memory.h"
#include "force.h"
#include "integrate.h"
#include "fix.h"
#include "error.h"
#include "math_extra.h"
#include "library.h"
#include "neigh_list.h"
#include "update.h"

#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
 ---------------------------------------------------------------------- */

FixAbrasion::FixAbrasion(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        vertexdata(nullptr),
        list(nullptr)
{
  restart_peratom = 1; //~ Per-atom information is saved to the restart file
  peratom_flag = 1;
  size_peratom_cols = 7; //~ normal x/y/z, area and displacement speed x/y/z
  peratom_freq = 1; // every step, **TODO change to user input utils::inumeric(FLERR,arg[5],false,lmp);
  create_attribute = 1; //fix stores attributes that need setting when a new atom is created

  if (narg != 5) error->all(FLERR, "Incorrect number of arguments to the abrasion fix.");

  pf = utils::numeric(FLERR,arg[3],false,lmp);
  mu = utils::numeric(FLERR,arg[4],false,lmp);

  // perform initial allocation of atom-based array
  // register with Atom class
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  // Set initial values to zero
  for (int i = 0; i < atom->nlocal; i++) {
    vertexdata[i][0] = vertexdata[i][1] = vertexdata[i][2] = 0.0;
    vertexdata[i][3] = 0.0;
    vertexdata[i][4] = vertexdata[i][5] = vertexdata[i][6] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

FixAbrasion::~FixAbrasion()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,Atom::GROW);
  atom->delete_callback(id,Atom::RESTART);

  // delete locally stored array
  memory->destroy(vertexdata);
}

/* ---------------------------------------------------------------------- */

int FixAbrasion::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAbrasion::init()
{
  // need a full perpetual neighbor list
  neighbor->add_request(this,NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

void FixAbrasion::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

  /* ---------------------------------------------------------------------- */

void FixAbrasion::post_force(int vflag)
{
  areas_and_normals();

  double **f = atom->f;

  //  NeighList *l = neighbor->lists[list_index];
  std::cout << "inum: " << list->inum << std::endl;
  std::cout << "nlocal: " << atom->nlocal << std::endl;
  //  for (int i = 0; i < list->inum; i++) {
  for (int i = 0; i < atom->nlocal; i++) {
    bool force_acting = MathExtra::len3(f[i]);
    if (force_acting) {
      int local_index = list->ilist[i];
      int no_of_neighs = list->numneigh[i];
      std::cout << "atom: " << i << ", local index: " << local_index << ", numneigh: "
		<< list->numneigh[i] << std::endl;
      for (int j = 0; j < list->numneigh[i]; j++) {
	int neigh_index = list->firstneigh[i][j];
	double v_rel[3];
	double x_rel[3];
	double speed_of_approach;
	if (gap_is_shrinking(local_index, neigh_index, v_rel, x_rel, &speed_of_approach)) {
	  // A displacement occurs only if the objects are approaching each other.
	  displacement_of_atom(i, (atom->radius)[neigh_index], speed_of_approach,
			       v_rel, x_rel);
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAbrasion::memory_usage()
{
  double bytes = atom->nmax*size_peratom_cols * sizeof(double); //~ For vertexdata array
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixAbrasion::grow_arrays(int nmax)
{
  memory->grow(vertexdata,nmax,size_peratom_cols,"particle_meshs:vertexdata");
  array_atom = vertexdata;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixAbrasion::copy_arrays(int i, int j, int delflag)
{
  for (int q = 0; q < size_peratom_cols; q++)
    vertexdata[j][q] = vertexdata[i][q];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixAbrasion::set_arrays(int i)
{
  vertexdata[i][0] = 0.0;
  vertexdata[i][1] = 0.0;
  vertexdata[i][2] = 0.0;
  vertexdata[i][3] = 0.0;
  vertexdata[i][4] = 0.0;
  vertexdata[i][5] = 0.0;
  vertexdata[i][6] = 0.0;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixAbrasion::pack_exchange(int i, double *buf)
{
  for (int q = 0; q < size_peratom_cols; q++)
    buf[q] = vertexdata[i][q];

  return size_peratom_cols;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixAbrasion::unpack_exchange(int nlocal, double *buf)
{
  for (int q = 0; q < size_peratom_cols; q++)
    vertexdata[nlocal][q] = buf[q];

  return size_peratom_cols;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixAbrasion::pack_restart(int i, double *buf)
{
  buf[0] = size_peratom_cols+1;
  for (int q = 0; q < size_peratom_cols; q++)
    buf[q+1] = vertexdata[i][q];

  return size_peratom_cols+1;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixAbrasion::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  for (int q = 0; q < size_peratom_cols; q++)
    vertexdata[nlocal][q] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixAbrasion::maxsize_restart()
{
  return size_peratom_cols+1;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixAbrasion::size_restart(int nlocal)
{
  return size_peratom_cols+1;
}

void FixAbrasion::areas_and_normals() {

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

  //  std::cout << "Entering areas_and_normals()" << std::endl;

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (int i = 0; i < nlocal; i++) {
    vertexdata[i][0] = 0.0;
    vertexdata[i][1] = 0.0;
    vertexdata[i][2] = 0.0;
    vertexdata[i][3] = 0.0;
    vertexdata[i][4] = 0.0;
    vertexdata[i][5] = 0.0;
    vertexdata[i][6] = 0.0;
  }

  norm1 = 0.0;
  norm2 = 0.0;
  norm3 = 0.0;
  sub_area = 0.0;

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

    //    std::cout << "Normal to facet: " << n1 << " " << n2 << " " << n3 << std::endl;
    //    std::cout << "Area of facet: " << area/2 << std::endl;

    // Centroid of the vertices making the current triangle
    centroid[0] = (x[i1][0] + x[i2][0] + x[i3][0])/3.0;
    centroid[1] = (x[i1][1] + x[i2][1] + x[i3][1])/3.0;
    centroid[2] = (x[i1][2] + x[i2][2] + x[i3][2])/3.0;
    //    std::cout << "Centroid: " << centroid[0] << " " << centroid[1]
    //	      << " " << centroid[2] << std::endl;

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
    //    std::cout << "i1: " << i1 << std::endl;
    //    std::cout << "rmass: " << atom->rmass[i1] << std::endl;
    //    std::cout << "radius: " << (atom->radius)[i1] << std::endl;

    sub_area = 0.25 * abs[0] * abs[1] * st[0];
    vertexdata[i1][3] += sub_area;
    vertexdata[i1][0] += n1*sub_area;
    vertexdata[i1][1] += n2*sub_area;
    vertexdata[i1][2] += n3*sub_area;
    vertexdata[i2][3] += sub_area;
    vertexdata[i2][0] += n1*sub_area;
    vertexdata[i2][1] += n2*sub_area;
    vertexdata[i2][2] += n3*sub_area;

    sub_area = 0.25 * abs[1] * abs[2] * st[1];
    vertexdata[i2][3] += sub_area;
    vertexdata[i2][0] += n1*sub_area;
    vertexdata[i2][1] += n2*sub_area;
    vertexdata[i2][2] += n3*sub_area;
    vertexdata[i3][3] += sub_area;
    vertexdata[i3][0] += n1*sub_area;
    vertexdata[i3][1] += n2*sub_area;
    vertexdata[i3][2] += n3*sub_area;

    sub_area = 0.25 * abs[2] * abs[0] * st[2];
    vertexdata[i3][3] += sub_area;
    vertexdata[i3][0] += n1*sub_area;
    vertexdata[i3][1] += n2*sub_area;
    vertexdata[i3][2] += n3*sub_area;
    vertexdata[i1][3] += sub_area;
    vertexdata[i1][0] += n1*sub_area;
    vertexdata[i1][1] += n2*sub_area;
    vertexdata[i1][2] += n3*sub_area;

  }

  for (int i = 0; i < nlocal; i++) {
    if (vertexdata[i][3] > 0) {
      norm1 = vertexdata[i][0];
      norm2 = vertexdata[i][1];
      norm3 = vertexdata[i][2];
      length = sqrt(norm1*norm1 + norm2*norm2 + norm3*norm3);
      vertexdata[i][0] = norm1/length;
      vertexdata[i][1] = norm2/length;
      vertexdata[i][2] = norm3/length;
    } else {
      vertexdata[i][0] = 0.0;
      vertexdata[i][1] = 0.0;
      vertexdata[i][2] = 0.0;
    }
    std::cout << "Constructed area associated with atom " << i << " is " << vertexdata[i][3]
	      << ". Normal to constructed area is: " << vertexdata[i][0]
	      << " " << vertexdata[i][1] << " " << vertexdata[i][2] << std::endl;
  }
}

bool FixAbrasion::gap_is_shrinking(int i, int neigh_index,
				   double v_rel[3], double x_rel[3],
				   double *vnorm_p) {

  double **v = atom->v;
  double **x = atom->x;

  std::cout << "neighbour vx, vy, vz: " << v[neigh_index][0] << ", " << v[neigh_index][1]
	    << ", " << v[neigh_index][2] << std::endl;

  std::cout << "neighbour x: " << x[neigh_index][0] << ", y: " << x[neigh_index][1]
	    << ", z: " << x[neigh_index][2] << std::endl;

  v_rel[0] = v[neigh_index][0] - v[i][0];
  v_rel[1] = v[neigh_index][1] - v[i][1];
  v_rel[2] = v[neigh_index][2] - v[i][2];

  double normal[3];
  normal[0] = vertexdata[i][0];
  normal[1] = vertexdata[i][1];
  normal[2] = vertexdata[i][2];

  *vnorm_p = MathExtra::dot3(v_rel, normal);
  std::cout << "vnorm = " << *vnorm_p
	    << ". vnorm < 0 indicates that the gap is shrinking." << std::endl;

  return *vnorm_p < 0;
}

void FixAbrasion::displacement_of_atom(int i, double r,
				       double vnorm,
				       double v_rel[3], double x_rel[3]) {
  double **f = atom->f;
  double normal[3];
  normal[0] = vertexdata[i][0];
  normal[1] = vertexdata[i][1];
  normal[2] = vertexdata[i][2];
  double area = vertexdata[i][3];

  double normal_component_of_force =  MathExtra::dot3(normal, f[i]);
  double tangential_force[3];
  tangential_force[0] = f[i][0] - normal_component_of_force*normal[0];
  tangential_force[1] = f[i][1] - normal_component_of_force*normal[1];
  tangential_force[2] = f[i][2] - normal_component_of_force*normal[2];
  double magnitude_of_tangential_force = MathExtra::len3(tangential_force);
  double dt = update->dt;

  double separation_of_centres = MathExtra::dot3(x_rel, normal);
  std::cout << "separation of centres = " << separation_of_centres << std::endl;

  // Effective radius.
  double r_eff = r + (atom->radius)[i];
  std::cout << "r =  " << r << ", r_eff =" << r_eff << std::endl;

  // Overlap of spheres.
  double h = r_eff - separation_of_centres;
  std::cout << "overlap = " << h << std::endl;
  
  // Tangential velocity.
  double normal_component_of_vel =  MathExtra::dot3(normal, v_rel);
  double v_tan[3];
  v_tan[0] = v_rel[0] -  normal_component_of_vel*normal[0];
  v_tan[1] = v_rel[1] -  normal_component_of_vel*normal[1];
  v_tan[2] = v_rel[2] -  normal_component_of_vel*normal[2];

  // Tangential displacement.
  double deltas = MathExtra::len3(v_tan)*dt;

  double deltah = 0.0;
  std::cout << "normal_component_of_force = " << normal_component_of_force << std::endl;
  std::cout << "area*pf = " << area*pf << std::endl;
  bool iV = abs(normal_component_of_force) > area*pf;
  if (iV) {
    deltah = vnorm*dt;
    std::cout << "deltah = " << deltah << std::endl; 
  }

  // This is the normal displacement corresponding to deltas.
  double dsh = 0.0;
  bool iL = magnitude_of_tangential_force > area*mu*pf;
  bool overlapping = h > 0;
  if (iL && overlapping) {
    dsh = h-r_eff+sqrt((r_eff-h)*(r_eff-h) + 2*sqrt(abs(2*r_eff*h-h*h))*deltas);
    std::cout << "dsh = " << dsh << ", deltas = " << deltah << std::endl;
  }

  // Total normal displacement.
  double total_normal_dispacement  = deltah - dsh;
  std::cout << "total_normal_dispacement = " << total_normal_dispacement << std::endl;
  double displ_speed = total_normal_dispacement/dt;
  vertexdata[i][4] = displ_speed*normal[0]; // x velocity for displacement
  vertexdata[i][5] = displ_speed*normal[1]; // y velocity for displacement
  vertexdata[i][6] = displ_speed*normal[2]; // z velocity for displacement
}
