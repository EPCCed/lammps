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

#include "compute_particlenodes.h"

#include "iostream"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "comm.h"
#include "atom_vec_spherharm.h"
#include "atom.h"
#include "update.h"
#include "group.h"
#include "error.h"
#include "math_spherharm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeParticlenodes::ComputeParticlenodes(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{

  std::cout << "COMPUTE PARTICLE NODES INIT" << std::endl;

  if (narg != 4) error->all(FLERR,"Illegal compute nodes/atom command");

  peratom_flag = 1;

  // input is the number of nodes on a line of longitude
  size_peratom_cols = utils::inumeric(FLERR, arg[3], true, lmp);
  size_peratom_cols *= size_peratom_cols*3;

  nmax = 0;
  nodes = nullptr;
  avec = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeParticlenodes::~ComputeParticlenodes()
{
  memory->destroy(nodes);
}

/* ---------------------------------------------------------------------- */

void ComputeParticlenodes::init()
{
  avec = (AtomVecSpherharm *) atom->style_match("spherharm");
  if (!avec) error->all(FLERR,"Compute nodes/atom requires atom style spherharm");

  // check that all particles are sh particles
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *shtype = atom->shtype;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (shtype[i] < 0)
        error->one(FLERR,"Compute nodes/atom requires spherical harmonic particles");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"nodes/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute nodes/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeParticlenodes::compute_peratom()
{

  std::cout<<"RUNNING"<<std::endl;

  invoked_peratom = update->ntimestep;

  // grow nodes array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(nodes);
    nmax = atom->nmax;
    memory->create(nodes,nmax,size_peratom_cols,"nodes/atom:nodes");
    array_atom = nodes;
  }

  int *shtype = atom->shtype;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int ishtype, ind;
  int NumLongLatNodes = int (std::sqrt(size_peratom_cols/3));
  double theta, phi, rad;
  double abscissa[NumLongLatNodes];
  MathSpherharm::QuadPair p;

  // Get the quadrature weights, and abscissa. Convert abscissa to theta angles
  for (int i = 0; i < NumLongLatNodes; i++) {
    p = MathSpherharm::GLPair(NumLongLatNodes, i + 1);
    abscissa[i] = p.x();
  }

  for (int p = 0; p < nlocal; p++) {
    if (mask[p] & groupbit) {
      ishtype = shtype[p];
      ind=0;
      for (int i = 0; i < NumLongLatNodes; i++) {
        for (int j = 0; j < NumLongLatNodes; j++) {
//          theta = ((double)(i) * MathConst::MY_PI) / ((double)(NumLongLatNodes));
//          phi = (2.0 * MathConst::MY_PI * (double)(j)) / ((double)((NumLongLatNodes)));
//          rad = avec->get_shape_radius(ishtype, theta, phi);
          theta = 0.5 * MathConst::MY_PI * (abscissa[i] + 1.0);
          phi = MathConst::MY_PI * (abscissa[j] + 1.0);
          rad = avec->get_shape_radius(ishtype, theta, phi);
          nodes[p][ind++] = (rad * std::sin(theta) * std::cos(phi));
          nodes[p][ind++] = (rad * std::sin(theta) * std::sin(phi));
          nodes[p][ind++] = (rad * std::cos(theta));
        }
      }
    }
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeParticlenodes::memory_usage()
{
  double bytes = double (nmax) * size_peratom_cols * sizeof(double);
  return bytes;
}
