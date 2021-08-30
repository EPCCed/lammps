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

#include "compute_particlemesh.h"

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

#include <CGAL/pca_estimate_normals.h>
#include <CGAL/vcm_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/property_map.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/tags.h>
#include <CGAL/IO/write_xyz_points.h>

#include "Octree.h"
#include "OctreeIterator.h"
#include "utilities.h"
#include "Vertex.h"
#include "Mesher.h"
#include "FileIO.h"
#include "types.h"
#include <iostream>
#include <sstream>
#include <ctime>
#include <getopt.h>
#include <vector>

// kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

// Simple geometric types
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point3;
typedef Kernel::Vector_3 Vector;

// Point with normal vector stored in a std::pair.
typedef std::pair<Point3, Vector> PointVectorPair;
typedef std::tuple<int, Point3, Vector> PointVectorTuple;
typedef std::vector<PointVectorPair> PointList;
typedef std::vector<PointVectorTuple> PointListwithindex;

// Concurrency
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeParticlemesh::ComputeParticlemesh(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg!=3) error->all(FLERR,"Illegal compute mesh/atom command");


  //~ Check whether this fix is already present
  int feb = -1;
  for (int q = 0; q < modify->ncompute; q++)
    if (strcmp(modify->compute[q]->style,"nodes/atom") == 0) feb = q;

  //~ Fix not presently active
  // Unable to set up a new compute within the init of another compute
  if (feb < 0) {
    error->all(FLERR,"Illegal compute mesh/atom command requires nodes/atom");
  }else
  {
    defcomp = modify->compute[feb];
  }

  // input is the number of nodes on a line of longitude
  size_peratom_cols = defcomp->size_peratom_cols;
  peratom_flag = 1;
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeParticlemesh::~ComputeParticlemesh()
{
  memory->destroy(mesh);
}

/* ---------------------------------------------------------------------- */

void ComputeParticlemesh::init()
{
  std::cout << "Starting with init" << std::endl;

  // check that all particles are sh particles
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *shtype = atom->shtype;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (shtype[i] < 0)
        error->one(FLERR,"Compute mesh/atom requires spherical harmonic particles");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"mesh/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute mesh/atom");

  std::cout << "Done with init" << std::endl;
}

/* ---------------------------------------------------------------------- */

void ComputeParticlemesh::compute_peratom()
{

  std::cout<<"RUNNING"<<std::endl;

  unsigned int nb_neighbors_pca_normals = 18; // K-nearest neighbors = 3 rings (estimate normals by PCA)
  unsigned int nb_neighbors_mst = 60; // K-nearest neighbors (orient normals by MST)


  invoked_peratom = update->ntimestep;

  // grow nodes array if necessary
  if (atom->nmax > nmax) {
    memory->destroy(mesh);
    nmax = atom->nmax;
    memory->create(mesh,nmax,size_peratom_cols,"nodes/atom:mesh");
    array_atom = mesh;
  }


  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  PointListwithindex pointswithindex;
  int count, sht;

  count = 0;
  sht = 0;
  // Just one atom for now
  for (int p = 0; p < 1; p++) {
    if (mask[p] & groupbit) {
      for (int i = 0; i < size_peratom_cols; i+=3) {
        mesh[p][i] = mesh[p][i+1] = mesh[p][i+2] =  0;
        pointswithindex.emplace_back(count,
                                     Point3{defcomp->array_atom[p][i],
                                            defcomp->array_atom[p][i+1],
                                            defcomp->array_atom[p][i+2]},
                                     Vector{0,0,0});
        std::cout << count << " " << defcomp->array_atom[p][i] << " " << defcomp->array_atom[p][i+1]
                << " " << defcomp->array_atom[p][i+2] << std::endl;
        count++;
      }
    }
  }

  std::cout << "Number of cols " <<  size_peratom_cols << " " << pointswithindex.size() << std::endl;

  CGAL::pca_estimate_normals<Concurrency_tag>(pointswithindex,
                                              nb_neighbors_pca_normals,
                                              CGAL::parameters::point_map (CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
                                                      normal_map (CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>()));

  PointListwithindex::iterator unoriented_points_begin =
          CGAL::mst_orient_normals(pointswithindex,
                                   nb_neighbors_mst,
                                   CGAL::parameters::point_map (CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
                                           normal_map(CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>()));

//  std::sort(pointswithindex.begin(), pointswithindex.end(),
//            [](const PointVectorTuple & a,
//               const PointVectorTuple & b) {
//                return (std::get<0>(a) < std::get<0>(b));
//            });

  pointswithindex.erase(unoriented_points_begin, pointswithindex.end());

  std::cout << "Number of cols " <<  size_peratom_cols << " " << pointswithindex.size() << std::endl;

  std::string output_filename ("pointandnormal.xyz");
  std::cerr << "Write file " << output_filename << std::endl << std::endl;
  std::ofstream ofile (output_filename, std::ios::binary);

  if(!CGAL::write_xyz_points(ofile, pointswithindex,
                             CGAL::parameters::point_map(CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
                                     normal_map(CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>())))
  {
    std::cerr << "Error: cannot write file " << output_filename << std::endl;
  }

  std::cout<<"end of per atom"<<std::endl;

  // MESHING

  //handling command line options
  int c;
  stringstream f;
  string infile, outfile, input_radii;
  unsigned int depth = 7;
  double radius = -1;
  int radius_flag = 1;
  int infile_flag = 1;
  int outfile_flag = 1;
  std::list<double> radii;
  int parallel_flag = 1;

  infile = "pointandnormal.xyz";
  outfile = "pointandnormal.ply";
  radii.push_back(1.0);
  radii.sort();
  radius = radii.front();

  time_t start,end;

  Octree octree;

  std::time(&start);
  bool ok;
  if(radius >0)
  {
    ok = FileIO::readAndSortPoints(infile.c_str(),octree,radius);
  }
  else
  {
    octree.setDepth(depth);
    ok = FileIO::readAndSortPoints(infile.c_str(),octree);
  }
  if( !ok )
  {
    std::cerr<<"Pb opening the file; exiting."<<std::endl;
  }
  std::time(&end);

  std::cout<<"Octree with depth "<<octree.getDepth()<<" created."<<std::endl;
  std::cout<<"Octree contains "<<octree.getNpoints()
           <<" points. The bounding box size is "
           <<octree.getSize()<<std::endl;
  std::cout<<"Reading and sorting points in this octree took "
           <<difftime(end,start)<<" s."<<std::endl;
  std::cout<<"Octree statistics"<<std::endl;
  octree.printOctreeStat();


  std::cout<<"****** Reconstructing with radii "<<std::flush;

  std::list<double>::const_iterator ri = radii.begin();
  while(ri != radii.end())
  {
    std::cout<< *ri <<"; ";
    ++ri;
  }
  std::cout<<"******"<<std::endl;

  OctreeIterator iterator(&octree);

  if(radius>0)
    iterator.setR(radius);

  std::time(&start);

  Mesher mesher(&octree, &iterator);
  if(parallel_flag == 1){
    std::cout << "TRUE!!!!!" <<std::endl;
    mesher.parallelReconstruct(radii);}
  else{
    std::cout << "FALSE!!!!!" <<std::endl;
    mesher.reconstruct(radii);}
  std::time(&end);

  std::cout<<"Reconstructed mesh: "<<mesher.nVertices()
           <<" vertices; "<<mesher.nFacets()<<" facets. "<<std::endl;
  std::cout<<mesher.nBorderEdges()<<" border edges"<<std::endl;
  std::cout<<"Reconstructing the mesh took "<<difftime(end,start)
           <<"s."<<std::endl;

  std::cout<<"Filling the holes... "<<std::flush;

  std::time(&start);
  mesher.fillHoles();
  std::time(&end);

  std::cout<<difftime(end,start)<<" s."<<std::endl;
  std::cout<<"Final mesh: "<<mesher.nVertices()
           <<" vertices; "<<mesher.nFacets()<<" facets. "<<std::endl;
  std::cout<<mesher.nBorderEdges()<<" border edges"<<std::endl;

  if(! FileIO::saveMesh(outfile.c_str(), mesher))
  {
    std::cerr<<"Pb saving the mesh; exiting."<<std::endl;
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeParticlemesh::memory_usage()
{
  double bytes = double (nmax) * size_peratom_cols * sizeof(double);
  return bytes;
}
