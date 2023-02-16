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

#include "fix_particlemesh_s.h"
#include <cstdlib>
#include "atom.h"
#include "update.h"
#include "memory.h"
#include "force.h"
#include "integrate.h"
#include "fix.h"
#include "error.h"
#include <math_extra.h>

// CGAL HEADER FILES REQUIRED
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/property_map.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/tags.h>
#include <CGAL/IO/write_xyz_points.h>

// Ball rolling vertexdata header files
#include "Octree.h"
#include "OctreeIterator.h"
#include "Mesher.h"
#include "FileIO.h"
#include "types.h"
#include <ctime>
#include <vector>

// TO DELETE
#include "iostream"

// CGAL TYPEDEFS
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point3;
typedef Kernel::Vector_3 Vector;
typedef std::tuple<int, Point3, Vector> PointVectorTuple;
typedef std::vector<PointVectorTuple> PointListwithindex;
typedef CGAL::Parallel_if_available_tag Concurrency_tag;

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
 ---------------------------------------------------------------------- */

FixParticleMeshS::FixParticleMeshS(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        vertexdata(nullptr)
{
  restart_peratom = 1; //~ Per-atom information is saved to the restart file
  peratom_flag = 1;
  size_peratom_cols = 4; //~ normal x/y/z and area
  peratom_freq = 1; // every step, **TODO change to user input utils::inumeric(FLERR,arg[5],false,lmp);
  create_attribute = 1; //fix stores attributes that need setting when a new atom is created

  int size;
  MPI_Comm_size(world, &size);
  if (size!=1) error->all(FLERR,"fix particle_meshs: requires serial execution");
  if (narg < 4) error->all(FLERR, "llegal pair_style particle_meshs command");

  // Read in the radii for the rolling ball meshing algorithm
  for (int i = 3; i < narg; i++) {
    radii.push_back(utils::numeric(FLERR, arg[i], true, lmp));
  }
  std::cout << std::endl;


  // perform initial allocation of atom-based array
  // register with Atom class
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  // Set initial values to zero
  for (int i = 0; i < atom->nlocal; i++) {
    vertexdata[i][0] = vertexdata[i][1] = vertexdata[i][2] = 0.0;
    vertexdata[i][3] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

FixParticleMeshS::~FixParticleMeshS()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,Atom::GROW);
  atom->delete_callback(id,Atom::RESTART);

  // delete locally stored array
  memory->destroy(vertexdata);
}

/* ---------------------------------------------------------------------- */

int FixParticleMeshS::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixParticleMeshS::pre_force(int vflag) {
  // vflag is protected?
  //  if (update->integrate->vflag <= 2) update->integrate->vflag += 4;

  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *mask = atom->mask;
  unsigned int nb_neighbors_pca_normals = 18; // K-nearest neighbors = 3 rings (estimate normals by PCA)
  unsigned int nb_neighbors_mst = 60; // K-nearest neighbors (orient normals by MST)
  PointListwithindex pointswithindex;
  stringstream f;
  string infile, outfile, input_radii;
  double radius;
  int parallel_flag = 1;
  time_t start,end;

  // Store all atom coords into the pointswithindex vector of tupples. Required for CGAL to estimate and orient normals
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      pointswithindex.emplace_back(i,
                                   Point3{x[i][0], x[i][1], x[i][2]},
                                   Vector{0, 0, 0});
    }
  }
  // Estimate normals (normals stored in the 3rd element of the tupple pointswithindex)
  CGAL::pca_estimate_normals<Concurrency_tag>(pointswithindex,
                                              nb_neighbors_pca_normals,
                                              CGAL::parameters::point_map (CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
                                                      normal_map (CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>()));
  /*
  Orient normals (irritating function reorganises the points, hence requiring the index as the first element of the
  tupple. Returns an iterator which points to the first of the points that couldn't be oriented).
  */
  PointListwithindex::iterator unoriented_points_begin =
          CGAL::mst_orient_normals(pointswithindex,
                                   nb_neighbors_mst,
                                   CGAL::parameters::point_map (CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
                                           normal_map(CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>()));
  // Zero the normal for the atom if it's normal couldn't be oriented
  for (auto point = unoriented_points_begin; point!=pointswithindex.end(); ++point) {
    int ind = std::get<0>(*point);
    std::cout << ind << std::endl;
    vertexdata[ind][0] = 0.0;
    vertexdata[ind][1] = 0.0;
    vertexdata[ind][2] = 0.0;
    vertexdata[ind][3] = 0.0; // set areas to zero too
  }
  // Storing the normals in the first three elements of vertexdata so that they can be output by the fix 
  for (auto & point : pointswithindex) {
    int ind = std::get<0>(point);
    vertexdata[ind][0] = std::get<2>(point)[0];
    vertexdata[ind][1] = std::get<2>(point)[1];
    vertexdata[ind][2] = std::get<2>(point)[2];
    vertexdata[ind][3] = 0.0; // set areas to zero too
  }
  // Delete normals which can't be oriented. Otherwise the the meshing might fail
  pointswithindex.erase(unoriented_points_begin, pointswithindex.end());

  // Meshing Section
  outfile = "pointandnormal.ply";
  radii.sort();
  radius = radii.front();

  Octree octree;
  octreefrompoints(pointswithindex, octree, radius);

  std::cout<<"Octree with depth "<<octree.getDepth()<<" created."<<std::endl;
  std::cout<<"Octree contains "<<octree.getNpoints()
           <<" points. The bounding box size is "
           <<octree.getSize()<<std::endl;
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
    mesher.parallelReconstruct(radii);}
  else{
    mesher.reconstruct(radii);}
  std::time(&end);

  std::cout<<"Reconstructed vertexdata: "<<mesher.nVertices()
           <<" vertices; "<<mesher.nFacets()<<" facets. "<<std::endl;
  std::cout<<mesher.nBorderEdges()<<" border edges"<<std::endl;
  std::cout<<"Reconstructing the vertexdata took "<<difftime(end,start)
           <<"s."<<std::endl;

  std::cout<<"Filling the holes... "<<std::flush;

  std::time(&start);
  mesher.fillHoles();
  std::time(&end);

  std::cout<<difftime(end,start)<<" s."<<std::endl;
  std::cout<<"Final vertexdata: "<<mesher.nVertices()
           <<" vertices; "<<mesher.nFacets()<<" facets. "<<std::endl;
  std::cout<<mesher.nBorderEdges()<<" border edges"<<std::endl;

  if(! FileIO::saveMesh(outfile.c_str(), mesher))
  {
    std::cerr<<"Pb saving the vertexdata; exiting."<<std::endl;
  }

  areasfrommesh(mesher, pointswithindex, mesher.nVertices());

  //------------
  // Debugging code
  //------------
  double total_area=0.0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      total_area += vertexdata[i][3];
    }
  }
  std::cout << std::endl << "Total area is " << total_area << std::endl << std::endl;
  //------------


}

/* ---------------------------------------------------------------------- */

void FixParticleMeshS::post_force(int vflag)
{
  /*
   * This is where we could put Enzo's code for displacing the atoms
   * For now, just leave blank
   */
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixParticleMeshS::memory_usage()
{
  double bytes = atom->nmax*size_peratom_cols * sizeof(double); //~ For vertexdata array
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixParticleMeshS::grow_arrays(int nmax)
{
  memory->grow(vertexdata,nmax,size_peratom_cols,"particle_meshs:vertexdata");
  array_atom = vertexdata;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixParticleMeshS::copy_arrays(int i, int j, int delflag)
{
  for (int q = 0; q < size_peratom_cols; q++)
    vertexdata[j][q] = vertexdata[i][q];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixParticleMeshS::set_arrays(int i)
{
  vertexdata[i][0] = 0.0;
  vertexdata[i][1] = 0.0;
  vertexdata[i][2] = 0.0;
  vertexdata[i][3] = 0.0;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixParticleMeshS::pack_exchange(int i, double *buf)
{
  for (int q = 0; q < size_peratom_cols; q++)
    buf[q] = vertexdata[i][q];

  return size_peratom_cols;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixParticleMeshS::unpack_exchange(int nlocal, double *buf)
{
  for (int q = 0; q < size_peratom_cols; q++)
    vertexdata[nlocal][q] = buf[q];

  return size_peratom_cols;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixParticleMeshS::pack_restart(int i, double *buf)
{
  buf[0] = size_peratom_cols+1;
  for (int q = 0; q < size_peratom_cols; q++)
    buf[q+1] = vertexdata[i][q];

  return size_peratom_cols+1;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixParticleMeshS::unpack_restart(int nlocal, int nth)
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

int FixParticleMeshS::maxsize_restart()
{
  return size_peratom_cols+1;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixParticleMeshS::size_restart(int nlocal)
{
  return size_peratom_cols+1;
}

/* ---------------------------------------------------------------------- */

void FixParticleMeshS::octreefrompoints(PointListwithindex points, Octree& octree,double min_radius)
{
  list<Vertex> input_vertices;
  double xmin, ymin, zmin, xmax, ymax, zmax;
  double x, y, z;
  xmin = xmax = std::get<1>(points[0])[0];
  ymin = ymax = std::get<1>(points[0])[1];
  zmin = zmax = std::get<1>(points[0])[2];
  for (auto & el : points) {
    x=std::get<1>(el)[0];
    y=std::get<1>(el)[1];
    z=std::get<1>(el)[2];
    input_vertices.emplace_back(x,y,z,
                                    std::get<2>(el)[0],
                                    std::get<2>(el)[1],
                                    std::get<2>(el)[2]);
    xmin = x < xmin ? x : xmin;
    xmax = x > xmax ? x : xmax;
    ymin = y < ymin ? y : ymin;
    ymax = y > ymax ? y : ymax;
    zmin = z < zmin ? z : zmin;
    zmax = z > zmax ? z : zmax;
  }
  std::cout<<input_vertices.size()<<" points read"<<std::endl;
  double lx = xmax - xmin;
  double ly = ymax - ymin;
  double lz = zmax - zmin;
  double size = lx > ly ? lx : ly;
  size = size > lz ? size : lz;
  size = 1.1 * size;
  double margin;
  if(min_radius > 0)
  {
    unsigned int depth = (unsigned int)ceil( log2( size / (min_radius) ));
    double adapted_size = pow2(depth) * min_radius;
    margin = 0.5 * (adapted_size - size);
    size = adapted_size;
    octree.setDepth(depth);
  }
  else
  {
    margin = 0.05 * size;
  }
  double ox = xmin - margin;
  double oy = ymin - margin;
  double oz = zmin - margin;
  Point origin(ox,oy,oz);
  octree.initialize(origin, size);
  octree.addPoints(input_vertices.begin(), input_vertices.end());
}

void FixParticleMeshS::areasfrommesh(Mesher& mesher, PointListwithindex& points, int numvertices) {

  double c[3], se1[3], se2[3];
  double se3[3], dots[3], abs[3];
  double st[3];
  int vinds[3];
  const double eps = 1e-10; // tolerance for matching the vertices in the mesh to the atom coords
  double area;
  int index;
  std::vector<int> vertexmap(numvertices, -1); // mapping the vertex in the mesh back to the atom indexes
  int vertcount=0;

  // Go through all the faces in the vertexdata
  for (Facet_star_list::const_iterator fi = mesher.facets_begin();
       fi != mesher.facets_end(); ++fi)
  {
    const Facet *f = *fi;
    Vertex *v0 = f->vertex(0);
    Vertex *v1 = f->vertex(1);
    Vertex *v2 = f->vertex(2);
    /*
    Get the indexes for mapping the areas back to the atoms
    This is horrible horrible code that shouldn't not exist, but I don't know how to get the
    rolling-ball algorithm to preserve the indexes.
    */
    for (int i=0; i<3; i++)
    {
      Vertex *v3 = f->vertex(i);
      index = v3->index();
      if (vertexmap[index]<0) { // only find if vertex not mapped (speeds up horrible search ~x6)
        auto it = std::find_if(points.begin() + vertcount, points.end(),
                               [=](const PointVectorTuple &e) {
                                   return (std::abs(std::get<1>(e)[0] - v3->x()) < eps and
                                           std::abs(std::get<1>(e)[1] - v3->y()) < eps and
                                           std::abs(std::get<1>(e)[2] - v3->z()) < eps);
                               });
        if (it != points.end()) {
          vinds[i] = get<0>(*it);
          vertexmap[index] = vinds[i]; // map vertex and store appropriate index
          /* Swapping vertices such that those that have been successfully mapped are put to the start of the stl.
           * These vertices are then no longer searched through as the find_if function starts at
           * points.begin() + vertcount. This gives a minor speed bump. */
          std::iter_swap(points.begin() + vertcount, it);
          vertcount++;
        }
        else
        {
          error->all(FLERR, "fix particle_meshs: can't match vertex to pointslist");
        }
      }
      else vinds[i] = vertexmap[index]; // vertex already mapped
    }

    // Centroid of the vertices making the current triangle
    c[0] = (v0->x() + v1->x() + v2->x())/3.0;
    c[1] = (v0->y() + v1->y() + v2->y())/3.0;
    c[2] = (v0->z() + v1->z() + v2->z())/3.0;

    // Sub-edge 1
    se1[0] = v0->x() - c[0];
    se1[1] = v0->y() - c[1];
    se1[2] = v0->z() - c[2];

    // Sub-edge 2
    se2[0] = v1->x() - c[0];
    se2[1] = v1->y() - c[1];
    se2[2] = v1->z() - c[2];

    // Sub-edge 3
    se3[0] = v2->x() - c[0];
    se3[1] = v2->y() - c[1];
    se3[2] = v2->z() - c[2];

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
    area = 0.25 * abs[0] * abs[1] * st[0];
    vertexdata[vinds[0]][3] += area;
    vertexdata[vinds[1]][3] += area;
    area = 0.25 * abs[1] * abs[2] * st[1];
    vertexdata[vinds[1]][3] += area;
    vertexdata[vinds[2]][3] += area;
    area = 0.25 * abs[2] * abs[0] * st[2];
    vertexdata[vinds[2]][3] += area;
    vertexdata[vinds[0]][3] += area;
  }
}