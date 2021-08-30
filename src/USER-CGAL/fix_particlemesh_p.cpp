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
   Contributing author: Kevin Hanley (Imperial)
------------------------------------------------------------------------- */

#include "fix_particlemesh_p.h"
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "memory.h"
#include "force.h"
#include "integrate.h"
#include "fix.h"

// CGAL HEADER FILES REQUIRED
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/vcm_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/property_map.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/tags.h>
#include <CGAL/IO/write_xyz_points.h>

// Ball rolling mesh header files
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
This is intended for use only with the granular rolling resistance model.
The purpose of this code is to store the angular velocities on the previous
timestep in a per-atom array so that they may be compared with the
current omega values in the rolling resistance model.

Values are stored using a post_force function; the omega values in the
post_force function are unchanged from the force computation as even if
damping is present, this affects only the torque, not omega (until the
final_integrate function is called) [KH - 24 October 2013]
 ---------------------------------------------------------------------- */

FixParticleMeshP::FixParticleMeshP(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        mesh(nullptr)
{
  restart_peratom = 1; //~ Per-atom information is saved to the restart file
  peratom_flag = 1;
  size_peratom_cols = 6; //~ omega x/y/z and positions of centroids
  peratom_freq = 1;
  create_attribute = 1;

  // perform initial allocation of atom-based array
  // register with Atom class
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  /*~ Initialise the values stored in mesh[*][*] to ridiculous
    values. The reason is that these crazy values can be easily 
    distinguished from physically reasonable values. In the setup
    function below, these crazy values are sought. If absent (due to
    the importation of data from a restart file), no need to 
    initialise. If present, then initialise at physically-reasonable
    values (using the initialised omega and x values which are now
    available)*/

  for (int i = 0; i < atom->nlocal; i++) {
    mesh[i][0] = mesh[i][1] = mesh[i][2] = 1.0e20;
    mesh[i][3] = mesh[i][4] = mesh[i][5] = 1.0e20;
  }
}

/* ---------------------------------------------------------------------- */

FixParticleMeshP::~FixParticleMeshP()
{
  // unregister callbacks to this fix from Atom class
  atom->delete_callback(id,Atom::GROW);
  atom->delete_callback(id,Atom::RESTART);

  // delete locally stored array
  memory->destroy(mesh);
}

/* ---------------------------------------------------------------------- */

int FixParticleMeshP::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixParticleMeshP::setup(int vflag)
{
  /*~ Now initialise everything for the first timestep if the data are
    not read in from a restart file. Initialise the oldomega values at
    the current omega values, and ditto for the x positions. This needs
    to be done in setup rather than init so that the ghosts are correct
    for the first timestep.*/
  int nall = atom->nlocal + atom->nghost;
  double **x = atom->x;
  double **omega = atom->omega;

  if (mesh[0][0] > 9.9e19 && mesh[0][1] > 9.9e19 &&
      mesh[0][2] > 9.9e19 && mesh[0][3] > 9.9e19 &&
      mesh[0][4] > 9.9e19 && mesh[0][5] > 9.9e19) {
    /*~ Since local atoms have smaller IDs than ghosts, checking only
      the first point is sufficient*/

    for (int i = 0; i < nall; i++) {
      mesh[i][0] = omega[i][0];
      mesh[i][1] = omega[i][1];
      mesh[i][2] = omega[i][2];
      mesh[i][3] = x[i][0]; //~ x position of centroid
      mesh[i][4] = x[i][1]; //~ y position
      mesh[i][5] = x[i][2]; //~ z position
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixParticleMeshP::pre_force(int vflag) {
  // vflag is protected?
  //  if (update->integrate->vflag <= 2) update->integrate->vflag += 4;

  int rank, size;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Datatype sendtype;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *mask = atom->mask;
  double **xrecv; // xrecv can't be a nullptr or else the MPI_GATHERV fails
  int *mrecv; // mrecv can't be a nullptr or else the MPI_GATHERV fails
  int num_atoms_per_proc[size];
  int displs[size];
  int num_tot_atoms = 0;
  int root = 0; // Need to see if LAMMPS has a default root
  unsigned int nb_neighbors_pca_normals = 18; // K-nearest neighbors = 3 rings (estimate normals by PCA)
  unsigned int nb_neighbors_mst = 60; // K-nearest neighbors (orient normals by MST)
  PointListwithindex pointswithindex;

  // root stores a vector containing the number of atoms on each processor (indexed by rank)
  // note that groups are ignored, we do this for all atoms in the domain
  MPI_Gather(&nlocal, 1, MPI_INT, num_atoms_per_proc, 1, MPI_INT, root, world);

  // Position vector, 3 blocks long with one element per block, no gap between data
  MPI_Type_vector(3, 1, 1, MPI_DOUBLE, &sendtype);
  MPI_Type_commit(&sendtype);

  // Require displacements as incoming data has different lengths and therefore the receive buffer has a
  // varying stride length
  if (rank == root) { //ok for recv to be uninitialised if rank!= root
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
      displs[i] = num_atoms_per_proc[i - 1] + displs[i - 1];
    }
    // Total number of atoms being sent into the receive buffer
    for (auto &n : num_atoms_per_proc) {
      num_tot_atoms += n;
    }
    // Creating the receive buffer
    // Need a contiguous block of memory for the MPU_Gatherv_call
    xrecv = (double **) calloc(num_tot_atoms,sizeof(double *));
    xrecv[0] = (double *) calloc(num_tot_atoms*3,sizeof(double));
    for (int i=1;i<num_tot_atoms;i++) {
      xrecv[i] = xrecv[0] + i*3;
    }
    mrecv = (int *) calloc(num_tot_atoms,sizeof(int));
//    memory->create(mrecv, num_tot_atoms, "FixParticleMeshP:mrecv");
  }

  // Gathering from the start of x[0][0] on each proc, where the send count is the number of atoms on each proc,
  // the send type is a vector with size 3, which is being sent to the receive buf with size tot number of atoms,
  // where the receive counts are set by the number of atoms on each proc.
  MPI_Gatherv(&(x[0][0]), nlocal, sendtype,
              &(xrecv[0][0]), num_atoms_per_proc, displs, sendtype,
              root, world);
//  MPI_Gatherv(&(x[0][0]), nlocal, sendtype,
//              &(xrecv[0][0]), num_atoms_per_proc, displs, sendtype,
//              root, world);
  MPI_Type_free(&sendtype);
//  // Gathering the atom masks too
  MPI_Gatherv(&(mask[0]), nlocal, MPI_INT,
              &(mrecv[0]), num_atoms_per_proc, displs, MPI_INT,
              root, world);

  if (rank==root){
//    for (int i=0; i<num_tot_atoms; i+=3){
//      std::cout << xrecv[i] << " "<< xrecv[i+1] << " "<< xrecv[i+2] << std::endl;
//    }
    for (int i=0; i<num_tot_atoms; i++){
//      std::cout << xrecv[i][0] << " "<< xrecv[i][1] << " "<< xrecv[i][2] << std::endl;
//      std::cout << xrecv[i][0] << " "<< xrecv[i][1] << " "<< xrecv[i][2] << " " <<mrecv[i]<< std::endl;
    }
//    memory->sfree(mrecv);
//    std::cout<<"avout to free"<<std::endl;
//    free(xrecv[0]);
//    std::cout<<"avout to free main"<<std::endl;
//    free(xrecv);
  }


  // Root proc sets the coords with correct mask into structure from which the normals will be calculated
  if (rank==root) {
    for (int i = 0; i < num_tot_atoms; i++) {
//      if (mrecv[i] & groupbit) {
        pointswithindex.emplace_back(i,
                                     Point3{xrecv[i][0],xrecv[i][1],xrecv[i][2]},
                                     Vector{0,0,0});
//      }
    }

    std::cout << "Prenormal"<<std::endl;
//
//    // Get normal
//    CGAL::pca_estimate_normals<Concurrency_tag>(pointswithindex,
//                                                nb_neighbors_pca_normals,
//                                                CGAL::parameters::point_map (CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
//                                                        normal_map (CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>()));
//    // Orient normals
//    PointListwithindex::iterator unoriented_points_begin =
//            CGAL::mst_orient_normals(pointswithindex,
//                                     nb_neighbors_mst,
//                                     CGAL::parameters::point_map (CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
//                                             normal_map(CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>()));
//
//    // Delete normals which can't be oriented
//    pointswithindex.erase(unoriented_points_begin, pointswithindex.end());
//
//    std::cout << "postnormal"<<std::endl;


////    // Write to file
////    std::string output_filename ("pointandnormal.xyz");
////    std::ofstream ofile (output_filename, std::ios::binary);
////    if(!CGAL::write_xyz_points(ofile, pointswithindex,
////                               CGAL::parameters::point_map(CGAL::Nth_of_tuple_property_map<1,PointVectorTuple>()).
////                                       normal_map(CGAL::Nth_of_tuple_property_map<2,PointVectorTuple>())))
////    {
////      std::cerr << "Error: cannot write file " << output_filename << std::endl;
////    }
//
//    memory->sfree(xrecv);
//    memory->sfree(mrecv);
//
//
//    // MESHING
//
//    //handling command line options
//    stringstream f;
//    string infile, outfile, input_radii;
//    unsigned int depth = 7;
//    double radius = -1;
//    std::list<double> radii;
//    int parallel_flag = 1;
//
//    infile = "pointandnormal.xyz";
//    outfile = "pointandnormal.ply";
//    radii.push_back(1.0);
//    radii.sort();
//    radius = radii.front();
//
//    time_t start,end;
//
//    Octree octree;
//
////    std::time(&start);
////    bool ok;
////    if(radius >0)
////    {
////      ok = FileIO::readAndSortPoints(infile.c_str(),octree,radius);
////    }
////    else
////    {
////      octree.setDepth(depth);
////      ok = FileIO::readAndSortPoints(infile.c_str(),octree);
////    }
////    if( !ok )
////    {
////      std::cerr<<"Pb opening the file; exiting."<<std::endl;
////    }
////    std::time(&end);
//
//    std::cout << "about to build octreee";
//
//    if(radius >0) {
//      octreefrompoints(pointswithindex, octree, radius);
//    }
//    else
//    {
//      octree.setDepth(depth);
//      octreefrompoints(pointswithindex, octree);
//    }
//
//    std::cout<<"Octree with depth "<<octree.getDepth()<<" created."<<std::endl;
//    std::cout<<"Octree contains "<<octree.getNpoints()
//             <<" points. The bounding box size is "
//             <<octree.getSize()<<std::endl;
//    std::cout<<"Reading and sorting points in this octree took "
//             <<difftime(end,start)<<" s."<<std::endl;
//    std::cout<<"Octree statistics"<<std::endl;
//    octree.printOctreeStat();
//
//
//    std::cout<<"****** Reconstructing with radii "<<std::flush;
//
//    std::list<double>::const_iterator ri = radii.begin();
//    while(ri != radii.end())
//    {
//      std::cout<< *ri <<"; ";
//      ++ri;
//    }
//    std::cout<<"******"<<std::endl;
//
//    OctreeIterator iterator(&octree);
//
//    if(radius>0)
//      iterator.setR(radius);
//
//    std::time(&start);
//
//    Mesher mesher(&octree, &iterator);
//    if(parallel_flag == 1){
//      mesher.parallelReconstruct(radii);}
//    else{
//      mesher.reconstruct(radii);}
//    std::time(&end);
//
//    std::cout<<"Reconstructed mesh: "<<mesher.nVertices()
//             <<" vertices; "<<mesher.nFacets()<<" facets. "<<std::endl;
//    std::cout<<mesher.nBorderEdges()<<" border edges"<<std::endl;
//    std::cout<<"Reconstructing the mesh took "<<difftime(end,start)
//             <<"s."<<std::endl;
//
//    std::cout<<"Filling the holes... "<<std::flush;
//
//    std::time(&start);
//    mesher.fillHoles();
//    std::time(&end);
//
//    std::cout<<difftime(end,start)<<" s."<<std::endl;
//    std::cout<<"Final mesh: "<<mesher.nVertices()
//             <<" vertices; "<<mesher.nFacets()<<" facets. "<<std::endl;
//    std::cout<<mesher.nBorderEdges()<<" border edges"<<std::endl;
//
//    if(! FileIO::saveMesh(outfile.c_str(), mesher))
//    {
//      std::cerr<<"Pb saving the mesh; exiting."<<std::endl;
//    }
//
//
  }
  MPI_Barrier(world);
}

/* ---------------------------------------------------------------------- */

void FixParticleMeshP::post_force(int vflag)
{
  int nall = atom->nlocal + atom->nghost; //~ Include ghosts
  double **x = atom->x;
  double **omega = atom->omega;

  for (int i = 0; i < nall; i++) {
    mesh[i][0] = omega[i][0];
    mesh[i][1] = omega[i][1];
    mesh[i][2] = omega[i][2];
    mesh[i][3] = x[i][0]; //~ x position of centroid
    mesh[i][4] = x[i][1];
    mesh[i][5] = x[i][2];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixParticleMeshP::memory_usage()
{
  double bytes = atom->nmax*6 * sizeof(double); //~ For mesh array
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixParticleMeshP::grow_arrays(int nmax)
{
  memory->grow(mesh,nmax,6,"old_omega:mesh");
  array_atom = mesh;
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixParticleMeshP::copy_arrays(int i, int j, int delflag)
{
  for (int q = 0; q < 6; q++)
    mesh[j][q] = mesh[i][q];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixParticleMeshP::set_arrays(int i)
{
  double **x = atom->x;
  double **omega = atom->omega;
  mesh[i][0] = omega[i][0];
  mesh[i][1] = omega[i][1];
  mesh[i][2] = omega[i][2];
  mesh[i][3] = x[i][0];
  mesh[i][4] = x[i][1];
  mesh[i][5] = x[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixParticleMeshP::pack_exchange(int i, double *buf)
{
  for (int q = 0; q < 6; q++)
    buf[q] = mesh[i][q];

  return 6;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixParticleMeshP::unpack_exchange(int nlocal, double *buf)
{
  for (int q = 0; q < 6; q++)
    mesh[nlocal][q] = buf[q];

  return 6;
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for restart file
------------------------------------------------------------------------- */

int FixParticleMeshP::pack_restart(int i, double *buf)
{
  buf[0] = 7;
  for (int q = 0; q < 6; q++)
    buf[q+1] = mesh[i][q];

  return 7;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixParticleMeshP::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values

  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  for (int q = 0; q < 6; q++)
    mesh[nlocal][q] = extra[nlocal][m++];
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixParticleMeshP::maxsize_restart()
{
  return 7;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixParticleMeshP::size_restart(int nlocal)
{
  return 7;
}

/* ---------------------------------------------------------------------- */

//void FixParticleMeshP::octreefrompoints(PointListwithindex points, Octree& octree,double min_radius)
//{
//  list<Vertex> input_vertices;
//  double xmin, ymin, zmin, xmax, ymax, zmax;
//  double x, y, z;
//  xmin = xmax = std::get<1>(points[0])[0];
//  ymin = ymax = std::get<1>(points[0])[1];
//  zmin = zmax = std::get<1>(points[0])[2];
//  for (auto & el : points) {
//    x=std::get<1>(el)[0];
//    y=std::get<1>(el)[1];
//    z=std::get<1>(el)[2];
//    input_vertices.push_back(Vertex(x,y,z,
//                                    std::get<2>(el)[0],
//                                    std::get<2>(el)[1],
//                                    std::get<2>(el)[2]));
//    xmin = x < xmin ? x : xmin;
//    xmax = x > xmax ? x : xmax;
//    ymin = y < ymin ? y : ymin;
//    ymax = y > ymax ? y : ymax;
//    zmin = z < zmin ? z : zmin;
//    zmax = z > zmax ? z : zmax;
//  }
//  std::cout<<input_vertices.size()<<" points read"<<std::endl;
//  double lx = xmax - xmin;
//  double ly = ymax - ymin;
//  double lz = zmax - zmin;
//  double size = lx > ly ? lx : ly;
//  size = size > lz ? size : lz;
//  size = 1.1 * size;
//  double margin;
//  if(min_radius > 0)
//  {
//    unsigned int depth = (unsigned int)ceil( log2( size / (min_radius) ));
//    double adapted_size = pow2(depth) * min_radius;
//    margin = 0.5 * (adapted_size - size);
//    size = adapted_size;
//    octree.setDepth(depth);
//  }
//  else
//  {
//    margin = 0.05 * size;
//  }
//  double ox = xmin - margin;
//  double oy = ymin - margin;
//  double oz = zmin - margin;
//  Point origin(ox,oy,oz);
//  octree.initialize(origin, size);
//  octree.addPoints(input_vertices.begin(), input_vertices.end());
//}