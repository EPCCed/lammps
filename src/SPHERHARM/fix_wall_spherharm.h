/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(wall/spherharm,FixWallSpherharm)

#else

#ifndef LMP_FIX_WALL_SPHERHARM_H
#define LMP_FIX_WALL_SPHERHARM_H

#include "math_extra.h"
#include "fix.h"
#include "math_spherharm.h"

namespace LAMMPS_NS {

class FixWallSpherharm : public Fix {
 public:
  FixWallSpherharm(class LAMMPS *, int, char **);
  virtual ~FixWallSpherharm();
  int setmask();
  virtual void init();
  void setup(int);
  virtual void post_force(int);
//  virtual void post_force_respa(int, int, int);

  virtual double memory_usage();
  virtual void grow_arrays(int);
  virtual void copy_arrays(int, int, int);
  virtual void set_arrays(int);
  virtual int pack_exchange(int, double *);
  virtual int unpack_exchange(int, double *);
//  virtual int pack_restart(int, double *);
//  virtual void unpack_restart(int, int);
//  virtual int size_restart(int);
//  virtual int maxsize_restart();
  void reset_dt();

 protected:
  int wallstyle,wiggle,wshear,axis;
  int pairstyle;
  bigint time_origin;
  double kn,mexpon,tangcoeff;
  bool tang;

  double lo,hi,cylradius;
  double amplitude,period,omega,vshear;
  double dt;
  char *idregion;

  class AtomVecSpherharm *avec{};

  // Creating a vector of functions, should be C++11 compliant, can't use auto lambdas.
  // https://stackoverflow.com/questions/30268507/in-c-how-to-choose-to-run-a-specific-member-function-without-using-if-stateme
  // https://stackoverflow.com/questions/7582546/using-generic-stdfunction-objects-with-member-functions-in-one-class
  std::vector<std::function<int(const double[3], const double(&)[3], const double(&)[3], double &, const double)>>
  wall_fns =
  {[this](const double ix[3], const double (&wall_normal)[3], const double (&unit_line_normal)[3],
          double &rad, const double numer)-> int{
      double denom;
      denom = MathExtra::dot3(unit_line_normal, wall_normal);
      if (denom==0.0) return 1; // wall and line normal are parallel
      rad = numer/denom;
      return 0;
  },
   [this](const double ix[3], const double (&wall_normal)[3], const double (&unit_line_normal)[3],
          double &rad, const double numer)-> int{
       double t1, t2;
       int not_ok;
       not_ok = MathSpherharm::line_cylinder_intersection(ix, unit_line_normal,t1,t2, cylradius);
       rad = t1*t2 > 0.0 ? std::min(t1,t2) : std::max(t1,t2);
       return(not_ok);
   }
  };

  // store particle interactions
  void clear_stored_contacts();
  void get_quadrature_values(int num_quadrature);
  void vol_based(double dx, double dy, double dz, double iang, int ishtype,
                 double *quat, double *x, double *f,
                 double *torque, double *v, double *omega, double *contact, double *maxrad, double (&vwall)[3]);
  int refine_cap_angle_plane(int &kk_count, int ishtype, double iang, double (&iquat_cont)[4],
                       double (&iquat_sf_bf)[4], const double xi[3], const double delvec[3]);
  int refine_cap_angle_cylinder(int &kk_count, int ishtype, double iang, double (&iquat_cont)[4],
                             double (&iquat_sf_bf)[4], const double xi[3], const double delvec[3]);
  void calc_velCoulomb_force_torque(int ishtype, double const (&normforce)[3],
                                    double const (&vr)[3], const double *omegaa,
                                    double const (&cp)[3], double const xi[3], double (&iquat_sf_bf)[4],
                                    double (&tforce)[3], double (&ttorque)[3]);
  void calc_force_torque(int wall_type, int ishtype, double iang, double (&iquat_cont)[4],
                          double (&iquat_sf_bf)[4], const double xi[3], double (&irot)[3][3],
                          double &vol_overlap, double (&iforce)[3], double (&torsum)[3],
                          double delvec[3]);


  // Gaussian quadrature arrays
  double *abscissa{};          // Abscissa of gaussian quadrature (same for all shapes)
  double *weights{};           // Weights of gaussian quadrature (same for all shapes)
  int num_pole_quad;

  static void write_surfpoints_to_file(double *x, int cont, double *norm, int file_count, bool first_call);
  };

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix wall/gran requires atom style sphere

Self-explanatory.

E: Invalid fix wall/gran interaction style

UNDOCUMENTED

E: Cannot use wall in periodic dimension

Self-explanatory.

E: Cannot wiggle and shear fix wall/gran

Cannot specify both options at the same time.

E: Invalid wiggle direction for fix wall/gran

Self-explanatory.

E: Invalid shear direction for fix wall/gran

Self-explanatory.

E: Cannot wiggle or shear with fix wall/gran/region

UNDOCUMENTED

U: Fix wall/gran is incompatible with Pair style

Must use a granular pair style to define the parameters needed for
this fix.

*/
