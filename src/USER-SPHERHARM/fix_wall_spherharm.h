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

#include "fix.h"

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

  void vol_based(double dx, double dy, double dz, double iang, int ishtype,
                 double *quat, double *x, double *f,
                 double *torque, double *contact);

 protected:
  int wallstyle,wiggle,wshear,axis;
  int pairstyle;
  bigint time_origin;
  double kn,mexpon;

  double lo,hi,cylradius;
  double amplitude,period,omega,vshear;
  double dt;
  char *idregion;

  class AtomVecSpherharm *avec{};

  // store particle interactions

  void clear_stored_contacts();

  void get_quadrature_values(int num_quadrature);
  void get_contact_quat(double (&xvecdist)[3], double (&quat)[4]);
  int refine_cap_angle(int &kk_count, int ishtype, double iang, double (&iquat_cont)[4],
                       double (&iquat_sf_bf)[4], const double xi[3], const double delvec[3]);
  void calc_force_torque(int kk_count, int ishtype, double iang,
                         double (&iquat_cont)[4], double (&iquat_sf_bf)[4],
                         const double xi[3], double (&irot)[3][3], double &vol_overlap,
                         double (&iforce)[3], double (&torsum)[3], double delvec[3]);
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
