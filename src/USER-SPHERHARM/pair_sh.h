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

#ifdef PAIR_CLASS

PairStyle(spherharm,PairSH)

#else

#ifndef LMP_PAIR_SH_H
#define LMP_PAIR_SH_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSH : public Pair {
 public:
  PairSH(class LAMMPS *);
  virtual ~PairSH();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);

 protected:
  double **cut{};

  class AtomVecSpherharm *avec{};

  virtual void allocate();

 private:

  // per-type coefficients, set in pair coeff command
  double ***normal_coeffs{};
  int *typetosh{};
  int matchtypes;
  double exponent;

  void matchtype();
  static void get_contact_quat(double (&)[3], double (&)[4]);

  double cur_time;
  int file_count;
  int write_surfpoints_to_file(double *x, bool append_file, int cont, int ifnorm, double *norm) const;
  int write_spherecentre_to_file(double *x, bool append_file, double rad) const;
  int write_ellipsoid(double *xi, double *xj, double irotmat[3][3], double jrotmat[3][3]) const;

  // Gaussian quadrature arrays
  double *abscissa{};          // Abscissa of gaussian quadrature (same for all shapes)
  double *weights{};            // Weights of gaussian quadrature (same for all shapes)
  int num_pole_quad;

  void get_quadrature_values(int num_quadrature);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
