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

#ifdef ATOM_CLASS

AtomStyle(shperatom,AtomVecShperatom)

#else

#ifndef LMP_ATOM_VEC_SHPERATOM_H
#define LMP_ATOM_VEC_SHPERATOM_H

#include "atom_vec.h"
#include <cmath>
#include <string>

namespace LAMMPS_NS {

class AtomVecShperatom : public AtomVec {
 public:
  AtomVecShperatom(class LAMMPS *);
  void process_args(int, char **);
  void init();

  void grow_pointers();
  void create_atom_post(int);
  void data_atom_post(int);
  void pack_data_pre(int);
  void pack_data_post(int);

  ~AtomVecShperatom();     // Destructor

  double** get_quadrature_rads(int &in_num_quad2){
    in_num_quad2 = num_quad2;
    return &quad_rads[0];
  }
  double** get_quadrature_angs(){return &angles[0];}
  double* get_max_rads(){return &maxrad[0];}
  double** get_quat_init(){return &orient_bytype[0];}
  double** get_pinertia_init(){return &pinertia_bytype[0];}
  int check_contact(int, double, double, double, double &);

  // FOR ELLIPSOID TEST ONLY
  void get_shape(int, double &, double &, double &);

 private:

  double *radius,*rmass;
  double **omega;
  int *shtype;
  double **angmom;
  double **quat;

  double **shcoeffs_bytype;
  double **pinertia_bytype;
  double **orient_bytype;
  int maxshexpan;
  double **angles;
  double *weights;
  double **quad_rads;
  int nshtypes;
  int num_quadrature;
  int num_quad2;
  int numcoeffs;
  double **expfacts;
  double *maxrad;

  int radvary;
  double radius_one,rmass_one;
  int me;

  // FOR ELLIPSOID TEST ONLY
  double **ellipsoidshape;

  void get_quadrature_values();
  void getI();
  void calcexpansionfactors();
  void calcexpansionfactors_gauss();
  void read_sh_coeffs(char *, int);

};

}

#endif
#endif
