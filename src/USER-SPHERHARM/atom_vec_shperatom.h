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
  int *shtype;                  // spherical harmonic type used
  double **angmom;
  double **quat;

  double **shcoeffs_bytype;
  double **pinertia_bytype;
  double **orient_bytype;
  int maxshexpan;
  int nfile,maxfile;            // current # and max # of open input files
  int curfile,curentry;         // current # and max # of open input files
  FILE **infiles;               // list of open input file
  int maxline,maxcopy;          // max lengths of char strings
  char *line,*copy,*work;       // input line & copy and work string
  int lnarg;                    // # of command args
  char **larg;                  // parsed args for command
  int maxarg;                   // max # of args in arg
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
  int me;                       // proc ID


  // FOR ELLIPSOID TEST ONLY
  double **ellipsoidshape;

  void get_quadrature_values();
  void getI();
  void calcexpansionfactors();
  void calcexpansionfactors_gauss();

  void read_coeffs(char *);// process an input file
  void read_coeffs();      // process all input
  void parse();                          // parse an input text line
  void reallocate(char *&, int &, int);  // reallocate a char string
  int numtriple(char *);                 // count number of triple quotes
  char *nextword(char *, char **);       // find next word in string with quotes

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Invalid radius in Atoms section of data file

Radius must be >= 0.0.

E: Invalid density in Atoms section of data file

Density value cannot be <= 0.0.

*/
