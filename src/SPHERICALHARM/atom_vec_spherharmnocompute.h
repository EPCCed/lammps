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

AtomStyle(spherharmnocompute,AtomVecSpherharmnocompute)

#else

#ifndef LMP_ATOM_VEC_SPHERHARMNOCOMPUTE_H
#define LMP_ATOM_VEC_SPHERHARMNOCOMPUTE_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecSpherharmnocompute : public AtomVec {
 public:
  AtomVecSpherharmnocompute(class LAMMPS *);
  void process_args(int, char **);
  void init();

  void grow_pointers();
  void create_atom_post(int);
  void data_atom_post(int);
  void pack_data_pre(int);
  void pack_data_post(int);

  // JY Added
  ~AtomVecSpherharmnocompute();     // Destructor
  void read_coeffs(char *);// process an input file
  void read_coeffs();      // process all input
  double **shcoeff;        // spherical harmonic coefficients
  int maxshexpan;          // maximum expansion of the spherical harmonic series

 private:
  double *radius,*rmass;
  double **omega;

  int radvary;
  double radius_one,rmass_one;

  // JY Added
  void parse();                          // parse an input text line
  void reallocate(char *&, int &, int);  // reallocate a char string
  int numtriple(char *);                 // count number of triple quotes
  char *nextword(char *, char **);       // find next word in string with quotes

  int *shtype;                  // spherical harmonic type used
  int me;                       // proc ID
  int nfile,maxfile;            // current # and max # of open input files
  int curfile,curentry;         // current # and max # of open input files
  FILE **infiles;               // list of open input file
  int maxline,maxcopy;          // max lengths of char strings
  char *line,*copy,*work;       // input line & copy and work string
  int lnarg;                    // # of command args
  char **larg;                  // parsed args for command
  int maxarg;                   // max # of args in arg

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
