/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributead under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom_vec_spherharmnocompute.h"
#include <cstring>
#include <iostream>
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "fix_adapt.h"
#include "math_const.h"
#include "error.h"
#include "utils.h"
#include "memory.h"

#include "fmt/format.h"

using namespace LAMMPS_NS;
using namespace MathConst;

//JY ADDED
#define DELTALINE 256
#define DELTA 4
/* ---------------------------------------------------------------------- */

AtomVecSpherharmnocompute::AtomVecSpherharmnocompute(LAMMPS *lmp) : AtomVec(lmp)
{
  //  JY ADDED
  MPI_Comm_rank(world,&me);
  maxline = maxcopy = 0;
  line = copy = work = NULL;
  if (me == 0) {
    nfile = 1;
    maxfile = 16;
    infiles = new FILE *[maxfile];
    infiles[0] = infile;
  } else infiles = NULL;
  lnarg = maxarg = 0;
  larg = NULL;
  curfile = curentry = 0;
  maxshexpan = 30;
  spherharm_type = 1;

  mass_type = 0;
  molecular = 0;

  atom->sphere_flag = 1;
  atom->radius_flag = atom->rmass_flag = atom->omega_flag =
    atom->torque_flag = 1;

  // strings with peratom variables to include in each AtomVec method
  // strings cannot contain fields in corresponding AtomVec default strings
  // order of fields in a string does not matter
  // except: fields_data_atom & fields_data_vel must match data file

  fields_grow = (char *) "radius rmass omega torque shtype";
  fields_copy = (char *) "radius rmass omega shtype";
  fields_comm = (char *) "";
  fields_comm_vel = (char *) "omega";
  fields_reverse = (char *) "torque";
  fields_border = (char *) "radius rmass shtype";
  fields_border_vel = (char *) "radius rmass omega shtype";
  fields_exchange = (char *) "radius rmass omega shtype";
  fields_restart = (char *) "radius rmass omega shtype";
  fields_create = (char *) "radius rmass omega shtype";
  fields_data_atom = (char *) "id type radius rmass x shtype";
  fields_data_vel = (char *) "id v omega";
}

/* ---------------------------------------------------------------------- */

AtomVecSpherharmnocompute::~AtomVecSpherharmnocompute()
{
//  memory->destroy(shcoeff);
  memory->sfree(line);
  memory->sfree(copy);
  memory->sfree(work);
  memory->sfree(larg);
  delete [] infiles;
}


/* ----------------------------------------------------------------------
   process sub-style args
   optional arg = 0/1 for static/dynamic particle radii
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::process_args(int narg, char **arg) {

  if (narg < 1)
    error->all(FLERR, "llegal atom_style atom_style spherharm command");

  radvary = 0;
  radvary = utils::numeric(FLERR, arg[0], true, lmp);
  if (radvary < 0 || radvary > 1)
    error->all(FLERR, "Illegal atom_style sphere command");

  // dynamic particle radius and mass must be communicated every step
  if (radvary) {
    fields_comm = (char *) "radius rmass";
    fields_comm_vel = (char *) "radius rmass omega";
  }

  int nshtypes = narg - 1;
  atom->nshtypes = nshtypes;
  atom->allocate_type_sh_arrays(maxshexpan);
  shcoeff = atom->shcoeff;
  //  memory->create(shcoeff, nshtypes, 2 * (maxshexpan + 1) * (maxshexpan + 1), "AtomVecSpherharmnocompute:shcoeff");
  for (int i = 1; i < narg; i++) {
    read_coeffs(arg[i]);
    curfile++;
  }
  curfile--;
  MPI_Bcast(&(shcoeff[0][0]), nshtypes * 2 * (maxshexpan + 1) * (maxshexpan + 1), MPI_DOUBLE, 0, world);

//  std::cout << me << std::endl;
//  if (me != 0) {
//    for (int i = 0; i < nshtypes; i++) {
//      int count = 1;
//      for (int j = 0; j < 2 * (maxshexpan + 1) * (maxshexpan + 1); j = j + 2) {
//        std::cout << i << " " << ++count << " " << shcoeff[i][j] << " " << shcoeff[i][j + 1] << std::endl;
//      }
//    }
//  }

  // delay setting up of fields until now
  setup_fields();
}

/* ---------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::init()
{
  AtomVec::init();

  // check if optional radvary setting should have been set to 1

  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"adapt") == 0) {
      FixAdapt *fix = (FixAdapt *) modify->fix[i];
      if (fix->diamflag && radvary == 0)
        error->all(FLERR,"Fix adapt changes particle radii "
                   "but atom_style sphere is not dynamic");
    }
}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::grow_pointers()
{
  radius = atom->radius;
  rmass = atom->rmass;
  omega = atom->omega;
  shtype = atom->shtype;
}

/* ----------------------------------------------------------------------
   initialize non-zero atom quantities
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::create_atom_post(int ilocal)
{
  radius[ilocal] = 0.5;
  rmass[ilocal] = 4.0*MY_PI/3.0 * 0.5*0.5*0.5;
  shtype[ilocal] = 0;
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::data_atom_post(int ilocal)
{
  radius_one = 0.5 * atom->radius[ilocal];
  radius[ilocal] = radius_one;
  if (radius_one > 0.0)
    rmass[ilocal] *= 4.0*MY_PI/3.0 * radius_one*radius_one*radius_one;

  if (rmass[ilocal] <= 0.0)
    error->one(FLERR,"Invalid density in Atoms section of data file");

  omega[ilocal][0] = 0.0;
  omega[ilocal][1] = 0.0;
  omega[ilocal][2] = 0.0;
}

/* ----------------------------------------------------------------------
   modify values for AtomVec::pack_data() to pack
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::pack_data_pre(int ilocal)
{
  radius_one = radius[ilocal];
  rmass_one = rmass[ilocal];

  radius[ilocal] *= 2.0;
  if (radius_one!= 0.0)
    rmass[ilocal] =
      rmass_one / (4.0*MY_PI/3.0 * radius_one*radius_one*radius_one);
}

/* ----------------------------------------------------------------------
   unmodify values packed by AtomVec::pack_data()
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::pack_data_post(int ilocal)
{
  radius[ilocal] = radius_one;
  rmass[ilocal] = rmass_one;
}

/* ---------------------------------------------------------------------- */
void AtomVecSpherharmnocompute::read_coeffs(char *filename)
{

//  std::cout << filename << std::endl;

  // error if another nested file still open, should not be possible
  // open new filename and set infile, infiles[0], nfile
  // call to file() will close filename and decrement nfile
  if (me == 0) {
    if (nfile == maxfile)
      error->one(FLERR, "Too many nested levels of input scripts");

    infile = fopen(filename, "r");
    if (infile == NULL)
      error->one(FLERR, fmt::format("Cannot open input script {}: {}",
                                    filename, utils::getsyserror()));
    infiles[nfile++] = infile;

    // process contents of file
    curentry = 0;
    read_coeffs();

    fclose(infile);
    nfile--;
    infile = infiles[nfile - 1];
  }
}

/* ---------------------------------------------------------------------- */
void AtomVecSpherharmnocompute::read_coeffs()
{
  int m,n;
  int max_exp;

  while (1) {

    // read a line from input script
    // n = length of line including str terminator, 0 if end of file
    // if line ends in continuation char '&', concatenate next line

    m = 0;
    while (1) {
      if (maxline-m < 2) reallocate(line,maxline,0);

      // end of file reached, so break
      // n == 0 if nothing read, else n = line with str terminator

      if (fgets(&line[m],maxline-m,infile) == NULL) {
        if (m) n = strlen(line) + 1;
        else n = 0;
        break;
      }

      // continue if last char read was not a newline
      // could happen if line is very long

      m = strlen(line);
      if (line[m-1] != '\n') continue;

      // continue reading if final printable char is & char
      // or if odd number of triple quotes
      // else break with n = line with str terminator

      m--;
      while (m >= 0 && isspace(line[m])) m--;
      if (m < 0 || line[m] != '&') {
        if (numtriple(line) % 2) {
          m += 2;
          continue;
        }
        line[m+1] = '\0';
        n = m+2;
        break;
      }
    }


    // if n = 0, end-of-file
    // if original input file, code is done
    // else go back to previous input file

    if (n == 0) break;

    if (n > maxline) reallocate(line,maxline,n);

    // echo the command unless scanning for label
//    if (me == 0) {
//      std::cout << "Line" << std::endl;
//      fprintf(screen,"%s\n",line);
//      fprintf(logfile,"%s\n",line);
//    }

    // parse the line
    // if no command, skip to next line in input script
    parse();
//    std::cout << lnarg << std::endl;
//    for (int i = 0; i < lnarg; i++) {
//      std::cout << larg[i] << ", ";
//    }
//    std::cout << std::endl;

    if (lnarg==1) {
      if (utils::numeric(FLERR, larg[0], true, lmp) > maxshexpan) {
        error->one(FLERR, "Spherical Harmonic file expansion exceeds memory allocation");
      }
    }
    else if (lnarg==4){
      shcoeff[curfile][curentry] = utils::numeric(FLERR, larg[2], true, lmp);
      shcoeff[curfile][++curentry] = utils::numeric(FLERR, larg[3], true, lmp);
      curentry++;
    }
    else{
      error->one(FLERR, "Too many entries in Spherical Harmonic file line");
    }

  }
}

/* ----------------------------------------------------------------------
   rellocate a string
   if n > 0: set max >= n in increments of DELTALINE
   if n = 0: just increment max by DELTALINE
------------------------------------------------------------------------- */

void AtomVecSpherharmnocompute::reallocate(char *&str, int &max, int n)
{
  if (n) {
    while (n > max) max += DELTALINE;
  } else max += DELTALINE;

  str = (char *) memory->srealloc(str,max*sizeof(char),"input:str");
}

/* ----------------------------------------------------------------------
   return number of triple quotes in line
------------------------------------------------------------------------- */

int AtomVecSpherharmnocompute::numtriple(char *line)
{
  int count = 0;
  char *ptr = line;
  while ((ptr = strstr(ptr,"\"\"\""))) {
    ptr += 3;
    count++;
  }
  return count;
}

/* ----------------------------------------------------------------------
   parse copy of command line by inserting string terminators
   strip comment = all chars from # on
   replace all $ via variable substitution except within quotes
   command = first word
   narg = # of args
   arg[] = individual args
   treat text between single/double/triple quotes as one arg via nextword()
------------------------------------------------------------------------- */
void AtomVecSpherharmnocompute::parse()
{
  // duplicate line into copy string to break into words

  int n = strlen(line) + 1;
  if (n > maxcopy) reallocate(copy,maxcopy,n);
  strcpy(copy,line);

  // strip any # comment by replacing it with 0
  // do not strip from a # inside single/double/triple quotes
  // quoteflag = 1,2,3 when encounter first single/double,triple quote
  // quoteflag = 0 when encounter matching single/double,triple quote

  int quoteflag = 0;
  char *ptr = copy;
  while (*ptr) {
    if (*ptr == '#' && !quoteflag) {
      *ptr = '\0';
      break;
    }
    if (quoteflag == 0) {
      if (strstr(ptr,"\"\"\"") == ptr) {
        quoteflag = 3;
        ptr += 2;
      }
      else if (*ptr == '"') quoteflag = 2;
      else if (*ptr == '\'') quoteflag = 1;
    } else {
      if (quoteflag == 3 && strstr(ptr,"\"\"\"") == ptr) {
        quoteflag = 0;
        ptr += 2;
      }
      else if (quoteflag == 2 && *ptr == '"') quoteflag = 0;
      else if (quoteflag == 1 && *ptr == '\'') quoteflag = 0;
    }
    ptr++;
  }

  char *next;
  lnarg = 0;
  if (lnarg == maxarg) {
    maxarg += DELTA;
    larg = (char **) memory->srealloc(larg,maxarg*sizeof(char *),"input:arg");
  }
  larg[lnarg] = nextword(copy,&next);
  lnarg++;

  // point arg[] at each subsequent arg in copy string
  // nextword() inserts string terminators into copy string to delimit args
  // nextword() treats text between single/double/triple quotes as one arg

  ptr = next;
  while (ptr) {
    if (lnarg == maxarg) {
      maxarg += DELTA;
      larg = (char **) memory->srealloc(larg,maxarg*sizeof(char *),"input:arg");
    }
    larg[lnarg] = nextword(ptr,&next);
    if (!larg[lnarg]) break;
    lnarg++;
    ptr = next;
  }
}

/* ----------------------------------------------------------------------
   find next word in str
   insert 0 at end of word
   ignore leading whitespace
   treat text between single/double/triple quotes as one arg
   matching quote must be followed by whitespace char if not end of string
   strip quotes from returned word
   return ptr to start of word or NULL if no word in string
   also return next = ptr after word
------------------------------------------------------------------------- */

char *AtomVecSpherharmnocompute::nextword(char *str, char **next)
{
  char *start,*stop;

  // start = first non-whitespace char

  start = &str[strspn(str," \t\n\v\f\r")];
  if (*start == '\0') return NULL;

  // if start is single/double/triple quote:
  //   start = first char beyond quote
  //   stop = first char of matching quote
  //   next = first char beyond matching quote
  //   next must be NULL or whitespace
  // if start is not single/double/triple quote:
  //   stop = first whitespace char after start
  //   next = char after stop, or stop itself if stop is NULL

  if (strstr(start,"\"\"\"") == start) {
    stop = strstr(&start[3],"\"\"\"");
    if (!stop) error->all(FLERR,"Unbalanced quotes in input line");
    start += 3;
    *next = stop+3;
    if (**next && !isspace(**next))
      error->all(FLERR,"Input line quote not followed by white-space");
  } else if (*start == '"' || *start == '\'') {
    stop = strchr(&start[1],*start);
    if (!stop) error->all(FLERR,"Unbalanced quotes in input line");
    start++;
    *next = stop+1;
    if (**next && !isspace(**next))
      error->all(FLERR,"Input line quote not followed by white-space");
  } else {
    stop = &start[strcspn(start," \t\n\v\f\r")];
    if (*stop == '\0') *next = stop;
    else *next = stop+1;
  }

  // set stop to NULL to terminate word

  *stop = '\0';
  return start;
}