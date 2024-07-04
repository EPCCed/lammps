# Spherical harmonic-based Discrete Element Method (SH-DEM)

This directory contains the source code for the implementation of irregularly
shaped particles using spherical harmonics (SH) as described by

REFERENCE PENDING

Dcomumentation can be found at PENDING.

The contents include:

```
./src/SH-DEM                             # SH-DEM; this directory
./src/SH-DEM/atom_vec_shdem.cpp          # defines the new 'shdem' atom style
./src/SH-DEM/compute_efunction_shdem.cpp # compute potential energy
./src/SH-DEM/compute_erotate_shdem.cpp   # compute rotational kinetic energy
./src/SH-DEM/fix_nve_sh.cpp  		 # defines a time integration scheme
./src/SH-DEM/fix_wall_shdem.cpp          # enables particle/wall interactions
./src/SH-DEM/gaussquad_const.h           # Gaussian quadrature weights etc
./src/SH-DEM/math_shdem.cpp              # Additional mathemeatic functions
./src/SH-DEM/pair_sh.cpp                 # SH pairstyle (aka "feng")
```
The following directory also contains example scripts to reproduce the figures
shown in the paper:
```
./examples/PACKAGES/sh-dem
```
These files include:
```
in.shapes1        	# particles with 5 different shapes placed randomly in a box and assigned random initial velocities
in.shapes2  		# particles with 5 different shapes placed randomly in a box with zero initial velocities
in.test-simple1      	# a head-on collision of two spherical harmonic particles
in.test-simple2     	# as for in.test-simple1 except with a lower frequency of standard potential energy computation
in.test-wall1  		# impact of a spherical harmonic particle with a wall defined as a z-plane
```
The standard LAMMPS output is included in each case as a `.log` file. Data
files contain the coefficients of the spherical harmonic expansion for
a number of different shapes.

The majority of the code was written by James Young, with significant contributions from Mohammad Imaran, who finalised the development, and Kevin Stratford.

## License

LAMMPS is released under a GNU Public License Version 2 (GPLv2).
Please cite the authors in any derived works using the code by citing the
accompanying paper.

THE CODE COMES WITH NO WARRANTY OF ANY KIND.

THE AUTHORS AND CONTRIBUTORS SHALL NOT BE RESPONSIBLE FOR YOUR USE OF THE CODE OR ANY INFORMATION CONTAINED IN THE CODE, ITS DOCUMENTATION, OR ANY OTHER SOURCE REFERRING TO THE CODE OR ITS DOCUMENTATION.

### Gaussian Quadrature

The Gaussian quadrature code included as `guassquad_const.h` and parts of
`math_shdem.cpp` is that available from

https://people.math.sc.edu/Burkardt/cpp_src/fastgl/fastgl.html (January,2024)

and is released under a BSD license. It originates from
```
Ignace Bogaert,
Iteration-free computation of Gauss-Legendre quadrature nodes and weights,
SIAM Journal on Scientific Computing,
Volume 36, Number 3, 2014, pages A1008-1026.
```

## Obtaining the Code

The current development branch is:
```
git clone --branch feature-sh-dem https://github.com/EPCCed/lammps
```

## Compiling LAMMPS

LAMMPS can be built using either GNU Make or CMake in a similar manner to the standard LAMMPS distribution. The spherical harmonic implementation makes use of mathematical special functions, e.g., std::sph_legendre. Therefore it must be compiled using C++17 and the chosen compiler must support these mathematical special functions. The following are minimal examples to configure and compile LAMMPS in parallel.

A. Using GNU Make

```
cd <path-to-lammps>/src
make yes-SH-DEM 	# install the SH-DEM package in the source tree
make mpi          	# build the LAMMPS executable with MPI, having added the '-std=c++17' compiler flag to Makefile.mpi
```

B. Using CMake

```
cd <path-to-lammps>                         # change to the LAMMPS distribution directory
mkdir build; cd build                       # create and change to build directory
cmake -D BUILD_MPI=yes PKG_SH-DEM=yes CMAKE_CXX_STANDARD=17 ../cmake/ # include the SH-DEM package and enforce C++17
cmake --build .                             # compilation (or type "make")
```

## Running LAMMPS

Assuming the LAMMPS executable is 'lmp_mpi' and this in the search path, one of the example scripts included with the code can be run in the following manner:

```
cd <path-to-lammps>/examples/PACKAGES/sh-dem	# change to examples directory
lmp_mpi -in in.shapes1            		# run LAMMPS with the input file
```
