#!/usr/bin/env bash

module load PrgEnv-gnu

module list

# See, e.g, https://doc.cgal.org/latest/Manual/usage.html
# In priciple: boost, GNU gmp and mpfr
# However, we want header-only CGAL, so nothing is required here.

# Specifically, for boost, don't load the module.
# The module version appears to interact badly with the non-default compiler.

git clone https://github.com/CGAL/cgal.git 
cd cgal

export MY_CGAL=$(pwd)

mkdir _build
cd _build

cmake -DCMAKE_INSTALL_PREFIX=${MY_CGAL} -DCMAKE_BUILD_TYPE=Release \
      -DCGAL_DIR=${MY_CGAL} ..

make
make install
