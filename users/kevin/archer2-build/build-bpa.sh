#!/usr/bin/env bash

set -e

# Ball pivoting GPL
# See https://www.ipol.im/pub/art/2014/81/?utm_source=doi

# A useful overview
# https://github.com/Lotemn102/Ball-Pivoting-Algorithm
# There are one or two other implementations in github

module load PrgEnv-gnu

wget https://www.ipol.im/pub/art/2014/81/BallPivoting.tgz

tar xf BallPivoting.tgz
cd BallPivoting

# The CMakeLists is badly wrong, so use an ammended one ...

cp ../CMakeLists-ballpivoting.txt CMakeLists.txt

mkdir include
mkdir lib
mkdir _build

cd _build

# Will produce the library in current directory
cmake -DCMAKE_INSTALL_PREFIX=$(pwd) -DCMAKE_CXX_FLAGS="-O2" ..
make

# "Install": we need the headers is a sensible place...
cp libballpivoting.a ../lib
cp ../src/*h ../include

