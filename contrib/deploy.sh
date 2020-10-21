#!/bin/bash

set -e
cd
git clone --depth 1 --recursive git@github.com:Nek5000/Nek5000.git
git clone -b f77 git@github.com:slitvinov/smarties
cd smarties
. contrib/smarties.env
./contrib/install.sh
cd contrib/turbChannel/dlopen
MPI=0 FFLAGS='-O2 -g -fPIC' CFLAGS='-O2 -g -fPIC' ~/Nek5000/bin/nekconfig  -build-dep
MPI=0 ~/Nek5000/bin/nekconfig
make -f make/lib.mk -j
make -f make/bin.mk
