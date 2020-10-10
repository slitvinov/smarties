#!/bin/bash

case `hostname` in
    eu-login-*)
	. /cluster/apps/local/env2lmod.sh
	module load gcc openmpi
    ;;
esac

FCFLAGS='-O0 -g'
CFLAGS='-O0 -g -fPIC'
CXXFLAGS='-O0 -g -fno-gnu-unique -fPIC'
MAKEFLAGS=-e
export FCFLAGS CFLAGS CXXFLAGS MAKEFLAGS
make install &&
(cd contrib/lib/f77 && make install) &&
(cd contrib/example/dlopen/lib && make) &&
(cd contrib/example/dlopen && make)
