#!/bin/bash

case `hostname` in
    eu-login-*)
	. /cluster/apps/local/env2lmod.sh
	module load gcc openmpi
    ;;
esac

m () {
    make \
	'FCFLAGS = -O0 -g'\
	'CFLAGS = -O0 -g -fPIC'\
	'CXXFLAGS = -O0 -g -fno-gnu-unique -fPIC' "$@"
}

m install &&
(cd contrib/lib/f77 && m install) &&
(cd contrib/example/dlopen/lib && m) &&
(cd contrib/example/dlopen && m)
