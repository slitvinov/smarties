#!/bin/sh

m () {
    make \
	'FCFLAGS = -O2 -g'\
	'CFLAGS = -O2 -g -fPIC'\
	'CXXFLAGS = -O2 -g -fno-gnu-unique -fPIC' "$@"
}

m install &&
    (cd contrib/lib/f77 && m install) &&
    (cd contrib/lib/dummy && m install) &&
    (cd contrib/example/dlopen/lib && m) &&
    (cd contrib/example/dlopen && m)
