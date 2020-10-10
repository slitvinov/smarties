#!/bin/sh

: ${PREFIX=$HOME/.local}

FCFLAGS='-Ofast -g'
CFLAGS='-Ofast -g -fPIC'
CXXFLAGS='-Ofast -g -fno-gnu-unique -fPIC'
MAKEFLAGS=-e

export FCFLAGS CFLAGS CXXFLAGS MAKEFLAGS
make install &&
(cd contrib/lib/f77 && make install) &&
(cd contrib/example/dlopen/lib && make) &&
(cd contrib/example/dlopen && make)
