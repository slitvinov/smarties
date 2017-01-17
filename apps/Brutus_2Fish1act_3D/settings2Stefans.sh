#!/bin/bash

BASENAME=2Stefans
NNODE=8
NNODEX=8
NNODEY=1

OPTIONS=
OPTIONS+=" -nActions 1"
OPTIONS+=" -bpdx 64 -bpdy 32 -bpdz 32"
OPTIONS+=" -2Ddump 1 -restart 0"
OPTIONS+=" -nprocsx ${NNODEX}"
OPTIONS+=" -nprocsy 1"
OPTIONS+=" -nprocsz 1"
OPTIONS+=" -CFL 0.2"
OPTIONS+=" -length 0.15"
OPTIONS+=" -lambda 1e4"
OPTIONS+=" -nu 0.0000045"
OPTIONS+=" -tend 80"
