#!/bin/bash
NNODEX=${NNODEX:-16}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-16}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-${BPDX}} #${BPDZ:-32}

NU=${NU:-0.003}

FACTORY=''

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -extentx 6.2831"
OPTIONS+=" -dump2D 0 -dump3D 1 -tdump 10"
OPTIONS+=" -BC_x periodic -BC_y periodic -BC_z periodic"
OPTIONS+=" -initCond HITurbulence"
OPTIONS+=" -spectralForcing 1"

#OPTIONS+=" -spectralIC cbc"
OPTIONS+=" -spectralIC art -tke0 0.67 -k0 4"
#OPTIONS+=" -spectralIC fromFile -spectralICFile /home/huguesl/smarties/apps/commonCUP3D/hit_re60/meanTarget.dat"

#OPTIONS+=" -sgs SSM -cs 0.2 -cs2spectrum 0"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy 1 -nprocsz 1"
OPTIONS+=" -CFL 0.1 -tend 30 -compute-dissipation 1"
OPTIONS+=" -analysis HIT -tAnalysis 0.1"
OPTIONS+=" -nu ${NU}"
