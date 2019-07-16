#!/bin/bash
NNODEX=${NNODEX:-16}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-16}
BPDY=${BPDY:-${BPDX}}
BPDZ=${BPDZ:-${BPDX}}

NU=${NU:-0.005}

FACTORY=''

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -extentx 6.2831"
OPTIONS+=" -dump2D 0 -dump3D 0 -tdump 0"
OPTIONS+=" -BC_x periodic -BC_y periodic -BC_z periodic"
OPTIONS+=" -initCond HITurbulence"
OPTIONS+=" -spectralForcing 1"

OPTIONS+=" -spectralIC art -tke0 0.67 -k0 4 -nu 0.00195"

OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -tend 30 -compute-dissipation 1"
OPTIONS+=" -analysis HIT -tAnalysis 0.1"
OPTIONS+=" -nu ${NU}"
