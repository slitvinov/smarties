#!/bin/bash

# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_3D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} rlHIT -j4
fi

# copy executable:
cp ${COMPILEDIR}/rlHIT ${RUNDIR}/exec

# write simulation settings files:
NNODEX=${NNODEX:-1}
NNODEY=${NNODEY:-1}
NNODEZ=${NNODEZ:-1}
NNODES=$(($NNODEX * $NNODEY * $NNODEZ))
BPDX=${BPDX:-4}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-${BPDX}} #${BPDZ:-32}
NU=${NU:-0.005}

cat <<EOF >${RUNDIR}/runArguments00.sh
./simulation -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -extentx 6.2831 \
-dump2D 0 -dump3D 0 -tdump 0.0 -BC_x periodic -BC_y periodic -BC_z periodic \
-initCond HITurbulence -spectralIC fromFile -keepMomentumConstant 1 \
-spectralICFile ../meanTarget.dat -nprocsx ${NNODEX} -nprocsy ${NNODEY} \
-nprocsz ${NNODEZ} -CFL 0.1 -tend 50 -sgs RLSM -compute-dissipation 1 \
-spectralForcing 1 -analysis HIT -tAnalysis 0.1 -nu ${NU}
EOF

#copy target files
cp targetHIT/re75/meanTarget.dat  ${RUNDIR}/
cp targetHIT/re75/kdeTarget.dat   ${RUNDIR}/
cp targetHIT/re75/scaleTarget.dat ${RUNDIR}/

export MPI_RANKS_PER_ENV=$NNODES
export EXTRA_LINE_ARGS=" --appSettings runArguments00.sh --nStepPappSett 0 "