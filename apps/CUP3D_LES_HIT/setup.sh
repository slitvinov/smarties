#!/bin/bash

# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_3D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} rlHIT -j4
fi

# copy executable:
cp ${COMPILEDIR}/rlHIT ${RUNDIR}/exec

# write simulation settings files:
cat <<EOF >${RUNDIR}/runArguments00.sh
./simulation -bpdx 4 -bpdy 4 -bpdz 4 -extentx 6.2831 -tend 500 -CFL 0.1 \
-dump2D 0 -dump3D 0 -tdump 0.0 -BC_x periodic -BC_y periodic -BC_z periodic \
-initCond HITurbulence -spectralIC fromFit -keepMomentumConstant 1 -nu 0.005 \
-nprocsx 1 -nprocsy 1 -nprocsz 1 -sgs RLSM -compute-dissipation 1  -cs 0.25 \
-spectralForcing 1 -analysis HIT -tAnalysis 0.1 -energyInjectionRate 0.2
EOF

#copy target files
#cp targetHIT/re75/meanTarget.dat  ${RUNDIR}/
#cp targetHIT/re75/kdeTarget.dat   ${RUNDIR}/
#cp targetHIT/re75/scaleTarget.dat ${RUNDIR}/

export MPI_RANKS_PER_ENV=1
export EXTRA_LINE_ARGS=" --appSettings runArguments00.sh --nStepPappSett 0 "