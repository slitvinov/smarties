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
-initCond HITurbulence -spectralIC fromFile -keepMomentumConstant 1 \
-nprocsx 1 -nprocsy 1 -nprocsz 1 -sgs RLSM -cs 0.25 -spectralForcing 1 \
-analysis HIT -tAnalysis 0.1 -nu 0.005 -energyInjectionRate 0.2 \
-initCondFileTokens EPS0.010_NU0.0016,EPS0.010_NU0.0032,EPS0.020_NU0.0016,EPS0.020_NU0.0032,EPS0.020_NU0.0064,EPS0.080_NU0.0032,EPS0.080_NU0.0064,EPS0.080_NU0.0032,EPS0.160_NU0.0064
EOF

#copy target files
cp ${SMARTIES_ROOT}/apps/CUP3D_LES_HIT/target/*  ${RUNDIR}/

export MPI_RANKS_PER_ENV=1
export EXTRA_LINE_ARGS=" --appSettings runArguments00.sh --nStepPappSett 0 "