#!/bin/bash
LES_RL_FREQ_A=${LES_RL_FREQ_A:-4}
LES_RL_N_TSIM=${LES_RL_N_TSIM:-10}
# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_3D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} rlHIT -j4
fi

# copy executable:
cp ${COMPILEDIR}/rlHIT ${RUNDIR}/exec

# write simulation settings files:
cat <<EOF >${RUNDIR}/runArguments00.sh
./simulation -bpdx 4 -bpdy 4 -bpdz 4 -extentx 6.2831853072 -tend 500 \
-dump2D 1 -dump3D 1 -tdump 1.0 -BC_x periodic -BC_y periodic -BC_z periodic \
-CFL 0.1 -initCond HITurbulence -spectralIC fromFit -keepMomentumConstant 1 \
-nprocsx 1 -nprocsy 1 -nprocsz 1 -sgs RLSM -cs 0.5 -spectralForcing 1 \
-analysis HIT -tAnalysis 100 -nu 0.005 -energyInjectionRate 0.2 \
-RL_freqActions ${LES_RL_FREQ_A} -RL_nIntTperSim ${LES_RL_N_TSIM} \
-initCondFileTokens RE060,RE080,RE100,RE120,RE140,RE160,RE180,RE200
EOF

#copy target files
cp ${SMARTIES_ROOT}/apps/CUP3D_LES_HIT/targetNonDimRe/*  ${RUNDIR}/

export MPI_RANKS_PER_ENV=1
export EXTRA_LINE_ARGS=" --appSettings runArguments00.sh --nStepPappSett 0 "
