#!/bin/bash
LES_RL_NETTYPE=${LES_RL_NETTYPE:-GRU}
LES_RL_FREQ_A=${LES_RL_FREQ_A:-4}
LES_RL_N_TSIM=${LES_RL_N_TSIM:-10}
LES_RL_NBLOCK=${LES_RL_NBLOCK:-4}
LES_RL_GRIDACT=${LES_RL_GRIDACT:-0}

# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_3D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} rlHIT -j4
fi

# copy executable:
cp ${COMPILEDIR}/rlHIT ${RUNDIR}/exec

# write simulation settings files:
cat <<EOF >${RUNDIR}/runArguments00.sh
./simulation -bpdx $LES_RL_NBLOCK -bpdy $LES_RL_NBLOCK -bpdz $LES_RL_NBLOCK \
-extentx 6.2831853072 -tend 500 -dump2D 1 -dump3D 1 -tdump 1.0 -CFL 0.1 \
-BC_x periodic -BC_y periodic -BC_z periodic -initCond HITurbulence \
-spectralIC fromFit -keepMomentumConstant 1 -sgs RLSM -cs 0.5 -RungeKutta23 1 \
-spectralForcing 1 -nprocsx 1 -nprocsy 1 -nprocsz 1 -Advection3rdOrder 0 \
-analysis HIT -tAnalysis 100 -nu 0.005 -energyInjectionRate 0.2 \
-RL_freqActions ${LES_RL_FREQ_A} -RL_nIntTperSim ${LES_RL_N_TSIM} \
-RL_gridPointAgents ${LES_RL_GRIDACT} \
-initCondFileTokens RE065,RE076,RE088,RE103,RE120,RE140,RE163
EOF
#-initCondFileTokens RE060,RE070,RE082,RE095,RE111,RE130,RE152,RE176

#copy target files
THISDIR=${SMARTIES_ROOT}/apps/CUP3D_LES_HIT
cp ${THISDIR}/target_RK_${LES_RL_NBLOCK}blocks/*  ${RUNDIR}/

#copy settings file
# 1) either FFNN or RNN
# 2) number of actions per grad steps affected by number of agents per sim
SETTINGSF=VRACER_LES_${LES_RL_NETTYPE}_${LES_RL_NBLOCK}blocks.json
cp ${THISDIR}/settings/${SETTINGSF} ${RUNDIR}/settings.json

export MPI_RANKS_PER_ENV=1
export EXTRA_LINE_ARGS=" --appSettings runArguments00.sh --nStepPappSett 0 "
