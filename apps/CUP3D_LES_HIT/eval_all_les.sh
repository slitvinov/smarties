export SKIPMAKE=true

export LES_RL_NETTYPE=${LES_RL_NETTYPE:-FFNN}
export LES_RL_N_TSIM=${LES_RL_N_TSIM:-100}
export LES_RL_NBLOCK=${LES_RL_NBLOCK:-2}

THISDIR=${SMARTIES_ROOT}/apps/CUP3D_LES_HIT

for GRIDAGENT in 0 1 ; do
for ACT in 2 4 8 16 ; do
for RE in RE065 RE076 RE088 RE103 RE120 RE140 RE163 ; do

export LES_RL_EVALUATE=${RE}
export LES_RL_FREQ_A=${ACT}
export LES_RL_GRIDACT=1 #${GRIDAGENT}

SPEC=${LES_RL_NETTYPE}_${LES_RL_NBLOCK}blocks_act`printf %02d $LES_RL_FREQ_A`

if [ ${GRIDAGENT} == 0 ] ; then
BASENAME=GridAgents_BlockRestart
RESTART=BlockAgent
export LES_RL_GRIDACTSETTINGS=0
else
BASENAME=GridAgents_GridRestart
RESTART=GridAgent
export LES_RL_GRIDACTSETTINGS=1
fi

RUNDIR=${BASENAME}_RK2ND_${SPEC}_sim${LES_RL_N_TSIM}_${RE}
RESTARTDIR=${THISDIR}/trained_${RESTART}_${SPEC}/
echo $RUNDIR

smarties.py CUP3D_LES_HIT -n 1 -r ${RUNDIR} --restart ${RESTARTDIR} --nEvalSeqs 1
#--printAppStdout

done
done
done
