export SKIPMAKE=true

# how many cases to consider
for RE in RE065 RE076 RE088 RE103 RE120 RE140 RE163; do


export LES_RL_NBLOCK=2
export LES_RL_FREQ_A=32
export LES_RL_N_TSIM=100
export LES_RL_GRIDACT=0
export LES_RL_NETTYPE=FFNN
export LES_RL_EVALUATE=$RE

RESTART=BlockAgent_${LES_RL_NETTYPE}_${LES_RL_NBLOCK}blocks_act${LES_RL_FREQ_A}

BASENAME=BlockAgents_RK2ND_${LES_RL_NETTYPE}_${nblocks}blocks

ACTSPEC=act`printf %02d $LES_RL_FREQ_A`

POSTNAME=sim${LES_RL_N_TSIM}_RE${RE}

RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
THISDIR=${SMARTIES_ROOT}/apps/CUP3D_LES_HIT
echo $RUNDIR

RESTARTDIR=${THISDIR}/trained_${RESTART}/
smarties.py CUP3D_LES_HIT -n 2 -r ${RUNDIR} --restart ${RESTARTDIR} --nEvalSeqs 1 --printAppStdout

done


