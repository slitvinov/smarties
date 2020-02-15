export SKIPMAKE=true

# how many cases to consider
for run in 8 9; do
for nblocks in 2; do

export LES_RL_NBLOCK=$nblocks
export LES_RL_N_TSIM=20
POSTNAME=sim${LES_RL_N_TSIM}_RUN${run}

# several options for actuation freq (relative to kolmogorov time)
# bcz it affects run time we allocate different number of resources:

################################################################################
export LES_RL_GRIDACT=0
export LES_RL_NETTYPE=GRU
BASENAME=BlockAgents_RK2ND_${LES_RL_NETTYPE}_${nblocks}blocks
echo $BASENAME
################################################################################

export LES_RL_FREQ_A=1
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 16 -r ${RUNDIR}

export LES_RL_FREQ_A=2
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 11 -r ${RUNDIR}

export LES_RL_FREQ_A=4
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 6 -r ${RUNDIR}

#export LES_RL_FREQ_A=8
#RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 4 -r ${RUNDIR}

export LES_RL_FREQ_A=10
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 4 -r ${RUNDIR}


export LES_RL_FREQ_A=20
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 3 -r ${RUNDIR}

################################################################################
export LES_RL_GRIDACT=0
export LES_RL_NETTYPE=FFNN
BASENAME=BlockAgents_RK2ND_${LES_RL_NETTYPE}_${nblocks}blocks
echo $BASENAME
################################################################################

export LES_RL_FREQ_A=1
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 25 -r ${RUNDIR}

export LES_RL_FREQ_A=2
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 11 -r ${RUNDIR}

export LES_RL_FREQ_A=4
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 7 -r ${RUNDIR}

#export LES_RL_FREQ_A=8
#RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 4 -r ${RUNDIR}

export LES_RL_FREQ_A=10
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT  -n 4 -r ${RUNDIR}


export LES_RL_FREQ_A=20
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 3 -r ${RUNDIR}

################################################################################
export LES_RL_GRIDACT=1
export LES_RL_NETTYPE=FFNN
BASENAME=GridAgents_RK2ND_${LES_RL_NETTYPE}_${nblocks}blocks
echo $BASENAME
################################################################################

export LES_RL_FREQ_A=1
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 18 -r ${RUNDIR}

export LES_RL_FREQ_A=2
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 11 -r ${RUNDIR}

export LES_RL_FREQ_A=4
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 6 -r ${RUNDIR}

#export LES_RL_FREQ_A=8
#RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 4 -r ${RUNDIR}

export LES_RL_FREQ_A=10
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT  -n 4 -r ${RUNDIR}


export LES_RL_FREQ_A=20
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 3 -r ${RUNDIR}

done
done


