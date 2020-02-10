export SKIPMAKE=true

# how many cases to consider
for run in 0 1; do
for nblocks in 2; do
for nntype in GRU FFNN; do
for gridAgent in 0 1; do
#for nntype in GRU; do
#for nntype in FFNN; do

export LES_RL_NETTYPE=$nntype
export LES_RL_NBLOCK=$nblocks
export LES_RL_N_TSIM=20
export LES_RL_GRIDACT=${gridAgent}

if [ ${gridAgent} == 1 ] ; then
  AGENT=Grid
else
  AGENT=Block
fi

# cannot use RNN and grid-based agents:
if [ ${gridAgent} == 1 ] && [ ${nntype} == GRU ]; then
  continue
fi

BASENAME=${AGENT}Agents_RK2ND_EXP_${LES_RL_NETTYPE}_${nblocks}blocks
POSTNAME=sim${LES_RL_N_TSIM}_RUN${run}

# several options for actuation freq (relative to kolmogorov time)
# bcz it affects run time we allocate different number of resources:

echo $BASENAME

export LES_RL_FREQ_A=1
ACTSPEC=act`printf %02d $LES_RL_FREQ_A`
RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
smarties.py CUP3D_LES_HIT  -n 20 -r ${RUNDIR}


export LES_RL_FREQ_A=2
ACTSPEC=act`printf %02d $LES_RL_FREQ_A`
RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
smarties.py CUP3D_LES_HIT  -n 11 -r ${RUNDIR}


export LES_RL_FREQ_A=4
ACTSPEC=act`printf %02d $LES_RL_FREQ_A`
RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
smarties.py CUP3D_LES_HIT  -n 6 -r ${RUNDIR}


export LES_RL_FREQ_A=8
ACTSPEC=act`printf %02d $LES_RL_FREQ_A`
RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
smarties.py CUP3D_LES_HIT  -n 4 -r ${RUNDIR}


export LES_RL_FREQ_A=16
ACTSPEC=act`printf %02d $LES_RL_FREQ_A`
RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 5 -r ${RUNDIR}


export LES_RL_FREQ_A=32
ACTSPEC=act`printf %02d $LES_RL_FREQ_A`
RUNDIR=${BASENAME}_${ACTSPEC}_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 3 -r ${RUNDIR}


done
done
done
done

#smarties.py CUP3D_LES_HIT VRACER_LES.json       -r \
#${BASENAME}FFNN_default_RUN${run} -n 8

#smarties.py CUP3D_LES_HIT VRACER_LES_gamma.json -r \
#${BASENAME}GRU_gamma_RUN${run}   -n 8

#smarties.py CUP3D_LES_HIT VRACER_LES_clip.json  -r \
#${BASENAME}GRU_refer_RUN${run}   -n 8

#smarties.py CUP3D_LES_HIT VRACER_LES_size.json  -r \
#${BASENAME}GRU_nnsize_RUN${run}  -n 8

#smarties.py CUP3D_LES_HIT VRACER_LES_noise.json -r \
#${BASENAME}GRU_noise_RUN${run}   -n 8

#smarties.py CUP3D_LES_HIT VRACER_LES_eta.json   -r \
#${BASENAME}GRU_nnlrate_RUN${run} -n 8

#smarties.py CUP3D_LES_HIT VRACER_LES_racer.json  -r \
#${BASENAME}GRU_racer_RUN${run} -n 8

