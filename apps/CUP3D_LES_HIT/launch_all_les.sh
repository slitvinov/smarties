export SKIPMAKE=true;

BASENAME=BlockAgents_TargetNondim_balanced_FFNN
NPROCS=8
SETTINGS=VRACER_LES.json
#SETTINGS=VRACER_LES_LSTM.json

# how many cases to consider
for run in 1 2 3; do

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

export LES_RL_N_TSIM=20

export LES_RL_FREQ_A=4
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 9 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_FREQ_A=1
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 33 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_FREQ_A=16
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 3 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_FREQ_A=2
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 17 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_FREQ_A=8
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 5 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_FREQ_A=32
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 2 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_FREQ_A=4

export LES_RL_N_TSIM=10
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 9 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_N_TSIM=40
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 9 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_N_TSIM=60
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 9 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

export LES_RL_N_TSIM=80
smarties.py CUP3D_LES_HIT ${SETTINGS}  -n 9 -r \
${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}

done
