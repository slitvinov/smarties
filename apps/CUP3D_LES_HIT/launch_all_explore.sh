export SKIPMAKE=true;

BASENAME=BlockAgents_NaturalNondim_NonDimRe_GRU_
#BASENAME=BlockAgents_NaturalNondim_noTarget_
#BASENAME=BlockAgents_NaturalNondim_noScales_
#BASENAME=BlockAgents_NaturalNondim_noLaplac_
SETTINGS=VRACER_LES_LSTM.json
NPROCS=9

# how many cases to consider
for run in 1 2 3; do
export LES_RL_N_TSIM=20
export LES_RL_FREQ_A=4
smarties.py CUP3D_LES_HIT ${SETTINGS} -n ${NPROCS} -r \
${BASENAME}act`printf %02d $LES_RL_FREQ_A`_sim${LES_RL_N_TSIM}_RUN${run}
done
