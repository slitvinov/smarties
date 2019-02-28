
for ASPECT in "0.025" "0.050" "0.100" "0.200" "0.400"; do
for DRATIO in "25" "50" "100" "200" "400" "800"; do
#for ASPECT in "0.01" "0.03" "0.1" "0.3" "1"; do
#for DRATIO in "10" "50" "200" "500" "1000"; do
for REWARD in "1" "2"; do

make -C ../apps/glider clean
make -C ../apps/glider train=0 aspectr=${ASPECT} density=${DRATIO} costfun=${REWARD}
PRRATIO=`printf "%03d" $DRATIO`
POSTFIX=${PRRATIO}_R${ASPECT}_C${REWARD}

CASE=dt05_alpha${ASPECT}_rho${PRRATIO}_cost${REWARD}
BASE=/cluster/scratch/novatig/smarties/ #/cluster/home/novatig/smarties/launch/
echo ${BASE} ${CASE} 

#cp /cluster/scratch/novatig/smarties/glider_bias_beta_0.1_rho_200_cost_${REWARD}/policy*  /cluster/scratch/novatig/smarties/glider_restart_beta_${ASPECT}_rho_${DRATIO}_cost_${REWARD}/
./launch.sh evals06_${CASE} 1 glider settings/settings_POAC_safe_restart.sh 1 1 1 ${BASE}/train06_${CASE}
#./launch_local.sh eval_${CASE} 2 2 glider settings/settings_POAC_restart.sh ${BASE}${CASE}/policy
#exit 0

done
done
done
