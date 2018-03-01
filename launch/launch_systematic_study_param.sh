COMMNAME=chosenParam_21

#for EXPTYPE in "QUAD" "GAUS"; do
for EXPTYPE in "GAUS"; do
for NEXPERTS in "1"; do
for TRICK in 0; do
for SKIPA in 1; do

make -C ../makefiles clean
make -C ../makefiles config=fit -j acertrick=$TRICK raceskip=$SKIPA exp=$EXPTYPE nexp=$NEXPERTS

#for BUFFSIZE in "65536" "131072"; do
for BUFFSIZE in "131072" "262144" "524288"; do

for DKLPARAM in "0.1" "0.2" "0.4"; do

for IMPSAMPR in "2"; do
for BATCHNUM in "256"; do
for EPERSTEP in "1"; do
for RUNTRIAL in "1" "2" "3"; do

POSTFIX=${COMMNAME}_${EXPTYPE}_S${SKIPA}_R${IMPSAMPR}_N${BUFFSIZE}_D${DKLPARAM}_TRICK${TRICK}_TRIAL${RUNTRIAL}
NMASTERS=1
source launchDaint_openai.sh standu_${POSTFIX} HumanoidStandup-v2        settings/settings_bench_args.sh
source launchDaint_openai.sh humanw_${POSTFIX} Humanoid-v2               settings/settings_bench_args.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} InvertedPendulum-v2       settings/settings_bench_args.sh
source launchDaint_openai.sh spider_${POSTFIX} Ant-v2                    settings/settings_bench_args.sh
#source launchDaint_openai.sh dblpnd_${POSTFIX} InvertedDoublePendulum-v2 settings/settings_bench_args.sh
source launchDaint_openai.sh walker_${POSTFIX} Walker2d-v2               settings/settings_bench_args.sh
source launchDaint_openai.sh cheeta_${POSTFIX} HalfCheetah-v2            settings/settings_bench_args.sh
source launchDaint_openai.sh swimmr_${POSTFIX} Swimmer-v2                settings/settings_bench_args.sh
#source launchDaint_openai.sh hopper_${POSTFIX} Hopper-v2                 settings/settings_bench_args.sh
#source launchDaint_openai.sh reachr_${POSTFIX} Reacher-v2                settings/settings_bench_args.sh

done
done
done
done
done
done

done
done
done
done
