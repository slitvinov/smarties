COMMNAME=noanneal2

for EXPTYPE in "GAUS"; do
for NEXPERTS in "1"; do
for SKIPA in "1"; do

make -C ../makefiles clean
make -C ../makefiles config=fit -j raceskip=$SKIPA exp=$EXPTYPE nexp=$NEXPERTS

#for BUFFSIZE in "131072" "262144" "524288" "1048576"; do
for BUFFSIZE in "131072" "262144" "524288" "1048576"; do
#for IMPSAMPR in "0.5" "1" "2" "4"; do
for IMPSAMPR in "1" "3" "7"; do
#for DKLPARAM in "0.05" "0.1" "0.2"; do
for DKLPARAM in "0.10"; do

for BATCHNUM in "256"; do
for EPERSTEP in "1"; do

for RUNTRIAL in "1" "2" "3" "4" "5"; do

POSTFIX=${COMMNAME}_R${IMPSAMPR}_N${BUFFSIZE}_D${DKLPARAM}_TRIAL${RUNTRIAL}
NMASTERS=1
echo $POSTFIX
#source launchDaint_openai.sh standu_${POSTFIX} HumanoidStandup-v2        settings/settings_bench_args.sh
source launchDaint_openai.sh humanw_${POSTFIX} Humanoid-v2               settings/settings_bench_args.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} InvertedPendulum-v2       settings/settings_bench_args.sh
source launchDaint_openai.sh spider_${POSTFIX} Ant-v2                    settings/settings_bench_args.sh
#source launchDaint_openai.sh dblpnd_${POSTFIX} InvertedDoublePendulum-v2 settings/settings_bench_args.sh
source launchDaint_openai.sh walker_${POSTFIX} Walker2d-v2               settings/settings_bench_args.sh
source launchDaint_openai.sh cheeta_${POSTFIX} HalfCheetah-v2            settings/settings_bench_args.sh
#source launchDaint_openai.sh swimmr_${POSTFIX} Swimmer-v2                settings/settings_bench_args.sh
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
