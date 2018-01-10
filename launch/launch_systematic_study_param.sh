
for EXPTYPE in "QUAD" "GAUS"; do
for NEXPERTS in "1"; do

export CPPFLAGS="-DNEXPERTS=${NEXPERTS} -DADV_${EXPTYPE}"
make -C ../makefiles clean
make -C ../makefiles config=fit -j

for BUFFSIZE in "16384" "65536" "262144"; do
for DKLPARAM in "0.02" "0.032" "0.05" "0.08" "0.125" "0.2" "0.32" "0.5"; do
for IMPSAMPR in "5" "2"; do
for BATCHNUM in "256"; do
for EPERSTEP in "1"; do
for RUNTRIAL in "1"; do

POSTFIX=obs_${EXPTYPE}_R${IMPSAMPR}_N${BUFFSIZE}_D${DKLPARAM}
NMASTERS=1

#source launchDaint_openai.sh standu_${POSTFIX} ${NMASTERS} HumanoidStandup-v1        settings/settings_bench_args.sh
source launchDaint_openai.sh humanw_${POSTFIX} ${NMASTERS} Humanoid-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} ${NMASTERS} InvertedPendulum-v1       settings/settings_bench_args.sh
source launchDaint_openai.sh spider_${POSTFIX} ${NMASTERS} Ant-v1                    settings/settings_bench_args.sh

source launchDaint_openai.sh dblpnd_${POSTFIX} ${NMASTERS} InvertedDoublePendulum-v1 settings/settings_bench_args.sh
source launchDaint_openai.sh walker_${POSTFIX} ${NMASTERS} Walker2d-v1               settings/settings_bench_args.sh
source launchDaint_openai.sh cheeta_${POSTFIX} ${NMASTERS} HalfCheetah-v1            settings/settings_bench_args.sh
source launchDaint_openai.sh swimmr_${POSTFIX} ${NMASTERS} Swimmer-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh hopper_${POSTFIX} ${NMASTERS} Hopper-v1                 settings/settings_bench_args.sh
source launchDaint_openai.sh reachr_${POSTFIX} ${NMASTERS} Reacher-v1                settings/settings_bench_args.sh

done
done
done
done
done
done
done
done
