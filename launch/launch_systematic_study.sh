#NMASTERS=$1
#BATCHNUM=$2
#TGTALPHA=$3
#DKLTARGT=$4
#EPERSTEP=$5

make -C ../makefiles clean
make -C ../makefiles config=fit -j

for BATCHNUM in "128" "256"; do
#for DKLTARGT in "0.01" "0.001"; do
for DKLTARGT in "0.001"; do
#for TGTALPHA in "0.1" "0.01"; do
for EPERSTEP in "0.1" "0.01"; do
for RUNTRIAL in "1" "2" "3"; do
TGTALPHA="0.1"

if [ $BATCHNUM -eq "128" ]; then
NMASTERS=2
else
NMASTERS=4
fi

POSTFIX=POPAC_TRIAL${RUNTRIAL}_B${BATCHNUM}_k${DKLTARGT}_o${EPERSTEP}

#source launchDaint_openai.sh spider_vanilla_${POSTFIX} ${NMASTERS} Ant-v1                    settings/settings_bench_args.sh
#source launchDaint_openai.sh standu_vanilla_${POSTFIX} ${NMASTERS} HumanoidStandup-v1        settings/settings_bench_args.sh
#source launchDaint_openai.sh humanw_vanilla_${POSTFIX} ${NMASTERS} Humanoid-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh walker_vanilla_${POSTFIX} ${NMASTERS} Walker2d-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh cheeta_vanilla_${POSTFIX} ${NMASTERS} HalfCheetah-v1            settings/settings_bench_args.sh
source launchDaint_openai.sh swimmr_vanilla_${POSTFIX} ${NMASTERS} Swimmer-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh hopper_vanilla_${POSTFIX} ${NMASTERS} Hopper-v1                 settings/settings_bench_args.sh
source launchDaint_openai.sh reachr_vanilla_${POSTFIX} ${NMASTERS} Reacher-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh invpnd_vanilla_${POSTFIX} ${NMASTERS} InvertedPendulum-v1       settings/settings_bench_args.sh
source launchDaint_openai.sh dblpnd_vanilla_${POSTFIX} ${NMASTERS} InvertedDoublePendulum-v1 settings/settings_bench_args.sh

done
done
done
done
#done


make -C ../makefiles clean
make -C ../makefiles config=fit tabc=on -j

for BATCHNUM in "128" "256"; do
#for DKLTARGT in "0.01" "0.001"; do
for DKLTARGT in "0.001"; do
#for TGTALPHA in "0.1" "0.01"; do
for EPERSTEP in "0.1" "0.01"; do
for RUNTRIAL in "1" "2" "3"; do
TGTALPHA="0.1"

if [ $BATCHNUM -eq "128" ]; then
NMASTERS=2
else
NMASTERS=4
fi

POSTFIX=POPAC_TRIAL${RUNTRIAL}_B${BATCHNUM}_k${DKLTARGT}_o${EPERSTEP}

#source launchDaint_openai.sh spider_tabc_${POSTFIX} ${NMASTERS} Ant-v1                    settings/settings_bench_args.sh
#source launchDaint_openai.sh standu_tabc_${POSTFIX} ${NMASTERS} HumanoidStandup-v1        settings/settings_bench_args.sh
#source launchDaint_openai.sh humanw_tabc_${POSTFIX} ${NMASTERS} Humanoid-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh walker_tabc_${POSTFIX} ${NMASTERS} Walker2d-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh cheeta_tabc_${POSTFIX} ${NMASTERS} HalfCheetah-v1            settings/settings_bench_args.sh
source launchDaint_openai.sh swimmr_tabc_${POSTFIX} ${NMASTERS} Swimmer-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh hopper_tabc_${POSTFIX} ${NMASTERS} Hopper-v1                 settings/settings_bench_args.sh
source launchDaint_openai.sh reachr_tabc_${POSTFIX} ${NMASTERS} Reacher-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh invpnd_tabc_${POSTFIX} ${NMASTERS} InvertedPendulum-v1       settings/settings_bench_args.sh
source launchDaint_openai.sh dblpnd_tabc_${POSTFIX} ${NMASTERS} InvertedDoublePendulum-v1 settings/settings_bench_args.sh

done
done
done
done
#done


make -C ../makefiles clean
make -C ../makefiles config=fit cvar=on acer=full -j

for BATCHNUM in "128" "256"; do
#for DKLTARGT in "0.01" "0.001"; do
for DKLTARGT in "0.001"; do
#for TGTALPHA in "0.1" "0.01"; do
for EPERSTEP in "0.1" "0.01"; do
for RUNTRIAL in "1" "2" "3"; do
TGTALPHA="0.1"

if [ $BATCHNUM -eq "128" ]; then
NMASTERS=2
else
NMASTERS=4
fi

POSTFIX=POPAC_TRIAL${RUNTRIAL}_B${BATCHNUM}_k${DKLTARGT}_o${EPERSTEP}

#source launchDaint_openai.sh spider_cvar_${POSTFIX} ${NMASTERS} Ant-v1                    settings/settings_bench_args.sh
#source launchDaint_openai.sh standu_cvar_${POSTFIX} ${NMASTERS} HumanoidStandup-v1        settings/settings_bench_args.sh
#source launchDaint_openai.sh humanw_cvar_${POSTFIX} ${NMASTERS} Humanoid-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh walker_cvar_${POSTFIX} ${NMASTERS} Walker2d-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh cheeta_cvar_${POSTFIX} ${NMASTERS} HalfCheetah-v1            settings/settings_bench_args.sh
source launchDaint_openai.sh swimmr_cvar_${POSTFIX} ${NMASTERS} Swimmer-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh hopper_cvar_${POSTFIX} ${NMASTERS} Hopper-v1                 settings/settings_bench_args.sh
source launchDaint_openai.sh reachr_cvar_${POSTFIX} ${NMASTERS} Reacher-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh invpnd_cvar_${POSTFIX} ${NMASTERS} InvertedPendulum-v1       settings/settings_bench_args.sh
source launchDaint_openai.sh dblpnd_cvar_${POSTFIX} ${NMASTERS} InvertedDoublePendulum-v1 settings/settings_bench_args.sh

done
done
done
done
#done


:'

make -C ../makefiles clean
make -C ../makefiles config=fit acer=relax -j

#./launchDaint_openai.sh spider_relax_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_relax_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_relax_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_relax_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_relax_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_relax_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_relax_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_relax_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_relax_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_relax_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=fit importance=on -j

#./launchDaint_openai.sh spider_importance_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_importance_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_importance_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_importance_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_importance_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_importance_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh hopper_importance_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_importance_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh invpnd_importance_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_importance_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh



make -C ../makefiles clean
make -C ../makefiles config=fit auxtask=on -j

#./launchDaint_openai.sh spider_auxtask_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_auxtask_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_auxtask_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_auxtask_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_auxtask_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_auxtask_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh hopper_auxtask_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_auxtask_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh invpnd_auxtask_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_auxtask_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=fit sortseq=on -j

./launchDaint_openai.sh spider_sortseq_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_sortseq_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_sortseq_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_sortseq_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_sortseq_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_sortseq_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh hopper_sortseq_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_sortseq_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh invpnd_sortseq_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_sortseq_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=fit sampseq=on -j

#./launchDaint_openai.sh spider_sampseq_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_sampseq_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_sampseq_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_sampseq_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_sampseq_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_sampseq_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_sampseq_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_sampseq_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_sampseq_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_sampseq_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=fit target=off -j

#./launchDaint_openai.sh spider_notarget_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_notarget_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_notarget_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_notarget_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_notarget_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_notarget_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_notarget_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_notarget_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_notarget_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_notarget_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh
'
