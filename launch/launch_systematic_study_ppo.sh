#NMASTERS=$1
#BATCHNUM=$2
#TGTALPHA=$3
#DKLTARGT=$4
#EPERSTEP=$5

make -C ../makefiles clean
make -C ../makefiles config=fit -j

for RUNTRIAL in "1" "2" "3"; do

NMASTERS=1
POSTFIX=PPO_00_TRIAL${RUNTRIAL}

source launchDaint_openai.sh humanw_${POSTFIX} Humanoid-v2               settings/settings_bench_ppo.sh
source launchDaint_openai.sh spider_${POSTFIX} Ant-v2                    settings/settings_bench_ppo.sh
source launchDaint_openai.sh walker_${POSTFIX} Walker2d-v2               settings/settings_bench_ppo.sh
source launchDaint_openai.sh cheeta_${POSTFIX} HalfCheetah-v2            settings/settings_bench_ppo.sh
source launchDaint_openai.sh swimmr_${POSTFIX} Swimmer-v2                settings/settings_bench_ppo.sh
source launchDaint_openai.sh hopper_${POSTFIX} Hopper-v2                 settings/settings_bench_ppo.sh
source launchDaint_openai.sh reachr_${POSTFIX} Reacher-v2                settings/settings_bench_ppo.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} InvertedPendulum-v1      settings/settings_bench_ppo.sh
source launchDaint_openai.sh dblpnd_${POSTFIX} InvertedDoublePendulum-v2 settings/settings_bench_ppo.sh

POSTFIX=ACE_00_TRIAL${RUNTRIAL}
source launchDaint_openai.sh humanw_${POSTFIX} Humanoid-v2               settings/settings_ACER.sh
source launchDaint_openai.sh spider_${POSTFIX} Ant-v2                    settings/settings_ACER.sh
source launchDaint_openai.sh walker_${POSTFIX} Walker2d-v2               settings/settings_ACER.sh
source launchDaint_openai.sh cheeta_${POSTFIX} HalfCheetah-v2            settings/settings_ACER.sh
source launchDaint_openai.sh swimmr_${POSTFIX} Swimmer-v2                settings/settings_ACER.sh
source launchDaint_openai.sh hopper_${POSTFIX} Hopper-v2                 settings/settings_ACER.sh
source launchDaint_openai.sh reachr_${POSTFIX} Reacher-v2                settings/settings_ACER.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} InvertedPendulum-v1      settings/settings_ACER.sh
source launchDaint_openai.sh dblpnd_${POSTFIX} InvertedDoublePendulum-v2 settings/settings_ACER.sh

POSTFIX=DPG_00_TRIAL${RUNTRIAL}
source launchDaint_openai.sh humanw_${POSTFIX} Humanoid-v2               settings/settings_DPG.sh
source launchDaint_openai.sh spider_${POSTFIX} Ant-v2                    settings/settings_DPG.sh
source launchDaint_openai.sh walker_${POSTFIX} Walker2d-v2               settings/settings_DPG.sh
source launchDaint_openai.sh cheeta_${POSTFIX} HalfCheetah-v2            settings/settings_DPG.sh
source launchDaint_openai.sh swimmr_${POSTFIX} Swimmer-v2                settings/settings_DPG.sh
source launchDaint_openai.sh hopper_${POSTFIX} Hopper-v2                 settings/settings_DPG.sh
source launchDaint_openai.sh reachr_${POSTFIX} Reacher-v2                settings/settings_DPG.sh
#source launchDaint_openai.sh invpnd_${POSTFIX} InvertedPendulum-v1      settings/settings_DPG.sh
source launchDaint_openai.sh dblpnd_${POSTFIX} InvertedDoublePendulum-v2 settings/settings_DPG.sh

done
