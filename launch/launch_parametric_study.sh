POSTFIX=40
make -C ../makefiles clean
make -C ../makefiles config=prod -j

#./launchDaint_openai.sh spider_vanilla_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_vanilla_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_vanilla_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_vanilla_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_vanilla_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_vanilla_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_vanilla_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_vanilla_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_vanilla_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_vanilla_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod acer=relax -j

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
make -C ../makefiles config=prod tabc=on -j

#./launchDaint_openai.sh spider_tabc_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_tabc_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_tabc_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_tabc_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_tabc_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_tabc_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_tabc_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_tabc_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_tabc_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_tabc_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod importance=on -j

#./launchDaint_openai.sh spider_importance_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_importance_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_importance_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_importance_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_importance_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_importance_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_importance_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_importance_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_importance_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_importance_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod lambda=on -j

#./launchDaint_openai.sh spider_lambda_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_lambda_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_lambda_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_lambda_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_lambda_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_lambda_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_lambda_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_lambda_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_lambda_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_lambda_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod cvar=on -j

#./launchDaint_openai.sh spider_cvar_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_cvar_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_cvar_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_cvar_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_cvar_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_cvar_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_cvar_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_cvar_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_cvar_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_cvar_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod auxtask=on -j

#./launchDaint_openai.sh spider_auxtask_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_auxtask_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_auxtask_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_auxtask_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_auxtask_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_auxtask_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_auxtask_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_auxtask_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_auxtask_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_auxtask_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod sortseq=on -j

#./launchDaint_openai.sh spider_sortseq_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_sortseq_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_sortseq_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_sortseq_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_sortseq_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_sortseq_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_sortseq_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_sortseq_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_sortseq_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_sortseq_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh

make -C ../makefiles clean
make -C ../makefiles config=prod sampseq=on -j

#./launchDaint_openai.sh spider_sortseq_${POSTFIX} 1 Ant-v1                    settings/settings_RACER_bench.sh
#./launchDaint_openai.sh standu_sortseq_${POSTFIX} 1 HumanoidStandup-v1        settings/settings_RACER_bench.sh
#./launchDaint_openai.sh humanw_sortseq_${POSTFIX} 1 Humanoid-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh walker_sortseq_${POSTFIX} 1 Walker2d-v1               settings/settings_RACER_bench.sh
./launchDaint_openai.sh cheeta_sortseq_${POSTFIX} 1 HalfCheetah-v1            settings/settings_RACER_bench.sh
./launchDaint_openai.sh swimmr_sortseq_${POSTFIX} 1 Swimmer-v1                settings/settings_RACER_bench.sh
#./launchDaint_openai.sh hopper_sortseq_${POSTFIX} 1 Hopper-v1                 settings/settings_RACER_bench.sh
./launchDaint_openai.sh reachr_sortseq_${POSTFIX} 1 Reacher-v1                settings/settings_RACER_bench.sh
./launchDaint_openai.sh invpnd_sortseq_${POSTFIX} 1 InvertedPendulum-v1       settings/settings_RACER_bench.sh
#./launchDaint_openai.sh dblpnd_sortseq_${POSTFIX} 1 InvertedDoublePendulum-v1 settings/settings_RACER_bench.sh
