#NMASTERS=$1
#BATCHNUM=$2
#TGTALPHA=$3
#DKLTARGT=$4
#EPERSTEP=$5


#for BATCHNUM in "128" "512"; do
#for BATCHNUM in "64"; do
#for BATCHNUM in "256"; do
#for ALGOFLAG in "ACER_TABC" "ACER_NOCLIP" "ACER_CLIP_1" "ACER_CLIP_5"; do

for ALGOFLAG in "ACER_2gausExp"; do
export CPPFLAGS=-DNEXPERTS=2 #-DADV_QUAD
#for ALGOFLAG in "ACER_2quadExp"; do
#export CPPFLAGS=-DNEXPERTS=2 -DADV_QUAD

#for ALGOFLAG in "ACER_TABC" "ACER_NOCLIP" "ACER_CLIP_5"; do

make -C ../makefiles clean
make -C ../makefiles config=fit -j
#make -C ../makefiles config=debug -j

#for BUFFSIZE in "100 1000"; do
for BUFFSIZE in "100"; do
for IMPSAMPW in "1" "5"; do
#for BATCHNUM in "64" "128"; do
for BATCHNUM in "128"; do
#for EPERSTEP in "1" "6.4"; do
for EPERSTEP in "6.4"; do
#for RUNTRIAL in "10" "11" "12"; do
#for RUNTRIAL in "13" "14" "15"; do
for RUNTRIAL in "1" "2" "3"; do

NMASTERS=1

#POSTFIX=POPACsigmatabc_notgt_p4_t1_i1_TRIAL${RUNTRIAL}_B${BATCHNUM}_O${EPERSTEP}
#POSTFIX=POPACsigmacap_notgt_p4_t1_i1_TRIAL${RUNTRIAL}_B${BATCHNUM}_O${EPERSTEP}
POSTFIX=${ALGOFLAG}_M${BUFFSIZE}_C${IMPSAMPW}_B${BATCHNUM}_O${EPERSTEP}_TRIAL${RUNTRIAL}

echo $POSTFIX
#POSTFIX=POPAC_clip_notgt_p4_t1_i1_TRIAL${RUNTRIAL}_B${BATCHNUM}_O${EPERSTEP}
#POSTFIX=POPACsigmaunb_notgt_p4_t1_i1_TRIAL${RUNTRIAL}_B${BATCHNUM}
#POSTFIX=POPACsigmadiag_notgt_p4_t5_i5_TRIAL${RUNTRIAL}_B${BATCHNUM}

#source launchDaint_openai.sh spider_vanilla_${POSTFIX} ${NMASTERS} Ant-v1                    settings/settings_bench_args.sh
#source launchDaint_openai.sh standu_vanilla_${POSTFIX} ${NMASTERS} HumanoidStandup-v1        settings/settings_bench_args.sh
#source launchDaint_openai.sh humanw_vanilla_${POSTFIX} ${NMASTERS} Humanoid-v1               settings/settings_bench_args.sh
#source launchDaint_openai.sh invpnd_vanilla_${POSTFIX} ${NMASTERS} InvertedPendulum-v1       settings/settings_bench_args.sh

source launchDaint_openai.sh walker_${POSTFIX} ${NMASTERS} Walker2d-v1               settings/settings_bench_args.sh
source launchDaint_openai.sh cheeta_${POSTFIX} ${NMASTERS} HalfCheetah-v1            settings/settings_bench_args.sh
source launchDaint_openai.sh swimmr_${POSTFIX} ${NMASTERS} Swimmer-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh hopper_${POSTFIX} ${NMASTERS} Hopper-v1                 settings/settings_bench_args.sh
source launchDaint_openai.sh reachr_${POSTFIX} ${NMASTERS} Reacher-v1                settings/settings_bench_args.sh
source launchDaint_openai.sh dblpnd_${POSTFIX} ${NMASTERS} InvertedDoublePendulum-v1 settings/settings_bench_args.sh

done
done
done
done
done
done

