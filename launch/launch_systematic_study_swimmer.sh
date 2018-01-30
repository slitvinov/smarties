
for EXPTYPE in QUAD GAUS; do
for NEXPERTS in 1; do
for TRICK in 1 0; do
for SKIPA in 1 3; do
for FWDA in 0 10; do
for BCKA in 1; do
for ADVE in 1 0; do

make -C ../makefiles clean
make -C ../makefiles config=fit -j acertrick=$TRICK raceskip=$SKIPA racefrwd=$FWDA raceadve=$ADVE raceback=$BCKA exp=$EXPTYPE nexp=$NEXPERTS

for BUFFSIZE in "131072" "65536"; do
for DKLPARAM in "0.01"; do
for IMPSAMPR in "5" "2"; do
for BATCHNUM in "256"; do
for EPERSTEP in "1"; do
for RUNTRIAL in "1"; do

POSTFIX=${EXPTYPE}_trick${TRICK}_offP${SKIPA}_fwd${FWDA}_bck${BCKA}_adv${ADVE}_R${IMPSAMPR}_N${BUFFSIZE}
NMASTERS=1
echo $POSTFIX

source launchDaint_openai.sh swimmr_${POSTFIX} ${NMASTERS} Swimmer-v1                settings/settings_bench_args.sh

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
done
done
done
