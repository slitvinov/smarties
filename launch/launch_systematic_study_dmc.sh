#for EXPTYPE in "QUAD" "GAUS"; do
for EXPTYPE in "GAUS"; do
for NEXPERTS in "1"; do
for TRICK in 0; do
for SKIPA in 1; do

make -C ../makefiles clean
make -C ../makefiles config=fit -j acertrick=$TRICK raceskip=$SKIPA exp=$EXPTYPE nexp=$NEXPERTS

for BUFFSIZE in "131072" "262144" "524288"; do
for DKLPARAM in "0.05" "0.1" "0.2"; do

for IMPSAMPR in "2"; do
for BATCHNUM in "256"; do
for EPERSTEP in "1"; do
for RUNTRIAL in "1" "2" "3"; do

POSTFIX=DCM11_${EXPTYPE}_R${IMPSAMPR}_N${BUFFSIZE}_D${DKLPARAM}_TRIAL${RUNTRIAL}
NMASTERS=1

declare -a listOfCases=( \
                        #"acrobot.swingup_sparse" \
                        "acrobot.swingup" \
                        #"ball_in_cup.catch" \
                        "cartpole.swingup" \
                        #"cartpole.balance_sparse" \
                        #"cartpole.balance" \
                        #"cartpole.swingup_sparse" \
                        "cheetah.run" \
                        #"finger.spin" \
                        "finger.turn_easy" \
                        "finger.turn_hard" \
                        #"fish.upright" \
                        "fish.swim" \
                        #"hopper.hop" \
                        #"hopper.stand" \
                        "humanoid.run" \
                        "humanoid.walk" \
                        "humanoid.stand" \
                        "manipulator.bring_ball" \
                        #"pendulum.swingup" \
                        #"point_mass.easy" \
                        #"reacher.easy" \
                        #"reacher.hard" \
                        "swimmer.swimmer15" \
                        "swimmer.swimmer6" \
                        #"walker.run" \
                        #"walker.walk" \
                        #"walker.stand" \
                      )
#
for RUNTRIAL in "${listOfCases[@]}" ; do

RUNNAME=${RUNTRIAL/./_}
RUNTRIAL=${RUNTRIAL/./ }
echo $RUNTRIAL $RUNNAME
source launchDaint_deepmind.sh ${RUNNAME}_${POSTFIX} $RUNTRIAL settings/settings_bench_args_DMC.sh

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
