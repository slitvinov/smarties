
BASEDIR="/scratch/snx3000/novatig/smarties/"
PREFIX="_chosenParam_12_GAUS_"
FNAME="/grads_dist.raw"

for ENV in "walker" "spider" "reachr" "humanw" "cheeta" "hopper" "swimmr" "dblpnd"; do
#for R in "2"; do
for R in "5"; do
#for N in "131072" "262144" "524288"; do
for N in "262144"; do
for D in "0.2"; do
for B in "256"; do
for O in "1.0"; do

RUN=${BASEDIR}${ENV}${PREFIX}_R${R}_N${N}_D${D}_B${B}_O${O}_TRIAL
echo
python excess_kurtosis.py 16 ${RUN}1${FNAME} ${RUN}2${FNAME} ${RUN}3${FNAME}

done
done
done
done
done
done
