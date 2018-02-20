
BASEDIR="/scratch/snx3000/novatig/smarties/"
PREFIX="_chosenParam_11_GAUS_S1_"
FNAME="/grads_dist.raw"

for ENV in "walker" "spider" "reachr" "humanw" "cheeta" "hopper" "swimmr" "dblpnd"; do
for R in "2"; do
for N in "131072" "262144" "524288"; do
for D in "0.05";

RUN=${BASEDIR}${ENV}${PREFIX}_R${R}_N${N}_D${D}_TRICK0_TRIAL
echo
python excess_kurtosis.py 16 ${RUN}1${FNAME} ${RUN}2${FNAME} ${RUN}3${FNAME}

done
done
done
done
