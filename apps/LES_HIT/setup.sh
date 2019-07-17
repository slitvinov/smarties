export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=LES_HIT precision=double config=prod -j4
fi

NNODEX=${NNODEX:-1}
NNODEY=${NNODEY:-1}
NNODEZ=${NNODEZ:-1}
NNODE=$(($NNODEX * $NNODEY * $NNODEZ))
BPDX=${BPDX:-4}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-${BPDX}} #${BPDZ:-32}

NU=${NU:-0.005}

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cp ${DIR}/targetHIT/re60/meanTarget.dat  ${BASEPATH}${RUNFOLDER}/
cp ${DIR}/targetHIT/re60/kdeTarget.dat   ${BASEPATH}${RUNFOLDER}/
cp ${DIR}/targetHIT/re60/scaleTarget.dat ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
./simulation -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -extentx 6.2831 -dump2D 0 -dump3D 0 -tdump 0.0 -BC_x periodic -BC_y periodic -BC_z periodic -initCond HITurbulence -spectralIC fromFile -spectralICFile ../meanTarget.dat -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz ${NNODEZ} -CFL 0.1 -tend 50 -sgs RLSM -compute-dissipation 1 -spectralForcing 1 -analysis HIT -tAnalysis 0.1 -nu ${NU}
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 0 "
SETTINGS+=" --workerProcessesPerEnv ${NNODE} "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
