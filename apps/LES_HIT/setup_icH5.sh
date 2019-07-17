if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=RL_LES_HIT precision=double config=prod -j4
fi

NNODEX=${NNODEX:-1}
NNODEY=${NNODEY:-1}
NNODEZ=${NNODEZ:-1}
NNODE=$(($NNODEX * $NNODEY * $NNODEZ))

icH5Path=${icH5Path:-~/icGenerator/Output/hit_re90/}
icH5File=${icH5File:-downsample_4_4_4}

EXTRASETTINGS=${icH5Path}
source ${icH5Path}/${icH5File}_settings.sh

icFile=${icH5Path}/${icH5File}.h5
if [ -f ${icFile} ]; then
  cp ${icFile} ${BASEPATH}/${RUNFOLDER}/
  echo "Copying ${icFile} to ${BASEPATH}/${RUNFOLDER}/"
else
  echo "${icFile} does not exist"
  exit 0
fi

NU=${NU:-0.005}

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cp ${DIR}/targetHIT/re60/kdeTarget.dat ${BASEPATH}${RUNFOLDER}/
cp ${DIR}/targetHIT/re60/meanTarget.dat ${BASEPATH}${RUNFOLDER}/
cp ${DIR}/targetHIT/re60/scaleTarget.dat ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
./simulation -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -extentx ${EXTENT_x} -dump2D 0 -dump3D 1 -tdump 0.1 -BC_x periodic -BC_y periodic -BC_z periodic -icFromH5 ../$icH5File  -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz ${NNODEZ} -CFL 0.1 -tend 50 -sgs RLSM -compute-dissipation 1 -analysis HIT -tAnalysis 0.1 -spectralForcing 1 -nu ${NU}
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 0 "
SETTINGS+=" --workersPerEnv ${NNODE} "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
