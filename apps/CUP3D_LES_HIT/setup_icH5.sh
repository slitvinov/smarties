export INTERNALAPP=true
echo "OUTDATED"
exit 0

# compile executable:
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/../CubismUP_3D/makefiles rlHIT -j4
fi

# copy executable:
cp ${SMARTIES_ROOT}/../CubismUP_3D/makefiles/rlHIT ${RUNDIR}/exec

# write simulation settings files:
NNODEX=${NNODEX:-1}
NNODEY=${NNODEY:-1}
NNODEZ=${NNODEZ:-1}
NNODE=$(($NNODEX * $NNODEY * $NNODEZ))
BPDX=${BPDX:-4}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-${BPDX}} #${BPDZ:-32}
NU=${NU:-0.005}
cat <<EOF >${RUNDIR}/runArguments00.sh
./simulation -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -extentx 6.2831 \
-dump2D 0 -dump3D 0 -tdump 0.0 -BC_x periodic -BC_y periodic -BC_z periodic \
-initCond HITurbulence -spectralIC fromFile -keepMomentumConstant 1 \
-spectralICFile ../meanTarget.dat -nprocsx ${NNODEX} -nprocsy ${NNODEY} \
-nprocsz ${NNODEZ} -CFL 0.1 -tend 50 -sgs RLSM -compute-dissipation 1 \
-spectralForcing 1 -analysis HIT -tAnalysis 0.1 -nu ${NU}
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
./simulation -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -extentx ${EXTENT_x} \
-dump2D 0 -dump3D 1 -tdump 0.1 -BC_x periodic -BC_y periodic -BC_z periodic \
-icFromH5 ../$icH5File  -nprocsx ${NNODEX} -nprocsy ${NNODEY} \
-nprocsz ${NNODEZ} -CFL 0.1 -tend 50 -sgs RLSM -compute-dissipation 1 \
-analysis HIT -tAnalysis 0.1 -spectralForcing 1 -nu ${NU}
EOF

#copy target files
icH5Path=${icH5Path:-~/icGenerator/Output/hit_re90/}
icH5File=${icH5File:-downsample_4_4_4}
source ${icH5Path}/${icH5File}_settings.sh

icFile=${icH5Path}/${icH5File}.h5
if [ -f ${icFile} ]; then
  cp ${icFile} ${RUNDIR}/
  echo "Copying ${icFile} to ${RUNDIR}/"
else
  echo "${icFile} does not exist"
  exit 0
fi

EXTRASETTINGS=${icH5Path}
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cp ${DIR}/targetHIT/re60/kdeTarget.dat ${BASEPATH}${RUNFOLDER}/
cp ${DIR}/targetHIT/re60/meanTarget.dat ${BASEPATH}${RUNFOLDER}/
cp ${DIR}/targetHIT/re60/scaleTarget.dat ${BASEPATH}${RUNFOLDER}/

# write file for launch_base.sh to read app-required settings:
cat <<EOF >${RUNDIR}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 0 "
SETTINGS+=" --workerProcessesPerEnv ${NNODE} "
EOF
chmod 755 ${RUNDIR}/appSettings.sh
