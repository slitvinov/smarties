export INTERNALAPP=true

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

#copy target files
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cp ${DIR}/targetHIT/re75/meanTarget.dat  ${RUNDIR}/
cp ${DIR}/targetHIT/re75/kdeTarget.dat   ${RUNDIR}/
cp ${DIR}/targetHIT/re75/scaleTarget.dat ${RUNDIR}/

# write file for launch_base.sh to read app-required settings:
cat <<EOF >${RUNDIR}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 0 "
SETTINGS+=" --workerProcessesPerEnv ${NNODE} "
EOF
chmod 755 ${RUNDIR}/appSettings.sh


