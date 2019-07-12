if [[ "${SKIPMAKE}" != "true" ]] ; then
rm ../makefiles/libsimulation.a
make -C ../makefiles/ app=commonCUP3D -j4
fi

NNODEX=${NNODEX:-4}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))
BPDX=${BPDX:-8}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/2))} #${BPDZ:-32}
NU=${NU:-0.0004} # RE = halfHeight * U / nu = 2500

# notes: eta_y is a param of mesh_density, implement noisy init cond
cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
./simulation -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -extentx 6.2831853072 -extenty 2 -extentz 4.7123889804 -mesh_density_y SinusoidalDensity -eta_y 0.5 -useSolver hypre -dump2D 1 -dump3D 1 -tdump 1 -BC_x periodic -BC_y wall -BC_z periodic -initCond channel -nslices 2 -slice1_direction 1 -slice2_direction 2 -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1 -CFL 0.1 -tend 10 -uMax_forced 1 -compute-dissipation 0 -nu ${NU}
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh "
SETTINGS+=" --nStepPappSett 0 "
SETTINGS+=" --workersPerEnv ${NNODE} "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
