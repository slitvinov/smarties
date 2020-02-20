# compile executable, assumes that smarties and CUP2D are in the same directory:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_2D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} blowfish -j4
fi

# copy executable:
cp ${COMPILEDIR}/blowfish ${RUNDIR}/exec

# copy simulation settings files:
# (8 * 32)^2 grid
# Re ~ sqrt(3*pi * radius / deltaRho / abs(gravity) / 8)
cat <<EOF >${RUNDIR}/runArguments00.sh
-poissonType cosine -muteAll 1 -bpdx 6 -bpdy 6 -tdump 0 -nu 0.001824116 \
-tend 0 -shapes 'blowfish radius=0.2 bFixed=1'
EOF

# command line args to find app-required settings, each to be used for fixed
# number of steps so as to increase sim fidelity as training progresses
export EXTRA_LINE_ARGS=" --appSettings runArguments00.sh "

# heavy application, needs dedicated processes
export MPI_RANKS_PER_ENV=1