#cp ../apps/glider_CUP2D/runArguments* ${BASEPATH}${RUNFOLDER}/

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=glider_CUP2D  -j4
fi

# notes: eta_y is a param of mesh_density, implement noisy init cond
#cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
#../launchSim.sh -poissonType cosine -muteAll 1 -bpdx 32 -bpdy 32 -tdump 1 -nu 0.000025 -tend 0 -shapes 'glider_semiAxisX=.125_semiAxisY=.025_rhoS=1.01_xpos=.5_ypos=.5_bFixed=1_bForced=0'
#EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/runArguments00.sh
../launchSim.sh -poissonType cosine -muteAll 1 -bpdx 32 -bpdy 32 -tdump 1 -nu 0.0002 -tend 0 -shapes 'glider_semiAxisX=.125_semiAxisY=.025_rhoS=2_xpos=.6_ypos=.4_bFixed=1_bForced=0'
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh"
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
