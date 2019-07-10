export INTERNALAPP=false

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../apps/test_two_cart_poles
fi

cp ../apps/test_two_cart_poles/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/test_two_cart_poles/cart-pole ${BASEPATH}${RUNFOLDER}/

