export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../apps/test_twompi_cart_poles
fi

cp ../apps/test_twompi_cart_poles/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/test_twompi_cart_poles/cart-pole ${BASEPATH}${RUNFOLDER}/
