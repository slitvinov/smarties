export INTERNALAPP=false
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../apps/test_cpp_cart_pole
fi

cp ../apps/test_cpp_cart_pole/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/test_cpp_cart_pole/cart-pole ${BASEPATH}${RUNFOLDER}/
