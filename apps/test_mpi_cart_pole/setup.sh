export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
rm ../makefiles/libsimulation.a
make -C ../makefiles/ app=test_mpi_cart_pole -j4
fi
