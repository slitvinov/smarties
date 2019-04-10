export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ../makefiles/ clean
make -C ../makefiles/ app=test_mpi_cart_pole -j4
fi
