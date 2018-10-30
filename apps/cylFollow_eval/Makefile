CXX ?= mpic++
config ?= prod
compiler ?= gnu
.DEFAULT_GOAL := app

app:
	# tell Cubism to compile itself for leadFollow
	make -C ../../../CubismUP_2D/makefiles/ -j4 precision=double hdf5=on \
    config=$(config) smarties=true CXX=$(CXX) compiler=$(compiler) leadFollow
	cp ../../../CubismUP_2D/makefiles/libsimulation.a ../../makefiles/
	rm ../../../CubismUP_2D/makefiles/libsimulation.a
clean:
	make -C ../../../CubismUP_2D/makefiles/ clean
