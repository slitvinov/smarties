CXX ?= mpic++
compiler ?= gnu
.DEFAULT_GOAL := app

app:
	# tell Cubism to compile itself for blowfish
	make -C ../../../CubismUP_2D/makefiles/ -j4 precision=single \
    config=production smarties=true CXX=$(CXX) compiler=$(compiler) blowfish
	cp ../../../CubismUP_2D/makefiles/libsimulation.a ../../makefiles/

clean:
	make -C ../../../CubismUP_2D/makefiles/ clean
