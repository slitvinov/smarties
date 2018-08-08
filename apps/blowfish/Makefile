CC ?= mpic++
.DEFAULT_GOAL := app

# TODO: expect user to put SMARTDIR in bashrc
# tell Cubism to compile itself for blowfish
app:
	make -C ../../../CubismUP_2D/makefiles/ -j4 precision=single \
    config=production smarties=true CC=$(CC) blowfish
	cp ../../../CubismUP_2D/makefiles/libsimulation.a ../../makefiles/

clean:
	make -C ../../../CubismUP_2D/makefiles/ clean
