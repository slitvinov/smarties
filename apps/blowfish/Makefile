CC ?= mpic++
.DEFAULT_GOAL := app

app:
	# tell Cubism to compile itself for blowfish
	make -C ../../../CubismUP_2D/makefiles/ -j4 smarties=true CC=$(CC) blowfish

clean:
	make -C ../../../CubismUP_2D/makefiles/ clean
