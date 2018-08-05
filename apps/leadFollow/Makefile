CC ?= mpic++
.DEFAULT_GOAL := app

app:
	# tell Cubism to compile itself for leadFollow
	make -C ../../../CubismUP_2D/makefiles/ -j4 precision=single \
    config=production cheap=true smarties=true CC=$(CC) leadFollow

clean:
	make -C ../../../CubismUP_2D/makefiles/ clean
