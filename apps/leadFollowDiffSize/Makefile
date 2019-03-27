CXX ?= mpic++
config ?= prod
compiler ?= gnu
precision ?= single
.DEFAULT_GOAL := app

app:
	make -C ../commonCUP2D CXX=$(CXX) config=$(config) compiler=$(compiler) precision=$(precision) mainApp=leadFollow

clean:
	make -C ../commonCUP2D clean
