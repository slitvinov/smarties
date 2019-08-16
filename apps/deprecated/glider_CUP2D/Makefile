CXX ?= mpic++
config ?= prod
compiler ?= gnu
precision ?= single
.DEFAULT_GOAL := app

app:
	make -C ../commonCUP2D CXX=$(CXX) config=$(config) compiler=$(compiler) precision=$(precision) mainApp=glider

clean:
	make -C ../commonCUP2D clean
