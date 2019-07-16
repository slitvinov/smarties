CXX ?= mpic++
config ?= prod
compiler ?= gnu
precision ?= double
.DEFAULT_GOAL := app

app:
	make -C ../commonCUP3D -j4 precision=$(precision) config=$(config) \
    CXX=$(CXX) compiler=$(compiler) mainApp=RL_HIT

clean:
	make -C ../commonCUP3D clean


