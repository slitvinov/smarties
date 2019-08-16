mainApp ?= RL_HIT
.DEFAULT_GOAL := app

app:
	make -C ../../../CubismUP_3D/makefiles/ -j4 precision=$(precision) hdf5=true \
    config=$(config) smarties=true MPICXX=$(CXX) compiler=$(compiler) \
    accfft=false $(mainApp)
	cp ../../../CubismUP_3D/makefiles/libsimulation.a ../../makefiles/
	rm ../../../CubismUP_3D/makefiles/libsimulation.a

clean:
	make -C ../../../CubismUP_3D/makefiles/ clean


