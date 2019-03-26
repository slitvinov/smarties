mainApp ?= testRL
.DEFAULT_GOAL := app

app:
	make -C ../../../CubismUP_3D/makefiles/ -j4 precision=$(precision) hdf5=true \
    config=$(config) smarties=true CXX=$(CXX) compiler=$(compiler) accfft=false $(mainApp)
	cp ../../../CubismUP_3D/makefiles/libsimulation.a ../../makefiles/
	rm ../../../CubismUP_3D/makefiles/libsimulation.a

clean:
	make -C ../../../CubismUP_3D/makefiles/ clean


# tell Cubism to compile itself for leadFollow
# #make -C ../../../CubismUP_2D/makefiles/ -j4 precision=single hdf5=off \
# # config=$(config) smarties=true CXX=$(CXX) compiler=$(compiler) smartCyl
