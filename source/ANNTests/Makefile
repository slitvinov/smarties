CC = g++
CPPFLAGS += -g -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated -std=c++11

CPPFLAGS += -O3 -msse3 -DNDEBUG -DBOOST_UBLAS_NDEBUG

CPPFLAGS+= -I/usr/local/Cellar/vtk5/5.10.1/include/vtk-5.10/ -I/opt/local/include/openmpi -I/usr/include/openblas -I/home/novatig/armadillo/usr/include -I./

LIBS += \
	-L/opt/local/lib -L/usr/lib64 \
	-L/home/novatig/armadillo/usr/lib64 -L/usr/lib/openblas-base  \
	-larmadillo -lopenblas -llapack -fopenmp

LIBS += -lstdc++

DIRS = ./
OBJ_FILES = anntest.o Network.o WaveletNet.o rng.o
						
CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif
    
rl: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) $(CPPFLAGSOPT) -c $< -o $@
	
%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -MD -c $<
    
vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f rl