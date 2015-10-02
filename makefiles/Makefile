CC = /opt/mpich/bin/mpicxx
CPPFLAGS+=-g -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated -std=c++11

OPENMP_FLAG=# -openmp

viz ?= 0
config ?= "debug"
extra ?=

ifeq "$(viz)" "1"
CPPFLAGS+= -D_RL_VIZ
OPENGL_LIB=-framework OpenGL -framework GLUT /opt/local/lib/libGLEW.a
endif

ifeq "$(config)" "debug"
CPPFLAGS+= -O0
else
CPPFLAGS += -O3 -msse3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
endif

ifeq "$(CC)" "icc"
CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
else

OPENMP_FLAG= -fopenmp
CPPFLAGSOPT+= -msse3
endif

CPPFLAGS+= -I/usr/local/Cellar/vtk5/5.10.1/include/vtk-5.10/ -I../MRAGapps/source -I../MRAG/source  -I/opt/local/include/openmpi -I/usr/include/openblas -I/home/novatig/armadillo/usr/include

CPPFLAGS+= -I../MRAGapps/IF2D_ROCKS/source/mattia-RL/
CPPFLAGS+= $(OPENMP_FLAG)

LIBS += \
	$(OPENGL_LIB) \
	$(OPENMP_FLAG) \
	-L$(MPI_ROOT)/lib \
	-L/opt/local/lib -L/usr/lib64 \
	-L/home/novatig/armadillo/usr/lib64 -L/usr/lib/openblas-base  \
	-larmadillo -lopenblas -llapack
	#-lmpi -lmpi_cxx 

LIBS += -lstdc++

DIRS = $(filter-out ../source/ANNTests/ ../source/CMAUtility/ ../source/Process/, $(sort $(dir $(wildcard ../source/*/))))
OBJ_FILES = 	ArgumentParser.o ObjectFactory.o \
		Profiler.o StateAction.o \
		main.o rng.o \
		Agent.o ExternalAgent.o CartAgent.o \
		SmartySelfAvoider.o SmartySwarmer.o GlideEnvironment.o \
		ExternalEnvironment.o ArtisticEnvironment.o CartEnvironment.o SelfAvoidEnvironment.o \
		SwarmEnvironment.o QLearning.o SpeedyQLearning.o\
		MultiTable.o Saver.o ALearning.o \
		Scheduler.o Sarsa.o HuntEnvironment.o \
		NFQApproximator.o NFQ.o \
		ANNApproximator.o Network.o WaveletNet.o 
						
CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

	
.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif
    
rl: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@
	
analyse: analyse.o MultiTable.o StateAction.o
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
