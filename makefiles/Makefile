#CC = /opt/mpich/bin/mpicxx
#CC = g++
#CC = CC
hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
viz ?= 0
config ?= prod
extra ?=
app ?= test
cvariate ?= off
acer ?= full
entropy ?= off
mpi_lib ?= default

ifeq "$(viz)" "1"
CPPFLAGS+= -D_RL_VIZ
OPENGL_LIB=-framework OpenGL -framework GLUT /opt/local/lib/libGLEW.a
endif

ifeq ($(UNAME_S),Darwin)
include make.macos
endif

ifeq "$(findstring euler,$(shell hostname))" "euler"
include make.euler
endif

ifeq "$(findstring brutus,$(shell hostname))" "brutus"
include make.brutus
endif

ifeq "$(findstring daint,$(shell hostname))" "daint"
ifeq "$(mpi_lib)" "mpich"
include make.daint_mpich # When using self-compiled MPI - for client runs with srun main app
else
include make.daint_srun # When using srun - Default
endif
endif

ifeq "$(findstring falcon,$(shell hostname))" "falcon"
include make.falcon
endif
ifeq "$(findstring panda,$(shell hostname))" "panda"
include make.falcon
endif

CPPFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow
CPPFLAGS+= -Woverloaded-virtual -std=c++11 -Wno-reorder -Wno-deprecated
CPPFLAGS+= -Wno-sign-compare -Wno-unused-value -Wno-unused-parameter -Wunused-variable
#CPPFLAGS+=-Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated

CPPFLAGS += -fstrict-aliasing -march=native -mtune=native
CPPFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=0 #-DNDEBUG
#CPPFLAGS += -funsafe-math-optimizations
CPPFLAGS += -falign-functions=32 #-DBOOST_UBLAS_NDEBUG
CPPFLAGS += -ffast-math

ifeq "$(acer)" "safe"
CPPFLAGS += -D__ACER_SAFE
else
ifeq "$(acer)" "relax"
CPPFLAGS += -D__ACER_RELAX
else
ifneq "$(acer)" "full"
#$(error acer flag is not recognized: safe, relax or full)
endif #!full
endif #relax
endif #safe

ifeq "$(entropy)" "on"
CPPFLAGS+= -D__EntropySGD
endif
ifeq "$(cvariate)" "on"
CPPFLAGS+= -D__ACER_VARIATE
endif

ifeq "$(app)" "cubism"
CPPFLAGS+= -D__Cubism3D
endif
ifeq "$(config)" "debug"
CPPFLAGS+= -g -O2
endif
ifeq "$(config)" "prod"
CPPFLAGS += -DNDEBUG -O2
endif
ifeq "$(config)" "fit"
CPPFLAGS += -DNDEBUG -O3
endif

ifeq "$(CC)" "icc"
CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
endif

CPPFLAGS+= -I/usr/local/include -fopenmp -D__Smarties_
LIBS += -L$(MPI_ROOT)/lib -L/opt/local/lib -L/usr/lib64 -lstdc++ -fopenmp

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = Communicator.o ArgumentParser.o main.o ObjectFactory.o Profiler.o  \
		Environment.o TestEnvironment.o CMAEnvironment.o CartEnvironment.o \
		TwoActFishEnvironment.o GliderEnvironment.o TwoFishEnvironment.o \
		NewFishEnvironment.o DeadFishEnvironment.o AcrobotEnvironment.o \
		Learner.o RACER.o NAF.o NFQ.o DPG.o Transitions.o \
		Scheduler.o LSTMNet.o Network.o Optimizer.o #Links.o Layers.o
#ExternalAgent.o
CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))


.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

ifeq "$(app)" "cubism"
lib:
	make -C ../../CubismUP_3D/makefiles/ clean
	make -C ../../CubismUP_3D/makefiles/ train=true librl -j
	#make -C ../../CubismUP_3D/makefiles/ librl -j
else
LIBAPP= -L. -lsimulation

lib: ext_app_test.o
	ar rs libsimulation.a $^
endif

rl: $(OBJ_FILES) lib
	$(CC) $(LIBS) $(OBJ_FILES) $(LIBAPP) -o $@

LSTM: anntest.o ArgumentParser.o Profiler.o Links.o Layers.o Network.o Optimizer.o LSTMNet.o
	g++ $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -MD -c $<

vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f libsimulation.a rl
