hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
viz ?= 0
config ?= prod
extra ?=

CPPFLAGS= -std=c++11 -D__Smarties_ -fopenmp
LIBS= -fopenmp -lstdc++ -lm -dynamic -ldl -lc -lgomp

ifeq ($(UNAME_S),Darwin)
include make.macos
endif

ifeq "$(findstring eu-,$(shell hostname))" "eu-"
include make.euler
endif
ifeq "$(findstring euler,$(shell hostname))" "euler"
include make.euler
endif

ifeq "$(findstring brutus,$(shell hostname))" "brutus"
include make.brutus
endif

ifeq "$(findstring daint,$(shell hostname))" "daint"
mpi_lib ?= default
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
include make.panda
endif

CPPFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CPPFLAGS+= -Wno-cpp -Wno-unused-parameter # does not make sense with function overloading
#CPPFLAGS+= -Wno-maybe-uninitialized #to switch on and check that only triggered by std random
#CPPFLAGS+= -Wno-float-equal #sometimes you just know, do you know what I am saying?
CPPFLAGS+= -Wno-unknown-pragmas
#CPPFLAGS+= -D__CHECK_DIFF 

acer ?= relax
ifeq "$(acer)" "relax"
CPPFLAGS += -DACER_RELAX
else
ifneq "$(acer)" "full"
#$(error acer flag is not recognized: safe, relax or full)
endif #!full
endif #relax

sortseq ?= off
ifeq "$(sortseq)" "on"
CPPFLAGS+= -DRESORT_SEQS
endif

tabc ?= off
ifeq "$(tabc)" "on"
CPPFLAGS+= -DRACER_TABC
endif

importance ?= off
ifeq "$(importance)" "on"
CPPFLAGS+= -DimportanceSampling
endif

acertrick ?= 0
ifeq "$(acertrick)" "1"
CPPFLAGS+= -DRACER_ACERTRICK
endif

raceskip ?= 1
CPPFLAGS+= -DRACER_SKIP=$(raceskip)

racefrwd ?= 0
CPPFLAGS+= -DRACER_FORWARD=$(racefrwd)

raceadve ?= 0
ifeq "$(raceadve)" "1"
CPPFLAGS+= -DRACER_ONESTEPADV
endif

raceback ?= 1
ifeq "$(raceback)" "1"
CPPFLAGS+= -DRACER_BACKWARD
endif

exp ?= GAUS
ifeq "$(exp)" "GAUS"
CPPFLAGS+= -DADV_GAUS
endif

nexp ?= 1
CPPFLAGS+= -DNEXPERTS=$(nexp)

entropy ?= off
ifeq "$(entropy)" "on"
CPPFLAGS+= -D__EntropySGD
endif

testdiff ?= off
ifeq "$(testdiff)" "on"
CPPFLAGS+= -D__CHECK_DIFF
endif

app ?= test
ifeq "$(app)" "cubism"
CPPFLAGS+= -D__Cubism3D
endif

ifeq "$(config)" "debug"
CPPFLAGS+= -g -O1
endif
ifeq "$(config)" "segf"
CPPFLAGS+= -g -O0 -ffast-math
CPPFLAGS+= -fsanitize=address
LIBS += -fsanitize=address
endif
ifeq "$(config)" "nan"
CPPFLAGS+= -g -O0 -ffast-math
CPPFLAGS+= -fsanitize=undefined
LIBS += -fsanitize=undefined
endif
ifeq "$(config)" "prod"
CPPFLAGS += -DNDEBUG -O2 -ffast-math
endif
ifeq "$(config)" "fit"
CPPFLAGS += -DNDEBUG -O3 -ffast-math
endif

CPPFLAGS+= -fstrict-aliasing -march=native -mtune=native
#CPPFLAGS+= -ftree-vectorize #-ftree-vectorizer-verbose=2 #-DNDEBUG
CPPFLAGS+= -fmerge-all-constants #-fassociative-math
#CPPFLAGS+= -funsafe-loop-optimizations
#CPPFLAGS+= -fprofile-use
CPPFLAGS+= -falign-functions=32 #-DBOOST_UBLAS_NDEBUG

#ifeq "$(CC)" "icc"
#CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
#endif

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = ArgParser.o main.o ObjectFactory.o Communicator.o \
		Environment.o TwoActFishEnvironment.o Learner.o NAF.o NFQ.o DPG.o \
		Learner_onPolicy.o MemoryBuffer.o Learner_offPolicy.o \
		Scheduler.o Network.o Math_tests.o Approximator.o AtariEnvironment.o

#ExternalAgent.o TestEnvironment.o CMAEnvironment.o CartEnvironment.o GliderEnvironment.o
#TwoFishEnvironment.o HingedFishEnvironment.o BlowfishEnvironment.o alebotEnvironment.o
#NewFishEnvironment.o DeadFishEnvironment.o AcrobotEnvironment.o openAICartEnvironment.o


CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

ifeq "$(app)" "cubism"
lib:
	make -C ../../CubismUP_3D/makefiles/ clean
	#make -C ../../CubismUP_3D/makefiles/ train=true librl -j
	make -C ../../CubismUP_3D/makefiles/ librl -j
else
LIBAPP= -L. -lsimulation

lib: ext_app_test.o
	ar rs libsimulation.a $^
endif

rl: $(OBJ_FILES) lib
	$(CC) $(OBJ_FILES) $(LIBAPP) $(LIBS) -o $@

LSTM: anntest.o ArgumentParser.o Profiler.o Links.o Layers.o Network.o Optimizer.o LSTMNet.o
	g++ $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c -MD $<


vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f libsimulation.a rl
