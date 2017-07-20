hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
viz ?= 0
config ?= prod
extra ?=
app ?= test
#cvar ?= simple
cvar ?= off
acer ?= full
testdiff ?= no
entropy ?= off
auxtask ?= off
importance ?= off
lambda ?= off
tabc ?= off
sortseq ?= off
mpi_lib ?= default
sampseq ?= off
target ?= on

CPPFLAGS+= -std=c++11 -D__Smarties_ -fopenmp
LIBS += -fopenmp -lstdc++ -lm

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
include make.panda
endif

CPPFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic -Woverloaded-virtual
CPPFLAGS+= -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CPPFLAGS+= -Wno-unused-parameter # does not make sense with function overloading
CPPFLAGS+= -Wno-maybe-uninitialized #to switch on and check that only triggered by std random
CPPFLAGS+= -Wno-float-equal #sometimes you just know, do you know what I am saying?

CPPFLAGS += -fstrict-aliasing -march=native -mtune=native
CPPFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=0 #-DNDEBUG
#CPPFLAGS += -funsafe-math-optimizations
CPPFLAGS += -falign-functions=32 #-DBOOST_UBLAS_NDEBUG

ifeq "$(acer)" "safe"
CPPFLAGS += -DACER_SAFE
else
ifeq "$(acer)" "relax"
CPPFLAGS += -DACER_RELAX
else
ifneq "$(acer)" "full"
#$(error acer flag is not recognized: safe, relax or full)
endif #!full
endif #relax
endif #safe

ifeq "$(target)" "off"
CPPFLAGS+= -DNOTARGETNET
endif
ifeq "$(sampseq)" "on"
CPPFLAGS+= -DFORCESAMPSEQ
endif
ifeq "$(sortseq)" "on"
CPPFLAGS+= -DRESORT_SEQS
endif
ifeq "$(tabc)" "on"
CPPFLAGS+= -DACER_TABC
endif
ifeq "$(importance)" "on"
CPPFLAGS+= -DIMPORTSAMPLE
endif
ifeq "$(lambda)" "on"
CPPFLAGS+= -DCUT_LAMBDA
endif
ifeq "$(auxtask)" "on"
CPPFLAGS+= -DFEAT_CONTROL
endif
ifeq "$(entropy)" "on"
CPPFLAGS+= -D__EntropySGD
endif
ifeq "$(cvar)" "on"
CPPFLAGS+= -DACER_PENALIZER
endif
ifeq "$(testdiff)" "yes"
CPPFLAGS+= -D__CHECK_DIFF
endif

ifeq "$(app)" "cubism"
CPPFLAGS+= -D__Cubism3D
endif
ifeq "$(config)" "debug"
CPPFLAGS+= -g -O0
endif
ifeq "$(config)" "segf"
CPPFLAGS+= -g# -O2 -ffast-math
CPPFLAGS+= -fsanitize=address
LIBS += -fsanitize=address
endif
ifeq "$(config)" "prod"
CPPFLAGS += -DNDEBUG -O2 -ffast-math
endif
ifeq "$(config)" "fit"
CPPFLAGS += -DNDEBUG -O3 -ffast-math
endif

#ifeq "$(CC)" "icc"
#CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
#endif

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = ArgumentParser.o main.o ObjectFactory.o Communicator.o \
		Environment.o TestEnvironment.o CMAEnvironment.o CartEnvironment.o \
		TwoActFishEnvironment.o GliderEnvironment.o TwoFishEnvironment.o \
		HingedFishEnvironment.o BlowfishEnvironment.o alebotEnvironment.o \
		NewFishEnvironment.o DeadFishEnvironment.o AcrobotEnvironment.o \
		openAICartEnvironment.o \
		Learner.o RACER.o DACER.o NAF.o NFQ.o DPG.o GAE.o Transitions.o \
		Scheduler.o  Network.o Optimizer.o Math_tests.o Learner_utils.o \
		Learner_onPolicy.o
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
