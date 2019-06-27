SHELL := /bin/bash
CURPWD := $(shell pwd)/
UNAME_S := $(shell uname -s)
config ?= prod
extra ?=

CPPFLAGS=
LIBS=

ifeq ($(UNAME_S),Darwin)
include make.macos
endif

ifeq "$(findstring eu-,$(shell hostname))" "eu-"
include make.euler
endif
ifeq "$(findstring euler,$(shell hostname))" "euler"
include make.euler
endif

ifeq "$(findstring daint,$(shell hostname))" "daint"
include make.daint_srun
endif

ifeq "$(findstring falcon,$(shell hostname))" "falcon"
include make.falcon
endif
ifeq "$(findstring panda,$(shell hostname))" "panda"
include make.falcon
endif

ifeq "$(MPICXX)" ""
CPPFLAGS+= -I$(CURPWD)../extern/build/mpich-3.3/include/
LIBS+= -L$(CURPWD)../extern/build/mpich-3.3/lib/
MPICXX=$(CURPWD)../extern/build/mpich-3.3/bin/mpic++
LD=$(CURPWD)../extern/build/mpich-3.3/bin/mpic++
endif

ifeq "$(shell $(MPICXX) --version | grep -ci 'icpc')" "0"
ifeq "$(shell $(MPICXX) --version | grep -ci 'llvm')" "0"
compiler=gnu
else
compiler=llvm
endif
else
compiler=intel
endif

testdiff ?= off
ifeq "$(testdiff)" "on"
CPPFLAGS+= -D__CHECK_DIFF
endif

precision ?= double
app_precision ?= $(precision)
ifeq "$(precision)" "single"
CPPFLAGS+= -DSINGLE_PREC
endif

app ?= ext_app_test

ifeq "$(compiler)" "intel"
include make.icc.flags
else
ifeq "$(compiler)" "llvm"
include make.llvm.flags
else
include make.gcc.flags
endif
endif

CPPFLAGS+= -I../extern/CLI11/include/

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = \
	RACER.o \
	Learner_pytorch.o \
	Learner_approximator.o \
	Sequences.o \
	Learner.o \
	Sampling.o \
	Collector.o \
	Math_tests.o \
	MemoryBuffer.o \
	DataCoordinator.o \
	MemoryProcessing.o \
	Builder.o \
	Network.o \
	Optimizer.o \
	CMA_Optimizer.o \
	StatsTracker.o \
	Communicator.o \
	Approximator.o \
	Launcher.o \
	Settings.o \
	Profiler.o \
	Master.o \
	Worker.o \
	main.o
	#PPO.o \
	#RETPG.o \
	#DPG.o \
	#NAF.o \
	#DQN.o \
	#ACER.o \
	#CMALearner.o \

CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

.DEFAULT_GOAL := all

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

LIBS+= -L. -lsimulation
CPPFLAGS+= -I../source

############### Compile application as static library if needed ###############
ifneq "$(app)" "ext_app_test"

# let's check if app requires some flag / lib
ifneq ("$(wildcard ../apps/$(app)/make.flags)","")
export app_precision $(app_precision)
include ../apps/$(app)/make.flags
endif
CPPFLAGS+= -DINTERNALAPP # app is linked directly to smarties

env:
	+make -C ../apps/$(app)/ clean
	+make -C ../apps/$(app)/ CC=$(MPICXX) CXX=$(MPICXX) compiler=$(compiler) \
	config=$(config) precision=$(app_precision)

else

# app is compiled independently and uses communication through sockets
# this is just for testing:
env: ext_app_test.o
	ar rs libsimulation.a $^

endif
###############################################################################

###################### Compile Communicator Library ###########################
OBJCOMM=../source/Communicators/Communicator.cpp
LIBNAME=librlcomm.so
OBJNAME=CommunicatorLib.o

test: $(OBJ_FILES) testempty.o
	g++ $(CPPFLAGS) -DSMARTIES_LIB -fPIC -c $(OBJCOMM) -o $(OBJNAME)
	$(LD) $(OBJNAME) testempty.o -o $@

lib:
	g++ $(CPPFLAGS) -DSMARTIES_LIB -fPIC -c $(OBJCOMM) -o $(OBJNAME)
	ar rs librlcomm.a $(OBJNAME)
	#g++ -shared -Wl,-soname,$(LIBNAME) -o $(LIBNAME) $(OBJNAME)
	#g++ -shared -o $(LIBNAME) $(OBJNAME)

exec: $(OBJ_FILES) env
	$(LD) $(OBJ_FILES) $(LIBS) -o rl

all: lib exec

%.o: %.cpp
	$(MPICXX) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(MPICXX) $(extra)  $(CPPFLAGS) -c -MD $<

vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o *.so *.s *.d libsimulation.a librlcomm.* rl
