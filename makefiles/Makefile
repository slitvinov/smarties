hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
config ?= prod
extra ?=

ifeq "$(findstring daint,$(shell hostname))" "daint"
CXX=CC
LD=CC
else
CXX=mpic++
LD=mpic++
endif

isICPC=$(shell $(CXX) --version | grep -ci "icpc" )
isLLVM=$(shell $(CXX) --version | grep -ci "llvm" )
ifeq "$(isICPC)" "0"
ifeq "$(isLLVM)" "0"
compiler=gnu
else
compiler=llvm
endif
else
compiler=intel
endif

CPPFLAGS=-DSMARTIES
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

exp ?= GAUS
ifeq "$(exp)" "GAUS"
CPPFLAGS+= -DADV_GAUS
endif
nexp ?= 1
CPPFLAGS+= -DNEXPERTS=$(nexp)

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
endif
include make.gcc.flags
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
	MemorySharing.o \
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

.DEFAULT_GOAL := rl

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

lib:
	+make -C ../apps/$(app)/ clean
	+make -C ../apps/$(app)/ CC=$(CXX) CXX=$(CXX) compiler=$(compiler) config=$(config) precision=$(app_precision)

else

# app is compiled independently and uses communication through sockets
# this is just for testing:
lib: ext_app_test.o
	ar rs libsimulation.a $^

endif
###############################################################################

rl: $(OBJ_FILES) lib
	$(LD) $(OBJ_FILES) $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra)  $(CPPFLAGS) -c -MD $<

vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o *.so *.s *.d libsimulation.a rl
