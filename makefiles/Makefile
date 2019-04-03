hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
config ?= prod
extra ?=
ifeq ($(INTEL_LICENSE_FILE),)
compiler ?= gnu
else
compiler ?= intel
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
include make.gcc.flags
endif

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = \
	ArgParser.o \
	Profiler.o \
	ObjectFactory.o \
	Communicator.o \
	Communicator_internal.o \
	AtariEnvironment.o \
	Environment.o \
	Network.o \
	Optimizer.o \
	CMA_Optimizer_MPI.o \
	Aggregator.o \
	Encapsulator.o \
	Approximator.o \
	Sequences.o \
	MemoryBuffer.o \
	MemoryProcessing.o \
	MemorySharing.o \
	StatsTracker.o \
	Collector.o \
	Sampling.o \
	Learner_approximator.o \
	Learner_pytorch.o \
	Learner.o \
	StateAction.o \
	Math_tests.o \
	Scheduler.o \
	RACER.o \
	VRACER.o \
	main.o
	#PPO.o \
	#RETPG.o \
	#DPG.o \
	#NAF.o \
	#DQN.o \
	#ACER.o \
	CMALearner.o \

CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

LIBS+= -L. -lsimulation

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
	+make -C ../apps/$(app)/ CC=$(CC) CXX=$(CC) compiler=$(compiler) config=$(config) precision=$(app_precision)

else

# app is compiled independently and uses communication through sockets
# this is just for testing:
lib: ext_app_test.o
	ar rs libsimulation.a $^

endif

rl: $(OBJ_FILES) lib
	$(CC) $(OBJ_FILES) $(LIBS) -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c -MD $<

vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o *.so *.s *.d libsimulation.a rl
