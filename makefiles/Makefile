#CC = /opt/mpich/bin/mpicxx
#CC = g++
#CC = CC
hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
viz ?= 0
config ?= "prod"
extra ?=

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
#include make.daint	# When using self-compiled MPI - AVOID !!
include make.daint_srun # When using srun - Default
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
CPPFLAGS+= -Wno-sign-compare -Wno-unused-value -Wno-unused-parameter
CPPFLAGS += -O2 -fstrict-aliasing -march=native -mtune=native
#CPPFLAGS+=-Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated


ifeq "$(config)" "debug"
CPPFLAGS+= -g
#else
endif
ifeq "$(config)" "prod"
CPPFLAGS += -DNDEBUG
#CPPFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=0 #-DNDEBUG
#CPPFLAGS += -funsafe-math-optimizations
#CPPFLAGS += -falign-functions=16 #-DBOOST_UBLAS_NDEBUG
#CPPFLAGS += -ffast-math
endif

ifeq "$(CC)" "icc"
CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
endif

CPPFLAGS+= -I/usr/local/include -fopenmp
LIBS += -L$(MPI_ROOT)/lib -L/opt/local/lib -L/usr/lib64 -lstdc++ -fopenmp

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = Communicator.o	ArgumentParser.o main.o ObjectFactory.o Profiler.o  \
		Environment.o CMAEnvironment.o CartEnvironment.o TwoActFishEnvironment.o TwoFishEnvironment.o NewFishEnvironment.o DeadFishEnvironment.o \
		Learner.o NAF.o NFQ.o DPG.o Transitions.o\
		Scheduler.o LSTMNet.o Network.o Optimizer.o #Links.o Layers.o
#ExternalAgent.o
CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))


.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

rl: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@

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
	rm -f rl
