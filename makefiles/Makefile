#CC = /opt/mpich/bin/mpicxx
CC = g++
#CC = CC
CPPFLAGS+=-std=c++11 -mavx 
#CPPFLAGS+=-Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated

viz ?= 0
config ?= “prod”
extra ?=

ifeq "$(viz)" "1"
CPPFLAGS+= -D_RL_VIZ
OPENGL_LIB=-framework OpenGL -framework GLUT /opt/local/lib/libGLEW.a
endif

ifeq "$(config)" "debug"
CPPFLAGS+= -O0 -g
#CPPFLAGS+= -dynamic -fsanitize=address
else
CPPFLAGS += -O3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
endif

ifeq "$(CC)" "icc"
CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
endif

CPPFLAGS+= -I/opt/local/include/openmpi

LIBS += -L$(MPI_ROOT)/lib -L/opt/local/lib -L/usr/lib64 -lstdc++ -fopenmp
LIBS+= -L/cluster/apps/openmpi/1.6.2/x86_64/gcc_4.8.2/lib -lmpi -lmpi_cxx

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = 	ArgumentParser.o main.o ObjectFactory.o Profiler.o rng.o  StateAction.o \
		Agent.o ExternalAgent.o SmartySelfAvoider.o SmartySwarmer.o \
		oldEnvironment.o ExternalEnvironment.o SelfAvoidEnvironment.o SwarmEnvironment.o TwoFishEnvironment.o \
		Explorer.o NFQ.o QLearning.o Sarsa.o \
		MultiTable.o NFQApproximator.o \
		Scheduler.o Saver.o util.o\
		Layers.o LSTMNet.o Network.o Optimizer.o 
						
CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

	
.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif
    
rl: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@
	
LSTM: anntest.o ArgumentParser.o Profiler.o Layers.o Network.o Optimizer.o LSTMNet.o
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
