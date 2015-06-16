CC = g++

CPPFLAGS+=-g -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated -std=c++11

OPENMP_FLAG= -openmp

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

OPENMP_FLAG= -fopenmp -ltbb -ltbbmalloc
CPPFLAGSOPT+= -msse3  -lm -lc -lstdc++
endif
CPPFLAGS+= -I/opt/local/include/openmpi  -I/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140416oss/include
CPPFLAGS+= -I../MRAGapps/IF2D_ROCKS/source/mattia-RL/
CPPFLAGS+= $(OPENMP_FLAG)

LIBS += \
	$(OPENGL_LIB) \
	$(OPENMP_FLAG) \
	-L/cluster/apps/openmpi/1.6.2/x86_64/gcc_4.8.2/lib \
	-L/opt/local/lib \
    -L/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release \
	-lmpi -lmpi_cxx -lpthread -ldl

LIBS += -lstdc++ -ltbb

DIRS = $(filter-out ../source/ANNTests/ ../source/CMAUtility/ ../source/Process/, $(sort $(dir $(wildcard ../source/*/))))
OBJ_FILES = 	ArgumentParser.o ObjectFactory.o \
		Profiler.o StateAction.o \
		main.o rng.o \
		Agent.o ExternalAgent.o CartAgent.o \
		SmartySelfAvoider.o SmartySwarmer.o GlideEnvironment.o \
		ExternalEnvironment.o ArtisticEnvironment.o CartEnvironment.o SelfAvoidEnvironment.o \
		SwarmEnvironment.o QLearning.o SpeedyQLearning.o\
		MultiTable.o Saver.o \
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
