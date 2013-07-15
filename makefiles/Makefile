CC = g++
CPPFLAGS+= -g -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated -std=c++11

OPENMP_FLAG= -openmp

viz ?= 0
config ?= "debug"

ifeq "$(viz)" "1"
CPPFLAGS+= -D_RL_VIZ
OPENGL_LIB=-framework OpenGL -framework GLUT /opt/local/lib/libGLEW.a
endif

ifeq "$(config)" "debug"
CPPFLAGS+= -O0
else ifeq "$(config)" "release"
CPPFLAGS += -O3 -msse3 -DNDEBUG -DBOOST_UBLAS_NDEBUG
else
CPPFLAGS += -O1 -DNDEBUG -DBOOST_UBLAS_NDEBUG
endif

ifeq "$(CC)" "icc"
CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
else
OPENMP_FLAG= -fopenmp
CPPFLAGSOPT+= -msse3
endif

CPPFLAGS+= $(OPENMP_FLAG)
CPPFLAGS+= -I$(BOOST_INCLUDEDIR)

LIBS += \
	$(OPENGL_LIB) \
	$(OPENMP_FLAG)

LIBS += -lstdc++

DIRS = $(filter-out ../source/ANNTests/ ../source/CMAUtility/, $(sort $(dir $(wildcard ../source/*/))))
CPP_FILES = $(wildcard $(DIRS:%/=%/*.cpp))
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
	
.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif
    
rl: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC)  $(CPPFLAGS) $(CPPFLAGSOPT) -c $< -o $@
	
%.d: %.cpp
	$(CC)  $(CPPFLAGS) -MD -c $<
    
vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f rl
