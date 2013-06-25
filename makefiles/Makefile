CC = g++
CPPFLAGS+= -g -Wno-unknown-pragmas -Wno-reorder -Wno-comment -Wno-deprecated -std=c++11

OPENMP_FLAG= -openmp

viz ?= 0
config ?= "debug"

ifeq "$(viz)" "1"
CPPFLAGS+= -D_RL_VIZ
endif

ifeq "$(config)" "debug"
CPPFLAGS+= -O0
else ifeq "$(config)" "release"
CPPFLAGS += -O3 -msse3
else
CPPFLAGS += -O1
endif

ifeq "$(CC)" "icc"
CPPFLAGSOPT+= -diag-disable remark -wd68 -xHOST -ansi-alias -fno-fnalias -fno-alias
else
OPENMP_FLAG= -fopenmp
CPPFLAGSOPT+= -msse3
endif

OPENGL_LIB=-framework OpenGL -framework GLUT /opt/local/lib/libGLEW.a

CPPFLAGS+= $(OPENMP_FLAG)

LIBS += \
	$(OPENGL_LIB) \
	$(OPENMP_FLAG)

LIBS += -lstdc++

CPP_FILES = $(wildcard ../source/*.cpp ../source/Agents/*.cpp ../source/Environments/*.cpp ../source/ANN/*.cpp)
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
	$(CC)  $(CPPFLAGS) -c $< -MD
    
vpath %.cpp ../source ../source/Agents ../source/Environments ../source/Savers ../source/ANN
vpath %.h   ../source ../source/Agents ../source/Environments ../source/Savers ../source/ANN

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f rl
