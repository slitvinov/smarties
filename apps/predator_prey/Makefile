CC = g++

#OBJ_FILES = pp.o Communicator.o
OBJ_FILES = twoPredOnePrey.o Communicator.o
SRC_DIR = ./
COM_DIR = ../../source/Communicators/
VPATH := $(SRC_DIR):$(COM_DIR)
#CPPFLAGS+=-std=c++11 -Wall -O0 -I$(COM_DIR)
CPPFLAGS+=-std=c++11 -Wall -O2 -I$(COM_DIR)


CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

CPPFLAGS+= -I/usr/include/python2.7
#-I/usr/local/Cellar/python/3.6.1/Frameworks/Python.framework/Versions/3.6/include/python3.6m/
#LIBS = -L/usr/local/Cellar/python/3.6.1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/config-3.6m-darwin/ -lpython3.6m
LIBS += -lpython2.7

#CPPFLAGS+= -fsanitize=address
#LIBS += -fsanitize=address

.DEFAULT_GOAL := pp

pp: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f pp
