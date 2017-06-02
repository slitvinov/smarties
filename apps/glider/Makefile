CC = g++
smart ?= 1

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LD=g++-7
CC=g++-7
endif

ifeq "$(smart)" "0"
OBJ_FILES =  glider.o
else
CPPFLAGS+= -D__SMARTIES_
OBJ_FILES = glider.o Communicator.o
endif

SRC_DIR = ./
COM_DIR = ../../source/
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS+=-std=c++11 -O2 -I$(COM_DIR)

.DEFAULT_GOAL := glider

glider: $(OBJ_FILES)
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
	rm -f glider
