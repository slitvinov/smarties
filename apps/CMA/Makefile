#UNAME_S := $(shell uname -s)
#ifeq ($(UNAME_S),Darwin)
#CPP = g++-6
#CC = gcc-6
#else
CPP = g++
CC = gcc
#endif

SRC_DIR = ./
COM_DIR = ../../source/Util
VPATH := $(SRC_DIR):$(COM_DIR)
WARNINGS = -Wall -Wextra -Wredundant-decls -Wshadow -Woverloaded-virtual
WARNINGS+= -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op
WARNINGS+= -Wmissing-declarations

#CPPFLAGS+=-std=c++11 -Wall -O0 # -fopenmp
CPPFLAGS+= -std=c++11 -O2 $(WARNINGS) -I$(COM_DIR)# -fopenmp

OBJ_FILES_PP = engine_cmaes.o Communicator.o
OBJ_FILES =    cmaes.o

.DEFAULT_GOAL := engine_cmaes

engine_cmaes: $(OBJ_FILES) $(OBJ_FILES_PP)
	$(CPP) $(LIBS) $^ -o $@ #-fopenmp

engine_cmaes.o: engine_cmaes.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

communicator.o: communicator.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@

cmaes.o: cmaes.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f engine_cmaes
