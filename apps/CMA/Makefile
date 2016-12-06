CPP = g++
CC = gcc

WARNINGS =  -Wall -Wextra -Wredundant-decls -Wshadow -Woverloaded-virtual -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wmissing-declarations

#CPPFLAGS+=-std=c++11 -Wall -O0 # -fopenmp
CPPFLAGS+=-std=c++11 -O2 -g -fopenmp $(WARNINGS)# -fopenmp
CFLAGS+= -O2 -g $(WARNINGS)

OBJ_FILES_PP = engine_cmaes.o communicator.o
OBJ_FILES =    cmaes.o

.DEFAULT_GOAL := engine_cmaes

engine_cmaes: $(OBJ_FILES) $(OBJ_FILES_PP)
	$(CPP) $(LIBS) -fopenmp $^ -o $@

engine_cmaes.o: engine_cmaes.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@
    
communicator.o: communicator.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@
    
cmaes.o: cmaes.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f engine_cmaes
