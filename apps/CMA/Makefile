CC = g++

#CPPFLAGS+=-std=c++11 -O3 -fopenmp
CPPFLAGS+=-std=c++11 -O0 -g -fopenmp

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = 	engine_cmaes.o communicator.o cmaes.o

.DEFAULT_GOAL := engine_cmaes

engine_cmaes: $(OBJ_FILES)
	$(CC) $(LIBS) -fopenmp $^ -o $@

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
