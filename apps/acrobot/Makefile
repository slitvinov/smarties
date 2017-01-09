CC = g++

CPPFLAGS+=-std=c++11 -O3

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = 	acrobot.o communicator.o

.DEFAULT_GOAL := acrobot

acrobot: $(OBJ_FILES)
	$(CC) $(LIBS) $^ -o $@

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
