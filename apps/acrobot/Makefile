CC = g++

SRC_DIR = ./
COM_DIR = ../../source/Util
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS+=-std=c++11 -O3 -I$(COM_DIR)
OBJ_FILES = 	acrobot.o Communicator.o

.DEFAULT_GOAL := acrobot

acrobot: $(OBJ_FILES)
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
	rm -f rl
