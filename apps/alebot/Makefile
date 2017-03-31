CC = g++

OBJ_FILES = 	alebot.o Communicator.o
SRC_DIR = ./
COM_DIR = ../../source
ALE = ./Arcade-Learning-Environment
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS =-std=c++11 -O3 -I$(COM_DIR) -I$(ALE)/src
LIBS += -I$(ALE)/src -I$(ALE)/src/controllers -I$(ALE)/src/os_dependent -I$(ALE)/src/environment -I$(ALE)/src/external -L$(ALE) -Wl,-rpath=$(ALE)
LDFLAGS := -lale -lz


.DEFAULT_GOAL := alebot

alebot: $(OBJ_FILES)
	$(CC) $(LIBS) $^ $(LDFLAGS) -o $@

%.o: %.cpp
	$(CC) $(extra) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra) $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f rl
