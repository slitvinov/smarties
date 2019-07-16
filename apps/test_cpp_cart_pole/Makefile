CC = g++

OBJ_FILES = cart-pole.o
SRC_DIR = ./
COM_DIR = ../../source/
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS+= -std=c++14 -Wall -O2 -I$(COM_DIR)
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

.DEFAULT_GOAL := cart-pole

cart-pole: $(OBJ_FILES)
	$(CC) -L../../makefiles/ -o $@ $^ -lsmarties

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -MD -c $<

clean:
	rm -f *.o *.s *.d rl
