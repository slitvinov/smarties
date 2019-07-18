CXX ?= g++

OBJ_FILES = cart-pole.o
SRC_DIR = ./
COM_DIR = ../../source/
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS+= -std=c++14 -Wall -O2 -I$(COM_DIR)
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
ifeq ($(CXX), c++)
#LD=g++-9
#CXX=g++-9
endif
endif

.DEFAULT_GOAL := cart-pole

cart-pole: $(OBJ_FILES)
	$(CXX) -L../../makefiles/ -o $@ $^ -lsmarties

%.o: %.cpp
	$(CXX) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra)  $(CPPFLAGS) -MD -c $<

clean:
	rm -f *.o *.s *.d cart-pole
