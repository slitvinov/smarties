CXX ?= g++

extra ?=
smart ?= 1
train ?= 1
multdt ?= 1
costfun ?= 1
aspectr ?= 0.1
density ?= 200
CPPFLAGS += -DRANDOM_START=$(train) -DRHORATIO=$(density) -DASPECTRATIO=$(aspectr) -DINSTREW=$(costfun) -DMULT_ACT_DT=$(multdt)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
ifeq ($(CXX), c++)
#LD=g++-9
#CXX=g++-9
endif
endif

ifeq "$(smart)" "1"
CPPFLAGS+= -DUSE_SMARTIES
endif

SRC_DIR = ./
COM_DIR = ../../source/
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS+=-std=c++14 -O2 -Wall -I$(COM_DIR)

.DEFAULT_GOAL := glider

glider: glider.o
	$(CXX) $(extra) $(LIBS) -L../../makefiles/ -o $@ $^ -lsmarties

%.o: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -MD -c $<

clean:
	rm -f *.o *.s *.d glider
