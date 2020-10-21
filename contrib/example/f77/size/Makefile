.POSIX:
.SUFFIXES: .cpp .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -O2 -g
FCFLAGS = -O2 -g
FC = gfortran
CXX = g++

M = \
f77\
cpp\

all: $M
.cpp:; $(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@
.f:; $(FC) $(FCFLAGS) $(LDFLAGS) $< -o $@
clean:; rm -f -- $M
