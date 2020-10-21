.POSIX:
.SUFFIXES: .o .cpp
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -O2 -g
FCFLAGS = -O2 -g
FC = gfortran
CXX = mpicxx
LINK = mpicxx

M_CXXFLAGS = -I$(PREFIX)/include
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties

O = \
main.o\

M = \
main\

all: $M
$M: $O
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
