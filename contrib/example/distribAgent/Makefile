.POSIX:
.SUFFIXES: .o .cpp
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
CXX = mpicxx
LINK = $(CXX)

M_CXXFLAGS = -I$(PREFIX)/include
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties_f77 -lsmarties

O = \
main.o\

M = \
main\

all: $M
$M: $O
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
