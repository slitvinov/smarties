.POSIX:
.SUFFIXES: .o .cpp .f90
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -O2 -g
FCFLAGS = -O2 -g
FC = mpifort
CXX = mpicxx
LINK = $(FC)

M_CXXFLAGS = -I$(PREFIX)/include
M_FCFLAGS = -I$(PREFIX)/include -I.
MPI_EXTRA_LIB = -lmpi_cxx -lstdc++
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties $(MPI_EXTRA_LIB)

O = \
app_main.o\
main.o\
smarties_extern.o\

M = \
app_main\

all: $M
$M: $O
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.f90.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O cart_pole.mod
