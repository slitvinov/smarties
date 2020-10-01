.POSIX:
.SUFFIXES: .o .cpp .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
FCFLAGS = -Ofast -g
FC = mpifort
CXX = mpicxx
LINK = $(FC)

M_CXXFLAGS = -I$(PREFIX)/include
M_FCFLAGS = -I$(PREFIX)/include -I.
MPI_EXTRA_LIB = -lmpi_cxx -lstdc++
M_LDFLAGS = -Wl,-R$(PREFIX)/lib -L$(PREFIX)/lib -lsmarties_f77 -lsmarties $(MPI_EXTRA_LIB)

O = \
app_main.o\
main.o\

M = \
app_main\

all: $M
$M: $O
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O cart_pole.mod
