.POSIX:
.SUFFIXES: .o .cpp .f90
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
FCFLAGS = -Ofast -g
FC = mpif90
CXX = mpicxx
LINK = $(FC)

M_CXXFLAGS = -I$(PREFIX)/include
M_FCFLAGS = -I$(PREFIX)/include -I.
M_LDFLAGS = -Wl,-R$(PREFIX)/lib -L$(PREFIX)/lib -lsmarties -lgfortran -lmpi++ -lstdc++

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
