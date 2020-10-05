.POSIX:
.SUFFIXES: .o .cpp .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
FCFLAGS = -Ofast -g
FC = gfortran
CXX = mpicxx
LINK = $(CXX)

M_CXXFLAGS = -I$(PREFIX)/include
MPI_EXTRA_LIB = -lgfortran
M_LDFLAGS = -L$(PREFIX)/lib -lsmarties_f77 -lsmarties -lgomp $(MPI_EXTRA_LIB)

O = \
app_main.o\
main.o\

M = \
app_main\

all: $M
$M: $O
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.f.o:; $(FC) -c $(FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
