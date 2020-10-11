.POSIX:
.SUFFIXES: .o .f
.PHONY: clean all install

PREFIX = $(HOME)/.local
FCFLAGS = -Ofast -g
FC = gfortran
LINK = $(FC)

M_FCFLAGS = -I$(PREFIX)/include -fPIC -Wno-align-commons
MPI_EXTRA_LIB = -lgfortran
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties_f77 -lsmarties $(MPI_EXTRA_LIB)

O = \
main.o\

L = libmain.so
$L: $O; $(LINK) $O $(LDFLAGS) -shared $(M_LDFLAGS) -o $@
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<
clean:; rm -f -- $L $O
