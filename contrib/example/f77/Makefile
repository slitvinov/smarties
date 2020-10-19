.POSIX:
.SUFFIXES: .o .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CCFLAGS = -Ofast -g
FCFLAGS = -Ofast -g
FC = gfortran
CC = mpicc
LINK = $(CC)

M_CCFLAGS = -I$(PREFIX)/include
MPI_EXTRA_LIB = -lgfortran
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties_f77 -lsmarties -lmpi_cxx -lstdc++ -lm $(MPI_EXTRA_LIB)

O = \
app_main.o\
main.o\

M = \
main\

all: $M
$M: $O
.c.o:; $(CC) -c $(CCFLAGS) $(M_CCFLAGS) $<
.f.o:; $(FC) -c $(FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
