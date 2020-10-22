.POSIX:
.SUFFIXES: .o .f .c
PREFIX = $(HOME)/.local
CFLAGS = -O2 -g
FCFLAGS = -O2 -g
FC = gfortran
CC = mpicc
LINK = mpicxx

M_CFLAGS = -I$(PREFIX)/include
MPI_EXTRA_LIB = -lgfortran
SMARTIES_F77 = -lsmarties_f77
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib $(SMARTIES_F77) -lsmarties -lm $(MPI_EXTRA_LIB)

O = \
app_main.o\
main.o\

M = \
main\

all: $M
$M: $O
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $<
.f.o:; $(FC) -c $(FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
