.POSIX:
.SUFFIXES: .o .f

PREFIX = $(HOME)/.local
CFLAGS = -O2 -g
CC = mpicc
LINK = mpicxx
SMARTIES_F77 = -lsmarties_f77
M_LDFLAGS = -fopenmp -Wl,-R"`pwd`" -L$(PREFIX)/lib $(SMARTIES_F77) -lsmarties -ldl
M_CFLAGS= -I$(PREFIX)/include

O = \
main.o\

M = \
main \

all: $M
$M: $O
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; -rm -f $M $O
