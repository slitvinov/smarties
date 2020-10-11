.POSIX:
.SUFFIXES: .o .f

PREFIX = $(HOME)/.local
CFLAGS = -Ofast -g
CC = mpicc
LINK = mpicxx
M_LDFLAGS = -fopenmp -Wl,-R"`pwd`" -L$(PREFIX)/lib -lsmarties_f77 -lsmarties -ldl
M_CFLAGS= -I$(PREFIX)/include

O = \
main.o\

M = \
main \

all: $M
$M: $O
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
binclean: clean; -rm -f $M
