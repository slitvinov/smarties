.POSIX:
.SUFFIXES: .o .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CFLAGS = -Ofast -g
CC = mpicc
LINK = mpicxx
SMARTIES_F77 = -lsmarties_f77
M_LDFLAGS = -fopenmp -Wl,-R"`pwd`"/lib -L$(PREFIX)/lib $(SMARTIES_F77) -lsmarties -ldl -rdynamic
M_CFLAGS= -I$(PREFIX)/include

O = \
main.o\

M = \
main \

all: $M
$M: $O
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
