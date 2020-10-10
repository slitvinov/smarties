.POSIX:
.SUFFIXES: .o .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CFLAGS = -Ofast -g
CC = mpicc
LINK = mpicxx
M_LDFLAGS = -fopenmp -Wl,-R"`pwd`"/lib -L$(PREFIX)/lib -lsmarties_f77 -lsmarties -ldl
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
