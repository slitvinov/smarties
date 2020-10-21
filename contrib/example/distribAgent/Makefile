.POSIX:
.SUFFIXES: .o .c
.PHONY: clean all

PREFIX = $(HOME)/.local
CFLAGS = -O2 -g
CC = mpicc
LINK = mpicxx

M_CFLAGS = -I$(PREFIX)/include
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties_f77 -lsmarties

O = \
main.o\

M = \
main\

all: $M
$M: $O
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O
