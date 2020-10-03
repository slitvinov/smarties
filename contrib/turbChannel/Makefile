.POSIX:
.SUFFIXES: .o .c .f
.PHONY: clean all

PREFIX = $(HOME)/.local
FC=mpif77
CC=mpicc

FCFLAGS = -c  -O2
M_CFLAGS  = 
M_FCFLAGS = -cpp -fdefault-real-8 -fdefault-double-8 -std=legacy  -DMPI -DUNDERSCORE -DGLOBAL_LONG_LONG -DTIMER -I.

include obj.mk
L = libnek5000.a
$L: $O; ar rv $@ $O && ranlib $@
.c.o:; $(CC) -c $(CFLAGS) $(M_CXXFLAGS) $<
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<

clean:; rm -f -- $L $O
