.POSIX:
.SUFFIXES: .o .c .f
.PHONY: clean all

PREFIX = $(HOME)/.local
FC = gfortran
CC = gcc

FCFLAGS = -w -O2 -g
CFLAGS = -w -O2 -g
M_CFLAGS  = -DUNDERSCORE -DTIMER -Icore -Icore/3rd_party -Icore/experimental -I$(PREFIX)/include
M_FCFLAGS = -cpp -fdefault-real-8 -fdefault-double-8 -std=legacy -DNOMPIIO -DTIMER -I. -Icore -I$(PREFIX)/include

include obj.mk
L = libnek5000.a
$L: $O turbChannel.o; ar rv $@ $O turbChannel.o && ranlib $@
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $< -o $@
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $< -o $@

turbChannel.f: turbChannel.usr
	PPS='UNDERSCORE GLOBAL_LONG_LONG TIMER' CASENAME=turbChannel core/mkuserfile

### f77 -cpp -Icore -I. -MM core/*.f *.f | awk '/:/{print "core/" $0; next}; 1' > dep.mk
include dep.mk
clean:; rm -f -- $L $O turbChannel.o turbChannel.f
