.POSIX:
.SUFFIXES: .o .c .f
.PHONY: clean all

PREFIX = $(HOME)/.local
FC = gfortran
CC = c99

FCFLAGS = -w -g -O2
CFLAGS = -w -g -O2
M_CFLAGS  = -DUNDERSCORE -DTIMER -Icore -Icore/3rd_party -Icore/experimental -I$(PREFIX)/include
M_FCFLAGS = -cpp -fdefault-real-8 -fdefault-double-8 -std=legacy -DNOMPIIO -DTIMER -I. -Icore -I$(PREFIX)/include

include obj.mk
L = libnek5000.a
$L: $O turbChannel.o; ar rv $@ $O turbChannel.o && ranlib $@
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $< -o $@
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $< -o $@

turbChannel.f: turbChannel.usr
	PPS='UNDERSCORE GLOBAL_LONG_LONG TIMER' CASENAME=turbChannel core/mkuserfile
include dep.mk
clean:; rm -f -- $L $O turbChannel.o turbChannel.f
