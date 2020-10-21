.POSIX:
.SUFFIXES: .o .f
.PHONY: clean all install

PREFIX = $(HOME)/.local
FCFLAGS = -Ofast -g
FC = gfortran
LINK = $(FC)

M_FCFLAGS = -I$(PREFIX)/include -fPIC -Wno-align-commons
SMARTIES_F77 = -lsmarties_f77
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib $(SMARTIES_F77) -lsmarties

O = \
main.o\

L = libmain.so
$L: $O; $(LINK) $O $(LDFLAGS) -shared $(M_LDFLAGS) -o $@
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<
clean:; rm -f -- $L $O
