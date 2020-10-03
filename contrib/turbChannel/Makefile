.POSIX:
.SUFFIXES: .o .f
.PHONY: clean all

PREFIX = $(HOME)/.local
FCFLAGS = -Ofast -g
FC = mpif77
LINK = $(FC)

M_LDFLAGS = -L$(PREFIX)/lib -Llib -lnek5000 -lgs -lblas -llapack

O = \
drive.o

M = \
drive \

all: $M
$M: $O
.f.o:; $(FC) -c $(FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $O $M
