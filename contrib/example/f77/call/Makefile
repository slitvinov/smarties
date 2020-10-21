.POSIX:
.SUFFIXES: .cpp .f
.PHONY: clean all

PREFIX = $(HOME)/.local
FCFLAGS = -O2 -g
FC = mpif90
M_LDFLAGS = -Wl,-R$(PREFIX)/lib -L$(PREFIX)/lib -lsmarties_f77 -lsmarties -lgfortran -lmpi++ -lstdc++

M = \
main \

all: $M
.f:; $(FC) $(FCFLAGS) $< $(M_LDFLAGS) $(LDFLAGS) -o $@
clean:; rm -f -- $M
