.POSIX:
.SUFFIXES: .o .cpp .f
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
FCFLAGS = -Ofast -g
FC = mpifort
CXX = mpicxx
LINK = $(FC)

M_CXXFLAGS = -I$(PREFIX)/include
M_FCFLAGS = -ffree-form -std=legacy -Wall -Wextra -I$(PREFIX)/include -I. -Wno-align-commons
M_LDFLAGS = -Wl,-R$(PREFIX)/lib -L$(PREFIX)/lib -lsmarties_f77 -lsmarties -lmpi_cxx -lstdc++

O = \
app_main.o\
main.o\

M = \
app_main\

all: $M
$M: $O
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
clean:; rm -f -- $M $O cart_pole.mod
