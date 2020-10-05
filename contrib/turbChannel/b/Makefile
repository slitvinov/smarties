.POSIX:
.SUFFIXES: .o .f .cpp
.PHONY: clean all

PREFIX = $(HOME)/.local
BLAS_LIBS = -lblas -llapack
FCFLAGS = -w -Ofast -g
CXXFLAGS = -Ofast -g
CXX = mpicxx
FC = f77
LINK = $(CXX)

M_CXXFLAGS = -I$(PREFIX)/include
M_FCFLAGS = -cpp -fdefault-real-8 -fdefault-double-8 -std=legacy -DNOMPIIO -DTIMER -Ilib -Ilib/core -I$(PREFIX)/include
M_LDFLAGS = -L$(PREFIX)/lib -Llib -Wl,--allow-multiple-definition -lsmarties_f77 -lsmarties -lgomp -lnek5000 -lgs $(BLAS_LIBS) -lgfortran

O = \
app_main.o\
main.o\
turbChannel.o\

M = \
app_main \

all: $M
$M: $O
.f.o:; $(FC) -c $(FCFLAGS) $(M_FCFLAGS) $<
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.o:; $(LINK) $O $(LDFLAGS) $(M_LDFLAGS) -o $@
turbChannel.f: turbChannel.usr
	PPS='UNDERSCORE GLOBAL_LONG_LONG TIMER' CASENAME=turbChannel lib/core/mkuserfile
clean:; rm -f -- $O $M turbChannel.o turbChannel.f
