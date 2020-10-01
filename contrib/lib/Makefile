.POSIX:
.SUFFIXES: .o .cpp
.PHONY: clean all

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
CXX = mpicxx
LINK = $(CXX)

M_CXXFLAGS = -I$(PREFIX)/include

O = \
main.o\

L = libsmarties_f77.a

$L: $O; ar rv $@ $O && ranlib $@
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
clean:; rm -f -- $L $O
