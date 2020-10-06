.POSIX:
.SUFFIXES: .o .cpp
.PHONY: clean all install

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
install: $L; mkdir -p $(PREFIX)/lib && cp $L $(PREFIX)/lib/

main.cpp: smarties_f77.h

clean:; rm -f -- $L $O
