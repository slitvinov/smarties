.POSIX:
.SUFFIXES: .o .cpp
.PHONY: clean all install

PREFIX = $(HOME)/.local
CXXFLAGS = -Ofast -g
CXX = mpicxx
LINK = $(CXX)

M_CXXFLAGS = -fPIC -I$(PREFIX)/include

O = \
main.o\

H = \
smarties_f77.h\

L = libsmarties_f77.a
$L: $O; ar rv $@ $O && ranlib $@
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
install: $L $H
	mkdir -p $(PREFIX)/lib $(PREFIX)/include && \
	cp $L $(PREFIX)/lib/ && cp $H $(PREFIX)/include/
main.o: smarties_f77.h
clean:; rm -f -- $L $O
