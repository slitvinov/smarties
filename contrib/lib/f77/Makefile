.POSIX:
.SUFFIXES: .o .c .cpp
.PHONY: clean all install

PREFIX = $(HOME)/.local
CXXFLAGS = -O2 -g
CFLAGS = -O2 -g
CXX = mpicxx
CC = mpicc
LINK = $(CXX)

M_CXXFLAGS = -I$(PREFIX)/include

O = \
main.o\
dlopen.o\

H = \
smarties_f77.h\

L = libsmarties_f77.a
$L: $O; ar rv $@ $O && ranlib $@
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $<
.c.o:; $(CC) -c $(CFLAGS) $<
install: $L $H
	mkdir -p $(PREFIX)/lib $(PREFIX)/include && \
	cp $L $(PREFIX)/lib/ && cp $H $(PREFIX)/include/
main.o: smarties_f77.h
clean:; rm -f -- $L $O
