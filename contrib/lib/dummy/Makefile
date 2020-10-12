.POSIX:
.SUFFIXES: .o .c
.PHONY: clean all install

PREFIX = $(HOME)/.local
CFLAGS = -Ofast -g
CC = mpicc
LINK = $(CC)

M_CFLAGS = -I$(PREFIX)/include

O = \
main.o\
dlopen.o\

L = libsmarties_dummy.a
$L: $O; ar rv $@ $O && ranlib $@
.c.o:; $(CC) -c $(CFLAGS) $(M_CFLAGS) $<
install: $L $H
	mkdir -p $(PREFIX)/lib && \
	cp $L $(PREFIX)/lib/
clean:; rm -f -- $L $O
