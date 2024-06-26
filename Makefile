.POSIX:
.SUFFIXES:
.SUFFIXES: .o .cpp

PREFIX = $(HOME)/.local
CXX = mpicxx
CXXFLAGS = -O2 -g
M_CXXFLAGS = -DNDEBUG -fopenmp

include obj.mk
include hdr.mk
include dir.mk
L = libsmarties.a
$L: $O; ar rv $@ $O && ranlib $@
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $< -o $@
install: $L $H
	mkdir -p -- '$(PREFIX)'/lib; \
	cp -- $L '$(PREFIX)'/lib; \
	for i in $D; \
	do mkdir -p -- '$(PREFIX)'/$$i; \
	done; \
	for i in $H; \
	do cp -- $$i '$(PREFIX)'/$$i; \
	done

clean:; rm -f -- $L $O

# tool/dep > dep.mk
include dep.mk
