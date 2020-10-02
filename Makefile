.POSIX:
.SUFFIXES: .o .cpp
.PHONY: clean all install

PREFIX = $(HOME)/.local
CXXFLAGS = -O3 -g
M_CXXFLAGS = -Dlibsmarties_EXPORTS -DNDEBUG -pthread -fopenmp

include obj.mk
L = libsmarties.a

$L: $O; ar rv $@ $O && ranlib $@
.cpp.o:; $(CXX) -c $(CXXFLAGS) $(M_CXXFLAGS) $< -o $@
install: $L; mkdir -p $(PREFIX)/lib && cp $L $(PREFIX)/lib/
clean:; rm -f -- $L $O
