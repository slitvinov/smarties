.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
M = cart-pole
CXX = mpicxx
PREFIX = $(HOME)/.local
FLAGS_SMARTIES = -I'$(PREFIX)'/include -L'$(PREFIX)'/lib -lsmarties -fopenmp
all: $M
.cpp:; $(CXX) $< $(CXXFLAGS) $(FLAGS_SMARTIES) $(LDFLAGS) -o $@
clean:; rm -f $M
