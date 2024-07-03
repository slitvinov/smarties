.POSIX:
.SUFFIXES:
CXX = mpicxx
PREFIX = $(HOME)/.local
FLAGS_SMARTIES = -I'$(PREFIX)'/include -L'$(PREFIX)'/lib -lsmarties -fopenmp
S = boatNav.cpp  odeSolve.cpp
main: $S
	$(CXX) $S $(CXXFLAGS) $(FLAGS_SMARTIES) $(LDFLAGS) -o $@
clean:; rm -f main
