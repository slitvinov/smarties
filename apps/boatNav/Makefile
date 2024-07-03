.POSIX:
.SUFFIXES:
.SUFFIXES: .o
.SUFFIXES: .cpp
CXX = mpicxx
PREFIX = $(HOME)/.local
CXXFLAGS = -O2 -g
FLAGS_SMARTIES = -I'$(PREFIX)'/include -L'$(PREFIX)'/lib -lsmarties -fopenmp
all: boatNav runTest
.cpp:
	$(CXX) $< odeSolve.o $(CXXFLAGS) $(FLAGS_SMARTIES) $(LDFLAGS) -o $@
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<
clean:; rm -f main runTest
boatNav: odeSolve.o
runTest: odeSolve.o
