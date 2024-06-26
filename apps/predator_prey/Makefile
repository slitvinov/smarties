.POSIX:
.SUFFIXES:
.SUFFIXES: .cpp
M = pp twoPredOnePrey
CXX = mpicxx
FLAGS_PYTHON != python3-config --cflags --libs --embed
PREFIX = $(HOME)/.local
FLAGS_SMARTIES = -I'$(PREFIX)'/include -L'$(PREFIX)'/lib -lsmarties -fopenmp
all: $M
pp: pp.cpp
	$(CXX) $< $(CXXFLAGS) $(FLAGS_SMARTIES) $(FLAGS_PYTHON) -fPIC $(LDFLAGS) -o pp
twoPredOnePrey: twoPredOnePrey.cpp
	$(CXX) $< $(CXXFLAGS) $(FLAGS_SMARTIES) $(LDFLAGS) -o twoPredOnePrey
clean:; rm -f $M
