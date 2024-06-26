.POSIX:
.SUFFIXES:
CXX = mpicxx
PREFIX = $(HOME)/.local
FLAGS_SMARTIES = -I'$(PREFIX)'/include -L'$(PREFIX)'/lib -lsmarties -fopenmp \
	-lmpi_mpifh -lgfortran
S = test-fit.cpp
main: $S
	$(CXX) $S $(CXXFLAGS) $(FLAGS_SMARTIES) $(LDFLAGS) -o $@
clean:; rm -f main smarties.mod cart_pole.mod
