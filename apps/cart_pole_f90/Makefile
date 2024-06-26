.POSIX:
.SUFFIXES:
CXX = mpicxx
FCFLAGS = -O2 -g
PREFIX = $(HOME)/.local
FLAGS_SMARTIES = -I'$(PREFIX)'/include -L'$(PREFIX)'/lib -lsmarties -fopenmp \
	-lmpi_mpifh -lgfortran
S = $(PREFIX)/include/smarties.f90 cart_pole.f90 app_main.f90 main.cpp
main: $S
	$(CXX) $S $(CXXFLAGS) $(FLAGS_SMARTIES) $(LDFLAGS) -o $@
clean:; rm -f main smarties.mod cart_pole.mod
