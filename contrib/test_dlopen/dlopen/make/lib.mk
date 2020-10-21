.SUFFIXES: .o .f
libso: libnek5000.so
include makefile

PREFIX = $(HOME)/.local
LINK = gfortran
FCFLAG = -O2 -g
FFLAGS += -fPIC $(FCFLAG)
CFLAGS += -fPIC
MPI_EXTRA_LIB = -lgfortran
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties_f77 -lsmarties $(MPI_EXTRA_LIB)
O = $(NOBJS) $(CASENAME).o app_main.o
${CASENAME}.f: usrfile
.f.o:; $(FC) $(FL2) -I$S -c $<
$(NOBJS): objdir
libnek5000.so: $O; $(LINK) -shared $O $(LDFLAGS) $(M_LDFLAGS) $(USR_LFLAGS) -o $@
libclean:; -rm -rf libnek5000.so $O $(CASENAME).f obj
