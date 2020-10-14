libso: libnek5000.so
include makefile

C = $(OBJDIR)/.cookie
PREFIX = $(HOME)/.local
LINK = gfortran
MPI_EXTRA_LIB = -lgfortran
FFLAGS += -fPIC
CFLAGS += -fPIC
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib -lsmarties_f77 -lsmarties $(MPI_EXTRA_LIB)
O = $(NOBJS) $(CASENAME).o app_main.o
$C: objdir; touch $@
$O: $C
$(CASENAME).o: usrfile
$(CASENAME).o: $(CASENAME).f; $(FC) -c $(FL2) $< -o $@
libnek5000.so: $O
	$(LINK) -shared $O $(LDFLAGS) $(M_LDFLAGS) $(USR_LFLAGS) -o $@
app_main.o: app_main.f
	$(FC) $(FL2) -c $(FCFLAGS) $(M_FCFLAGS) app_main.f
libclean:; -rm -f libnek5000.o $C
