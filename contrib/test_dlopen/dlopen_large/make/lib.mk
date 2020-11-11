.SUFFIXES: .o .f
libso: libnek5000.so
include makefile

PREFIX = $(HOME)/.local
LINK = mpif77
FCFLAG = -O2 -g
FFLAGS += -fPIC $(FCFLAG)
CFLAGS += -fPIC
SMARTIES_F77 = -lsmarties_f77
M_LDFLAGS = -fopenmp -L$(PREFIX)/lib $(SMARTIES_F77) -lsmarties
O = $(NOBJS) $(CASENAME).o app_main.o
${CASENAME}.f: usrfile
.f.o:; $(FC) $(FL2) -I$S -c $<
$(NOBJS): objdir
libnek5000.so: $O; $(LINK) -shared $O $(LDFLAGS) $(M_LDFLAGS) $(USR_LFLAGS) -o $@
libclean:; -rm -rf libnek5000.so $O $(CASENAME).f obj
