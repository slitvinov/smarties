.DEFAULT_GOAL := app

SMARTIESDIR = ../../source/ #Communicators/
CPPFLAGS += -std=c++14 -fopenmp -I$(SMARTIESDIR)
#CPPFLAGS += -fsanitize=undefined -g -O0
#CPPFLAGS += -g -O0 -fsanitize=address
VPATH := $(VPATH):$(SMARTIESDIR)

app: cart-pole.o
	ar rs libsimulation.a $^
	cp libsimulation.a ../../makefiles/libsimulation.a

clean:
	rm -f *.o libsimulation.a

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
