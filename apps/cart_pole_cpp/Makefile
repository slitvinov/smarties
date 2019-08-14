CXX = mpic++

SMARTIES_DIR = ../../

LIBS+= -L$(SMARTIES_DIR)lib/
CPPFLAGS+= -std=c++14 -Wall -O2 -I$(SMARTIES_DIR)include
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

.DEFAULT_GOAL := cart-pole

cart-pole: cart-pole.o
	$(CXX) $(extra) $(LIBS) -o $@ $^ -lsmarties

%.o: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -MD -c $<

clean:
	rm -f *.o *.s *.d cart-pole
