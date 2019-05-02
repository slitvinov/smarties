#include <vector>
#include <memory>
//#include "Utils/Warnings.h"
//#include "Core/Environment.h"
//#include "Communicators/Communicator.h"

struct A
{
const int val;
A(int _val) : val(_val) {}
};

struct B
{
const std::vector<std::unique_ptr<A>> vec;
B(std::vector<std::unique_ptr<A>>& _vec) :
vec(std::move(_vec)) {}
void print() {
  printf("B: size : %lu\n", vec.size());
}
};


int main()
{
 std::vector<std::unique_ptr<A>> vec;
 vec.reserve(5);
 for(volatile int i=0; i<10; ++i) {
   volatile int val = 2313;
   vec.emplace_back( std::make_unique<A>(val) );
 }
 B b(vec);
 b.print();
 printf("main: size : %lu\n", vec.size());
 using OutputMatrices = __attribute__(( aligned(32) )) float[16][4][8];
 OutputMatrices O;
 //smarties::Environment ENV;
 //smarties::Communicator ENV(1);
 return 0;
}
