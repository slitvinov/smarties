//#include "Agent.h"
#include <cstring>
#include <cassert>
#include <vector>
#include <memory>

struct COMM_buffer
{
  COMM_buffer(const size_t maxStateDim, const size_t maxActionDim) :
    maxStateDim(maxStateDim), maxActionDim(maxActionDim),
    sizeStateMsg(Agent::computeStateMsgSize(maxStateDimension)),
    sizeActionMsg(Agent::computeActionMsgSize(maxActionDimension)),
    dataStateBuf(malloc(sizeStateMsg)), dataActionBuf(malloc(sizeActionMsg)) { }

  ~COMM_buffer() {
    assert(dataStateBuf not_eq nullptr && dataActionBuf not_eq nullptr);
    free(dataActionBuf);
    free(dataStateBuf);
  }

  COMM_buffer(const COMM_buffer& c) = delete;
  COMM_buffer& operator= (const COMM_buffer& s) = delete;

  const size_t maxStateDim, maxActionDim, sizeStateMsg, sizeActionMsg;
  void * const dataStateBuf;
  void * const dataActionBuf;
};

#if 0 // testing
int main()
{
  std::vector<std::unique_ptr<COMM_buffer>> vec;
  vec.reserve(5);
  vec[0] = std::make_unique<COMM_buffer>(4, 2);
  static_assert( sizeof(char) == 1, "wtf is the size of a char?");
  return 0;
}
#endif
