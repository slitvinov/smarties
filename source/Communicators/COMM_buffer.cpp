//#include "Agent.h"
#include <cstring>
#include <cassert>
#include <vector>
#include <memory>

enum episodeStatus {INIT, CONT, TERM, TRNC, FAIL};
enum learnerStatus {WORK, KILL};

struct Agent
{
  const unsigned ID;
  const unsigned workerID;
  const unsigned localID;

  episodeStatus agentStatus = INIT;
  unsigned timeStepInEpisode = 0;

  learnerStatus learnStatus = WORK;
  unsigned learnerStepID = 0;

  bool trackSequence = true;

  std::vector<double> sOld; // previous state
  std::vector<double> state;  // current state
  std::vector<double> action;
  double reward; // current reward
  double cumulative_rewards = 0;

  void reset()
  {
    agentStatus = INIT;
    timeStepInEpisode=0;
    cumulative_rewards=0;
    reward=0;
  }
};

struct COMM_buffer
{
  static size_t computeStateMsgSize(const size_t maxSdim) {
   return 2*sizeof(unsigned) +sizeof(episodeStatus) +(maxSdim+1)*sizeof(double);
  }
  static size_t computeActionMsgSize(const size_t maxAdim) {
   return 2*sizeof(unsigned) +sizeof(learnerStatus) + maxAdim*sizeof(double);
  }

  COMM_buffer(const size_t maxStateDim, const size_t maxActionDim) :
    maxStateDimension(maxStateDim),
    sizeStateMsg(computeStateMsgSize(maxStateDimension)),
    dataStateBuf(malloc(sizeStateMsg)),
    maxActionDimension(maxActionDim),
    sizeActionMsg(computeActionMsgSize(maxActionDimension)),
    dataActionBuf(malloc(sizeActionMsg))
  { }
  ~COMM_buffer()
  {
    assert(dataStateBuf not_eq nullptr && dataActionBuf not_eq nullptr);
    free(dataActionBuf);
    free(dataStateBuf);
  }

  COMM_buffer(const COMM_buffer& c) = delete;
  COMM_buffer& operator= (const COMM_buffer& s) = delete;

  const size_t maxStateDimension, sizeStateMsg;
  void * const dataStateBuf;

  void packStateMsg(const Agent& A) const // put agent's state into buffer
  {
    assert(dataStateBuf not_eq nullptr);
    char * msgPos = (char*) dataStateBuf;
    memcpy(msgPos, &A.localID,           sizeof(unsigned));
    msgPos +=                            sizeof(unsigned) ;
    memcpy(msgPos, &A.agentStatus,       sizeof(episodeStatus));
    msgPos +=                            sizeof(episodeStatus) ;
    memcpy(msgPos, &A.timeStepInEpisode, sizeof(unsigned));
    msgPos +=                            sizeof(unsigned) ;
    memcpy(msgPos, &A.reward,            sizeof(double));
    msgPos +=                            sizeof(double) ;
    memcpy(msgPos,  A.state.data(),      sizeof(double) * A.state.size());
    msgPos +=                            sizeof(double) * A.state.size() ;
    assert(msgPos - (const char*) dataStateBuf <= (ptrdiff_t) sizeStateMsg);
  }

  void unpackStateMsg(Agent& A) const // get agent's state from buffer
  {
    assert(dataStateBuf not_eq nullptr);
    const char * msgPos = (const char*) dataStateBuf;
    unsigned testAgentID;
    memcpy(&testAgentID,         msgPos, sizeof(unsigned));
    assert( testAgentID == A.localID ); // worker may not know number of workers
    msgPos +=                            sizeof(unsigned) ;
    memcpy(&A.agentStatus,       msgPos, sizeof(episodeStatus));
    msgPos +=                            sizeof(episodeStatus) ;
    memcpy(&A.timeStepInEpisode, msgPos, sizeof(unsigned));
    msgPos +=                            sizeof(unsigned) ;
    memcpy(&A.reward,            msgPos, sizeof(double));
    msgPos +=                            sizeof(double) ;
    memcpy( A.state.data(),      msgPos, sizeof(double) * A.state.size());
    msgPos +=                            sizeof(double) * A.state.size() ;
    assert(msgPos - (const char*) dataStateBuf <= (ptrdiff_t) sizeStateMsg);
  }

  const size_t maxActionDimension, sizeActionMsg;
  void * const dataActionBuf;

  void packActionMsg(const Agent& A) const
  {
    assert(dataActionBuf not_eq nullptr);
    char * msgPos = (char*) dataActionBuf;
    memcpy(msgPos, &A.localID,       sizeof(unsigned));
    msgPos +=                        sizeof(unsigned) ;
    memcpy(msgPos, &A.learnStatus,   sizeof(learnerStatus));
    msgPos +=                        sizeof(learnerStatus) ;
    memcpy(msgPos, &A.learnerStepID, sizeof(unsigned));
    msgPos +=                        sizeof(unsigned) ;
    memcpy(msgPos,  A.action.data(), sizeof(double) * A.action.size());
    msgPos +=                        sizeof(double) * A.action.size() ;
    assert(msgPos - (const char*) dataActionBuf <= (ptrdiff_t) sizeActionMsg);
  }

  void unpackActionMsg(Agent& A) const
  {
    assert(dataActionBuf not_eq nullptr);
    const char * msgPos = (const char*) dataActionBuf;
    unsigned testAgentID;
    memcpy(&testAgentID,     msgPos, sizeof(unsigned));
    assert( testAgentID == A.localID ); // worker may not know number of workers
    msgPos +=                        sizeof(unsigned) ;
    memcpy(&A.learnStatus,   msgPos, sizeof(learnerStatus));
    msgPos +=                        sizeof(learnerStatus) ;
    memcpy(&A.learnerStepID, msgPos, sizeof(unsigned));
    msgPos +=                        sizeof(unsigned) ;
    memcpy( A.action.data(), msgPos, sizeof(double) * A.action.size());
    msgPos +=                        sizeof(double) * A.action.size() ;
    assert(msgPos - (const char*) dataActionBuf <= (ptrdiff_t) sizeActionMsg);
  }
};

int main()
{
  std::vector<std::unique_ptr<COMM_buffer>> vec;
  vec.reserve(5);
  vec[0] = std::make_unique<COMM_buffer>(4, 2);
  static_assert( sizeof(char) == 1, "wtf is the size of a char?");
  return 0;
}
