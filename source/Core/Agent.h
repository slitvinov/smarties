//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#ifndef smarties_Agent_h
#define smarties_Agent_h

#include "StateAction.h"
#include <cstring> // memcpy

#ifndef SMARTIES_LIB
#include <atomic>
#define OUTBUFFSIZE 65536
#endif

namespace smarties
{

enum episodeStatus {INIT, CONT, TERM, TRNC, FAIL};
enum learnerStatus {WORK, KILL};

inline int status2int(const episodeStatus status) {
  if(status == INIT) return 0;
  if(status == CONT) return 1;
  if(status == TERM) return 2;
  if(status == TRNC) return 3;
  if(status == FAIL) return 4;
  die("unreachable");
}

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

  MDPdescriptor& MDP;
  const StateInfo  sInfo = StateInfo(MDP);
  const ActionInfo aInfo = ActionInfo(MDP);

  std::vector<double> sOld; // previous state
  std::vector<double> state;  // current state
  std::vector<double> action;
  double reward; // current reward
  double cumulativeRewards = 0;

  Agent(Uint _ID, Uint workID, Uint _localID, MDPdescriptor& _MDP) :
    ID(_ID), workerID(workID), localID(_localID), MDP(_MDP) {}

  void reset()
  {
    agentStatus = INIT;
    timeStepInEpisode=0;
    cumulativeRewards=0;
    reward=0;
  }

  template<typename T>
  void update(const episodeStatus E, const std::vector<T>& S, const double R)
  {
    assert( S.size() == sInfo.dim() );
    agentStatus = E;

    if(E == FAIL) { // app crash, probably
      reset();
      return;
    }

    std::swap(sOld, state); //what is stored in state now is sold
    state = std::vector<double>(S.begin(), S.end());
    reward = R;

    if(E == INIT) {
      cumulativeRewards = 0;
      timeStepInEpisode = 0;
    } else {
      cumulativeRewards += R;
      ++timeStepInEpisode;
    }
  }
  void act(const Uint label)
  {
    action = aInfo.label2actionMessage<double>(label);
  }
  void act(const Rvec& _act)
  {
    action = std::vector<double>(_act.begin(), _act.end());
  }

  void packStateMsg(void * const buffer) const // put agent's state into buffer
  {
    assert(buffer not_eq nullptr);
    char * msgPos = (char*) buffer;
    memcpy(msgPos, &localID,           sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(msgPos, &agentStatus,       sizeof(episodeStatus));
    msgPos +=                          sizeof(episodeStatus) ;
    memcpy(msgPos, &timeStepInEpisode, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(msgPos, &reward,            sizeof(double));
    msgPos +=                          sizeof(double) ;
    memcpy(msgPos,  state.data(),      sizeof(double) * state.size());
    msgPos +=                          sizeof(double) * state.size() ;
  }

  void unpackStateMsg(const void * const buffer) // get state from buffer
  {
    assert(buffer not_eq nullptr);
    const char * msgPos = (const char*) buffer;
    unsigned testAgentID, testStepID;
    std::swap(sOld, state); //what is stored in state now is sold

    memcpy(&testAgentID,       msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(&agentStatus,       msgPos, sizeof(episodeStatus));
    msgPos +=                          sizeof(episodeStatus) ;
    memcpy(&testStepID,        msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(&reward,            msgPos, sizeof(double));
    msgPos +=                          sizeof(double) ;
    memcpy( state.data(),      msgPos, sizeof(double) * state.size());
    msgPos +=                          sizeof(double) * state.size() ;

    if(agentStatus == INIT) {
      cumulativeRewards = 0;
      timeStepInEpisode = 0;
    } else {
      cumulativeRewards += reward;
      ++timeStepInEpisode;
    }
    assert(testStepID == timeStepInEpisode && testAgentID == localID);
  }

  void packActionMsg(void * const buffer) const
  {
    assert(buffer not_eq nullptr);
    char * msgPos = (char*) buffer;
    memcpy(msgPos, &localID,       sizeof(unsigned));
    msgPos +=                      sizeof(unsigned) ;
    memcpy(msgPos, &learnStatus,   sizeof(learnerStatus));
    msgPos +=                      sizeof(learnerStatus) ;
    memcpy(msgPos, &learnerStepID, sizeof(unsigned));
    msgPos +=                      sizeof(unsigned) ;
    memcpy(msgPos,  action.data(), sizeof(double) * action.size());
    msgPos +=                      sizeof(double) * action.size() ;
  }

  void unpackActionMsg(const void * const buffer)
  {
    assert(buffer not_eq nullptr);
    const char * msgPos = (const char*) buffer;
    unsigned testAgentID;
    memcpy(&testAgentID,   msgPos, sizeof(unsigned));
    msgPos +=                      sizeof(unsigned) ;
    memcpy(&learnStatus,   msgPos, sizeof(learnerStatus));
    msgPos +=                      sizeof(learnerStatus) ;
    memcpy(&learnerStepID, msgPos, sizeof(unsigned));
    msgPos +=                      sizeof(unsigned) ;
    memcpy( action.data(), msgPos, sizeof(double) * action.size());
    msgPos +=                      sizeof(double) * action.size() ;
    assert( testAgentID == localID );
  }

  static unsigned getMessageAgentID(const void *const buffer)
  {
    return * (unsigned*) buffer;
  }
  static episodeStatus& messageEpisodeStatus(char * buffer)
  {
    buffer += sizeof(unsigned);
    return * (episodeStatus *) buffer;
  }
  static learnerStatus& messageLearnerStatus(char * buffer)
  {
    buffer += sizeof(unsigned);
    return * (learnerStatus *) buffer;
  }
  static size_t computeStateMsgSize(const size_t sDim)
  {
   return 2*sizeof(unsigned) + sizeof(episodeStatus) + (sDim+1)*sizeof(double);
  }
  static size_t computeActionMsgSize(const size_t aDim)
  {
   return 2*sizeof(unsigned) +sizeof(learnerStatus) + aDim*sizeof(double);
  }

#ifndef SMARTIES_LIB
  // for dumping to state-action-reward-policy binary log (writeBuffer):
  mutable float buf[OUTBUFFSIZE];
  mutable std::atomic<Uint> buffCnter {0};

  void writeBuffer(const int rank) const
  {
    if(buffCnter == 0) return;
    char cpath[256];
    sprintf(cpath, "agent%03d_rank%02d_obs.raw", ID, rank);
    FILE * pFile = fopen (cpath, "ab");

    fwrite (buf, sizeof(float), buffCnter, pFile);
    fflush(pFile); fclose(pFile);
    buffCnter = 0;
  }

  void writeData(const int rank, const Rvec mu, const Uint globalTstep) const
  {
    // possible race conditions, avoided by the fact that each worker
    // (and therefore agent) can only be handled by one thread at the time
    // atomic op is to make sure that counter gets flushed to all threads
    const Uint writesize = 4 + sInfo.dim() + aInfo.dim() + mu.size();
    if(OUTBUFFSIZE<writesize) die("Increase compile-time OUTBUFFSIZE variable");
    assert( buffCnter % writesize == 0 );
    if(buffCnter+writesize > OUTBUFFSIZE) writeBuffer(rank);
    Uint ind = buffCnter;
    buf[ind++] = globalTstep + 0.1;
    buf[ind++] = status2int(agentStatus) + 0.1;
    buf[ind++] = timeStepInEpisode + 0.1;
    assert( state.size() == sInfo.dim());
    for (Uint i=0; i<state.size(); i++) buf[ind++] = (float) state[i];
    assert(action.size() == aInfo.dim());
    for (Uint i=0; i<action.size(); i++) buf[ind++] = (float) action[i];
    buf[ind++] = reward;
    for (Uint i=0; i<mu.size(); i++) buf[ind++] = (float) mu[i];

    buffCnter += writesize;
    assert(buffCnter == ind);
  }

#endif

  void checkNanInf() const {
    #ifndef NDEBUG
      const auto assertValid = [] (const double val) {
        assert( not std::isnan(val) and not std::isinf(val) );
      };
      for (Uint j=0; j<action.size(); ++j) assertValid(action[j]);
      for (Uint j=0; j<state.size(); ++j) assertValid(state[j]);
      assertValid(reward);
    #endif
  }
};

} // end namespace smarties
#undef OUTBUFFSIZE
#endif // smarties_Agent_h
