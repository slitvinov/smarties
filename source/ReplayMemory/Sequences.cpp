//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Sequences.h"
#include <cstring>

namespace smarties
{

std::vector<Fval> Sequence::packSequence(const Uint dS, const Uint dA, const Uint dP)
{
  const Uint seq_len = states.size();
  assert(states.size() == actions.size() && states.size() == policies.size());
  const Uint totalSize = Sequence::computeTotalEpisodeSize(dS, dA, dP, seq_len);
  std::vector<Fval> ret(totalSize, 0);
  Fval* buf = ret.data();

  for (Uint i = 0; i<seq_len; ++i)
  {
    assert(states[i].size() == dS);
    assert(actions[i].size() == dA);
    assert(policies[i].size() == dP);
    std::copy(states[i].begin(), states[i].end(), buf);
    buf[dS] = rewards[i]; buf += dS + 1;
    std::copy(actions[i].begin(),  actions[i].end(),  buf); buf += dA;
    std::copy(policies[i].begin(), policies[i].end(), buf); buf += dP;
  }

  /////////////////////////////////////////////////////////////////////////////
  // following vectors may be of size less than seq_len because
  // some algorithms do not allocate them. I.e. Q-learning-based
  // algorithms do not need to advance retrace-like value estimates

  assert(Q_RET.size() <= seq_len);        Q_RET.resize(seq_len);
  std::copy(Q_RET.begin(), Q_RET.end(), buf); buf += seq_len;

  assert(action_adv.size() <= seq_len);   action_adv.resize(seq_len);
  std::copy(action_adv.begin(), action_adv.end(), buf); buf += seq_len;

  assert(state_vals.size() <= seq_len);   state_vals.resize(seq_len);
  std::copy(state_vals.begin(), state_vals.end(), buf); buf += seq_len;

  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  // post processing quantities might not be already allocated

  assert(SquaredError.size() <= seq_len); SquaredError.resize(seq_len);
  std::copy(SquaredError.begin(), SquaredError.end(), buf); buf += seq_len;

  assert(offPolicImpW.size() <= seq_len); offPolicImpW.resize(seq_len);
  std::copy(offPolicImpW.begin(), offPolicImpW.end(), buf); buf += seq_len;

  assert(KullbLeibDiv.size() <= seq_len); KullbLeibDiv.resize(seq_len);
  std::copy(KullbLeibDiv.begin(), KullbLeibDiv.end(), buf); buf += seq_len;

  /////////////////////////////////////////////////////////////////////////////

  *(buf++) = nOffPol; //fval
  *(buf++) = MSE; //fval
  *(buf++) = sumKLDiv; //fval
  *(buf++) = totR; //fval

  char * charPos = (char*) buf;
  memcpy(charPos, &       ended, sizeof(bool)); charPos += sizeof(bool);
  memcpy(charPos, &          ID, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &just_sampled, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(charPos, &      prefix, sizeof(Uint)); charPos += sizeof(Uint);
  memcpy(charPos, &     agentID, sizeof(Uint)); charPos += sizeof(Uint);

  // assert(buf - ret.data() == (ptrdiff_t) totalSize);
  return ret;
}

void Sequence::save(FILE * f, const Uint dS, const Uint dA, const Uint dP) {
  const Uint seq_len = states.size();
  fwrite(& seq_len, sizeof(Uint), 1, f);
  Fvec buffer = packSequence(dS, dA, dP);
  fwrite(buffer.data(), sizeof(Fval), buffer.size(), f);
}

void Sequence::unpackSequence(const std::vector<Fval>& data, const Uint dS,
  const Uint dA, const Uint dP)
{
  const Uint seq_len = Sequence::computeTotalEpisodeNstep(dS,dA,dP,data.size());
  const Fval* buf = data.data();
  assert(states.size() == 0);
  for (Uint i = 0; i<seq_len; ++i) {
    states.push_back(std::vector<nnReal>(buf, buf+dS));
    rewards.push_back(buf[dS]); buf += dS + 1;
    actions.push_back(std::vector<Fval>(buf, buf+dA)); buf += dA;
    policies.push_back(std::vector<Fval>(buf, buf+dP)); buf += dP;
  }

  /////////////////////////////////////////////////////////////////////////////
  Q_RET = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  action_adv = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  state_vals = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  /////////////////////////////////////////////////////////////////////////////
  SquaredError = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  offPolicImpW = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  KullbLeibDiv = std::vector<Fval>(buf, buf + seq_len); buf += seq_len;
  /////////////////////////////////////////////////////////////////////////////
  priorityImpW = std::vector<float>(seq_len, 1);
  /////////////////////////////////////////////////////////////////////////////
  nOffPol  = *(buf++);
  MSE      = *(buf++);
  sumKLDiv = *(buf++);
  totR     = *(buf++);

  const char * charPos = (const char *) buf;
  memcpy(&       ended, charPos, sizeof(bool)); charPos += sizeof(bool);
  memcpy(&          ID, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&just_sampled, charPos, sizeof(Sint)); charPos += sizeof(Sint);
  memcpy(&      prefix, charPos, sizeof(Uint)); charPos += sizeof(Uint);
  memcpy(&     agentID, charPos, sizeof(Uint)); charPos += sizeof(Uint);
  //assert(buf-data.data()==(ptrdiff_t)computeTotalEpisodeSize(dS,dA,dP,seq_len));
}

int Sequence::restart(FILE * f, const Uint dS, const Uint dA, const Uint dP)
{
  Uint seq_len = 0;
  if(fread(& seq_len, sizeof(Uint), 1, f) != 1) return 1;
  const Uint totalSize = Sequence::computeTotalEpisodeSize(dS, dA, dP, seq_len);
  std::vector<Fval> buffer(totalSize);
  if(fread(buffer.data(), sizeof(Fval), totalSize, f) != totalSize)
    die("mismatch");
  unpackSequence(buffer, dS, dA, dP);
  return 0;
}

inline isDifferent(const double& a, const double& b) {
  return std::fabs(a-b) > std::numeric_limits<double>::epsilon();
}
inline isDifferent(const float& a, const float& b) {
  return std::fabs(a-b) > std::numeric_limits<float>::epsilon();
}
template<typename T>
inline isDifferent(const std::vector<T>& a, const std::vector<T>& b) {
  if(a.size() not_eq b.size()) return true;
  for(size_t i=0; i<b.size(); ++i) if( isDifferent(a[i], b[i]) ) return true;
}

bool Sequence::isEqual(const Sequence * const S) const
{
  if( isDifferent(S->states      , states      ) ) return false;
  if( isDifferent(S->actions     , actions     ) ) return false;
  if( isDifferent(S->policies    , policies    ) ) return false;
  if( isDifferent(S->rewards     , rewards     ) ) return false;
  if( isDifferent(S->Q_RET       , Q_RET       ) ) return false;
  if( isDifferent(S->action_adv  , action_adv  ) ) return false;
  if( isDifferent(S->state_vals  , state_vals  ) ) return false;
  if( isDifferent(S->SquaredError, SquaredError) ) return false;
  if( isDifferent(S->offPolicImpW, offPolicImpW) ) return false;
  if( isDifferent(S->KullbLeibDiv, KullbLeibDiv) ) return false;
  if( isDifferent(S->nOffPol     , nOffPol     ) ) return false;
  if( isDifferent(S->MSE         , MSE         ) ) return false;
  if( isDifferent(S->sumKLDiv    , sumKLDiv    ) ) return false;
  if( isDifferent(S->totR        , totR        ) ) return false;
  if(S->ended        not_eq ended       ) return false;
  if(S->ID           not_eq ID          ) return false;
  if(S->just_sampled not_eq just_sampled) return false;
  if(S->prefix       not_eq prefix      ) return false;
  if(S->agentID      not_eq agentID     ) return false;
}

}
