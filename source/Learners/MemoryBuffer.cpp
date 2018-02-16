/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 */

#include "MemoryBuffer.h"
#include <dirent.h>
#include <iterator>
#include <parallel/algorithm>

MemoryBuffer::MemoryBuffer(Environment* const _env, Settings & _s):
 mastersComm(_s.mastersComm), env(_env), bWriteToFile(_s.samplesFile!="none"),
 bTrain(_s.bTrain), bSampleSeq(_s.bSampleSequences), nAppended(_s.appendedObs),
 batchSize(_s.batchSize), maxTotObsNum(_s.maxTotObsNum), nThreads(_s.nThreads),
 policyVecDim(_s.policyVecDim), sI(env->sI), aI(env->aI), _agents(env->agents),
 generators(_s.generators), mean(sI.inUseMean()), invstd(sI.inUseInvStd()),
 std(sI.inUseStd()), learn_rank(_s.learner_rank), learn_size(_s.learner_size),
 gamma(_s.gamma) {
  assert(_s.nAgents>0);
  inProgress.resize(_s.nAgents);
  for (int i=0; i<_s.nAgents; i++) inProgress[i] = new Sequence();
  gen = new Gen(&generators[0]);
  Set.reserve(maxTotObsNum);
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
int MemoryBuffer::add_state(const Agent&a)
{
  int ret=0; //ret is 1 if state will be placed as first in a sequence
  #if 1
    if (inProgress[a.ID]->tuples.size() && a.Status == 1) {
      //prev sequence not empty, yet received an initial state, push back prev
      warn("Detected partial sequence\n");
      push_back(a.ID);
      ret = 1;
    } else if(inProgress[a.ID]->tuples.size()==0) {
      if(a.Status not_eq 1) die("Missing initial state\n");
    }
  #endif

  #ifndef NDEBUG // check that last new state and new old state are the same
  if(inProgress[a.ID]->tuples.size()) {
    bool same = true;
    const Rvec vecSold = a.sOld->copy_observed();
    const Tuple*const last = inProgress[a.ID]->tuples.back();
    for (Uint i=0; i<sI.dimUsed && same; i++) //scaled vec only has used dims:
      same = same && std::fabs(last->s[i]-vecSold[i]) < 1e-4;
    if (!same) { //create new sequence
      warn("Detected partial sequence"); push_back(a.ID); ret = 1; }
  }
  #endif

  env->pickReward(a);
  inProgress[a.ID]->ended = a.Status==2;
  inProgress[a.ID]->add_state(a.s->copy_observed(), a.r);
  return ret;
}

// Once network picked next action, call this method
void MemoryBuffer::add_action(const Agent& a, Rvec pol) const
{
  if(pol.size() not_eq policyVecDim) die("add_action");
  inProgress[a.ID]->add_action(a.a->vals, pol);
  if(bWriteToFile || a.ID == 0 ) a.writeData(learn_rank, pol);
}

// If the state is terminal, instead of calling `add_action`, call this:
void MemoryBuffer::terminate_seq(const Agent&a)
{
  assert(a.Status==2);
  assert(inProgress[a.ID]->tuples.back()->mu.size() == 0);
  assert(inProgress[a.ID]->tuples.back()->a.size()  == 0);
  a.a->set(Rvec(aI.dim,0));
  add_action(a, Rvec(policyVecDim, 0) );
  push_back(a.ID);
}

// update the second order moment of the rewards in the memory buffer
void MemoryBuffer::updateRewardsStats()
{
  if(!bTrain) return; //if not training, keep the stored values

  long double count = 0, newstdvr = 0;
  #pragma omp parallel for reduction(+ : count, newstdvr) //schedule(dynamic)
  for(Uint i=0; i<Set.size(); i++)
    for(Uint j=0; j<Set[i]->ndata(); j++) {
      newstdvr += std::pow(Set[i]->tuples[j+1]->r, 2);
      count++;
    }

  //add up gradients across nodes (masters)
  if (learn_size > 1) {
    const bool firstUpdate = rewRequest == MPI_REQUEST_NULL;
    if(not firstUpdate) MPI_Wait(&rewRequest, MPI_STATUS_IGNORE);
    // prepare an allreduce with the current data:
    partial_sum[0] = count;
    partial_sum[1] = newstdvr;
    //use result from prev reduce to update rewards (before calling new reduce)
    count = rew_reduce_result[0];
    newstdvr = rew_reduce_result[1];

    MPI_Iallreduce(partial_sum, rew_reduce_result, 2, MPI_LONG_DOUBLE, MPI_SUM,
                   mastersComm, &rewRequest);
    // if no reduction done, partial sums are meaningless
    if(firstUpdate) return;
  }

  if(count<batchSize) return;
  //const Real stdev_reward = std::sqrt((newstdvr-newmeanr*newmeanr/count)/count);
  const Real stdev_reward = std::sqrt(newstdvr/count);
  const Real weight = 1;//first_pass ? 1 : 0.01;
  first_pass = false;
  //mean_reward = (1-weight)*mean_reward +weight*newmeanr/count;
  invstd_reward = (1-weight)*invstd_reward +weight/stdev_reward;
  #ifndef NDEBUG
  Uint cntSamp = 0;
  for(Uint i=0; i<Set.size(); i++) {
    assert(Set[i] not_eq nullptr);
    cntSamp += Set[i]->ndata();
  }
  assert(cntSamp==nTransitions);
  #endif
  //printf("new invstd reward %g\n",invstd_reward);
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
// This is the thread-safe version that if Set is full instead of messing with
// it, this function stores the seq onto a buffer.
// REQUIRES CALLING insertBufferedSequences() once a serial region is reached
void MemoryBuffer::push_back(const int & agentId)
{
  if(inProgress[agentId]->tuples.size() > 2 ) {
    inProgress[agentId]->ID = readNSeenSeq();
    inProgress[agentId]->SquaredError.resize(inProgress[agentId]->ndata(), 0);
    inProgress[agentId]->offPol_weight.resize(inProgress[agentId]->ndata(), 1);
    #pragma omp atomic
    nSeenSequences++;
    #pragma omp atomic
    nSeenTransitions += inProgress[agentId]->ndata();

    pushBackSequence(inProgress[agentId]);
  } else {
    printf("Trashing %lu obs.\n",inProgress[agentId]->tuples.size());
    fflush(0);
    _dispose_object(inProgress[agentId]);
  }

  if(readNSeq() >= maxTotObsNum)
    die("maxTotObsNum setting is too low for given problem");

  inProgress[agentId] = new Sequence();
}

void MemoryBuffer::prune(const Real CmaxRho, const FORGET ALGO)
{
  //checkNData();
  //no need for pruning:
  if(nTransitions <= maxTotObsNum) return;

  assert(CmaxRho>1);
  vector<pair<int, Real>> delete_location(nThreads, {-1, 2e20});
  vector<pair<int,  int>> oldest_sequence(nThreads, {-1, nSeenSequences});
  Real _nOffPol = 0, _totMSE = 0;
  const Real invC = 1/CmaxRho, EPS = 1e-9;
  #pragma omp parallel reduction(+ : _nOffPol,_totMSE)
  {
    const int thrID = omp_get_thread_num();
    #pragma omp for //schedule(dynamic)
    for(Uint i = 0; i < Set.size(); i++)
    {
      if(Set[i]->just_sampled >= 0) {
        Set[i]->nOffPol = 0; Set[i]->MSE = 0;
        for(Uint j=0; j<Set[i]->ndata(); j++) {
          Set[i]->MSE += Set[i]->SquaredError[j];
          const Real W = Set[i]->offPol_weight[j];
          assert(Set[i]->SquaredError[j]>=0 && W>=0);
          if(W>CmaxRho || W<invC) Set[i]->nOffPol += 1;
        }
        Set[i]->just_sampled = -1;
      }
      assert(ALGO == OLDEST || ALGO == MAXERROR); //TODO generalize
      //either delete seq with smallest index or with largest "error"
      const Real W=ALGO==OLDEST? Set[i]->ID : Set[i]->ndata()/(EPS+Set[i]->MSE);
      _nOffPol += Set[i]->nOffPol; _totMSE += Set[i]->MSE;
      if(Set[i]->ID < oldest_sequence[thrID].second) {
        oldest_sequence[thrID].second = Set[i]->ID;
        oldest_sequence[thrID].first = i;
      }
      //locate smallest sequence id/mse/impW
      if(W < delete_location[thrID].second) {
        delete_location[thrID].second = W;
        delete_location[thrID].first = i;
      }
    }
  }

  nOffPol = _nOffPol; totMSE = _totMSE;
  const Uint nB4 = Set.size();
  int deli = -1, oldi = -1, oldj = nSeenSequences; Real delv = 2e20;
  for(const auto&P: oldest_sequence)
    if(oldj>P.second && P.first>=0) {
      oldi = P.first; oldj = P.second;
    }
  for(const auto&P: delete_location)
    if(delv>P.second && P.first>=0) {
      deli = P.first; delv = P.second;
    }
  minInd = oldj;
  assert(deli>=0 && deli<(int)Set.size());
  assert(oldi>=0 && oldi<(int)Set.size());
  if(Set[oldi]->ID + (int)Set.size() < Set[deli]->ID) deli = oldi;
  if(nTransitions-Set[deli]->ndata() > maxTotObsNum) {
    //cout<< "delete:"<<Set[deli]->ID<<" "<<Set[deli]->MSE<<" "<<Set[deli]->nOffPol
    //    <<" oldest:"<<Set[oldi]->ID<<" "<<Set[oldi]->MSE<<" "<<Set[oldi]->nOffPol<<endl;
    std::swap(Set[deli], Set.back());
    popBackSequence();
  }
  nPruned += nB4-Set.size();

  #ifdef IMPORTSAMPLE
    updateImportanceWeights();
  #endif
}

void MemoryBuffer::updateImportanceWeights()
{
  const Uint ndata = bSampleSeq ? nSequences : nTransitions;
  vector<Uint> inds(ndata);
  std::iota(inds.begin(), inds.end(), 0);
  Rvec errors(ndata), Ps(ndata), Ws(ndata);

  for(Uint i=0, k=0; i<Set.size(); i++)
    for(Uint j=0; j<Set[i]->ndata(); j++)
      //if(bSampleSeq) errors[i] = std::max(maxerr, Set[i]->SquaredError[j]);
      if(bSampleSeq) errors[i] += Set[i]->SquaredError[j]/Set[i]->ndata();
      else errors[k++] = Set[i]->SquaredError[j];

  #if 1
    // As in the prioritized exp replay paper, importance weight is ~sqrt()
    // of 1/rank of the sample's MSE error (ie big errors are sampled more).
    const auto comp=[&](const Uint a,const Uint b){return errors[a]>errors[b];};
    __gnu_parallel::sort(inds.begin(), inds.end(), comp);
    assert(errors[inds.front()] >= errors[inds.back()]);

    //sort in decreasing order of the error. Points with zero error
    //(which means that they are not yet processed) are put at the top:
    #pragma omp parallel for
    for(Uint i=0; i<ndata; i++)
      Ps[inds[i]] = errors[inds[i]]>0 ? std::sqrt(1./(i+1.)) : 1;
  #else
    Ps = errors; //otherwise we use error magnitude as probability
  #endif

  Real minP = 2, sumP = 0;
  #pragma omp parallel for reduction(min: minP) reduction(+: sumP)
  for(Uint i=0; i<ndata; i++) { minP = std::min(minP, Ps[i]); sumP += Ps[i]; }
  assert(minP<=1 && sumP>0);

  #pragma omp parallel for
  for(Uint i=0; i<ndata; i++) { Ws[i] = minP / Ps[i]; Ps[i] = Ps[i ] /sumP; }

  if(dist not_eq nullptr) delete dist;
  dist = new std::discrete_distribution<Uint>(Ps.begin(), Ps.end());

  for(Uint i=0, k=0; i<Set.size(); i++)
    for(Uint j=0; j<Set[i]->ndata(); j++)
      if(bSampleSeq) Set[i]->imp_weight[j] = Ws[i];
      else Set[i]->imp_weight[j] = Ws[k++];
}

void MemoryBuffer::getMetrics(ostringstream& buff)
{
  buff<<" "<<std::setw(8)<<nSequences;
  buff<<" "<<std::setw(8)<<nTransitions;
  buff<<" "<<std::setw(8)<<nSeenSequences;
  buff<<" "<<std::setw(8)<<nSeenTransitions;
  buff<<" "<<std::setw(7)<<minInd;
  buff<<" "<<std::setw(7)<<(int)nOffPol;
  buff<<" "<<std::setw(6)<<std::setprecision(2)<<1./invstd_reward;
  buff<<" "<<std::setw(6)<<std::setprecision(0)<<totMSE;
  nPruned=0;
}
void MemoryBuffer::getHeaders(ostringstream& buff)
{
  buff <<
  "| curSeq | curObs | totSeq | totObs | minID | nOffP | stdR | tMSE ";
}

void MemoryBuffer::restart()
{
  return;
  const Uint writesize = 3 +sI.dim +aI.dim +policyVecDim;
  int agentID = 0, info = 0, sampID = 0;
  Rvec policy(policyVecDim), action(aI.dim), state(sI.dim);
  char cpath[256];

  while (true) {
    sprintf(cpath, "obs_rank%02d_agent%03d.raw", learn_rank, agentID);
    FILE*pFile = fopen(cpath, "rb");
    if(pFile==NULL) { printf("Couldnt open file %s.\n", cpath); break; }

    float* buf = (float*) malloc(writesize*sizeof(float));
    while(true) {
      size_t ret = fread(buf, sizeof(float), writesize, pFile);
      if (ret == 0) break;
      if (ret != writesize) _die("Error reading datafile %s", cpath);
      Uint k = 0;
      info = buf[k++]; sampID = buf[k++];

      if((sampID==0) != (info==1)) die("Mismatch in transition counter\n");
      if(sampID!=_agents[0]->transitionID+1 && info!=1) die(" transitionID");

      for(Uint i=0; i<sI.dim; i++) state[i]  = buf[k++];
      for(Uint i=0; i<aI.dim; i++) action[i] = buf[k++];
      Real reward = buf[k++];
      for(Uint i=0; i<policyVecDim; i++) policy[i] = buf[k++];
      assert(k == writesize);

      _agents[0]->update(info, state, reward);
      add_state(*_agents[0]);
      inProgress[0]->add_action(action, policy);
      if(info == 2) push_back(0);
    }
    if(_agents[0]->getStatus() not_eq 2) push_back(0); //(agentID is 0)
    fclose(pFile); free(buf);
    agentID++;
  }
  if(agentID==0) { printf("Couldn't restart transition data.\n"); } //return 1;
  //push_back(0);
  printf("Found %d broken seq out of %d/%d.\n",nBroken,nSequences,nTransitions);
  //return 0;
}

// number of returned samples depends on size of seq! (== to that of trans)
void MemoryBuffer::sampleTransition(Uint& seq, Uint& obs, const int thrID)
{
  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<int> distObs(0, readNData()-1);
    const Uint ind = distObs(generators[thrID]);
  #else
    const Uint ind = (*dist)(generators[thrID]);
  #endif
  indexToSample(ind, seq, obs);
}

Uint MemoryBuffer::sampleTransition(const Uint seq, const int thrID)
{
  std::uniform_int_distribution<Uint> distObs(0, Set[seq]->ndata()-1);
  return distObs(generators[thrID]);
}

void MemoryBuffer::sampleSequence(Uint& seq, const int thrID)
{
  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<int> distSeq(0, readNSeq()-1);
    seq = distSeq(generators[thrID]);
  #else
    seq = (*dist)(generators[thrID]);
  #endif
}

vector<Uint> MemoryBuffer::sampleSequences(const Uint N)
{
  assert(N<=readNSeq());
  vector<Uint> inds;
  if( N*5 < readNSeq() ) {
    inds.resize(N);
    for(Uint i=0; i<N; i++) sampleSequence(inds[i], 0);
  } else { // if N is large, make sure we do not repeat indices
    inds.resize(readNSeq());
    std::iota(inds.begin(), inds.end(), 0);
    std::shuffle(inds.begin(), inds.end(), generators[0]);
    inds = vector<Uint>(&inds[0], &inds[N]);
  }
  const auto compare = [&](Uint a, Uint b) {
    return Set[a]->ndata() > Set[b]->ndata();
  };
  std::sort(inds.begin(), inds.end(), compare);
  return inds;
}

void MemoryBuffer::indexToSample(const int nSample, Uint& seq, Uint& obs) const
{
  int k = 0, back = 0, indT = Set[0]->ndata();
  while (nSample >= indT) {
    assert(k+2<=(int)Set.size());
    back = indT;
    indT += Set[++k]->ndata();
  }
  assert(nSample>=back && Set[k]->ndata()>(Uint)nSample-back);
  seq = k; obs = nSample-back;
}
