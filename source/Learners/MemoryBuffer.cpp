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
  mastersComm(_s.mastersComm), env(_env), bTrain(_s.bTrain), bWriteToFile(!(_s.samplesFile=="none")),
  bSampleSeq(_s.bSampleSequences), maxTotSeqNum(_s.maxTotSeqNum),
  maxSeqLen(_s.maxSeqLen),minSeqLen(_s.minSeqLen), nAppended(_s.appendedObs),
  batchSize(_s.batchSize), policyVecDim(_s.policyVecDim),
  learn_rank(_s.learner_rank), learn_size(_s.learner_size),
  generators(_s.generators), sI(_env->sI), aI(_env->aI),
  _agents(_env->agents)
{
  mean.resize(sI.dimUsed, 0);
  std.resize(sI.dimUsed, 1);
  invstd.resize(sI.dimUsed, 1);
  if (sI.mean.size()){
    Uint k = 0;
    for(Uint i=0; i<sI.dim; i++) {
      if(sI.inUse[i]) {
        mean[k] = sI.mean[i];
        std[k] = sI.scale[i];
        invstd[k] = 1./sI.scale[i];
        k++;
      }
    }
    assert(k == sI.dimUsed);
  }

  assert(_s.nAgents>0);
  inProgress.resize(_s.nAgents);
  for (int i=0; i<_s.nAgents; i++) inProgress[i] = new Sequence();
  gen = new Gen(&generators[0]);
  Set.reserve(maxTotSeqNum);
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
int MemoryBuffer::add_state(const Agent&a)
{
  int ret=0; //ret is 1 if state will be placed as first in a sequence
  #if 0
    if (inProgress[a.ID]->tuples.size() && a.Status == 1) {
      //prev sequence not empty, yet received an initial state, push back prev
      warn("Detected partial sequence\n");
      push_back(a.ID);
      ret = 1;
    } else if(inProgress[a.ID]->tuples.size()==0) {
      if(a.Status not_eq 1) die("Missing initial state\n");
      ret = 1; //status is 1
    }
  #endif

  #ifndef NDEBUG // check that last new state and new old state are the same
  if(inProgress[a.ID]->tuples.size()) {
    bool same = true;
    const vector<Real> vecSold = a.sOld->copy_observed();
    const Tuple*const last = inProgress[a.ID]->tuples.back();
    for (Uint i=0; i<sI.dimUsed && same; i++) //scaled vec only has used dims:
      same = same && std::fabs(last->s[i]-vecSold[i]) < 1e-8;
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
void MemoryBuffer::add_action(const Agent& a, vector<Real> pol) const
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
  a.a->set(vector<Real>(aI.dim,0));
  add_action(a, vector<Real>(policyVecDim, 0) );
  push_back(a.ID);
}

// update the second order moment of the rewards in the memory buffer
void MemoryBuffer::updateRewardsStats()
{
  if(!bTrain) return; //if not training, keep the stored values

  long double count = 0, newstdvr = 0;
  #pragma omp parallel for schedule(dynamic) reduction(+ : count, newstdvr)
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
  const Real weight = first_pass ? 1 : 0.01;
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
  if(inProgress[agentId]->tuples.size() > minSeqLen ) {
    lock_guard<mutex> lock(dataset_mutex);

    inProgress[agentId]->ID = nSeenSequences++;
    nSeenTransitions += inProgress[agentId]->ndata();
    assert(nSequences == Set.size());

    if (nSequences >= adapt_TotSeqNum)
      Buffered.push_back(inProgress[agentId]);
    else
      pushBackSequence(inProgress[agentId]);
  } else {
    printf("Trashing %lu obs.\n",inProgress[agentId]->tuples.size());
    fflush(0);
    _dispose_object(inProgress[agentId]);
  }

  inProgress[agentId] = new Sequence();
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
void MemoryBuffer::push_back_sequential(const int & agentId)
{
  if(inProgress[agentId]->tuples.size() > minSeqLen ) {
    lock_guard<mutex> lock(dataset_mutex);

    inProgress[agentId]->ID = nSeenSequences++;
    nSeenTransitions += inProgress[agentId]->ndata();
    assert(nSequences == Set.size());

    if (nSequences >= adapt_TotSeqNum) {
      const Uint ind = iOldestSaved >= Set.size()? 0 : iOldestSaved;
      iOldestSaved++;
      removeSequence(ind);
      addSequence(ind, inProgress[agentId]);
    } else
      pushBackSequence(inProgress[agentId]);
  } else {
    printf("Trashing %lu obs.\n",inProgress[agentId]->tuples.size());
    fflush(0);
    _dispose_object(inProgress[agentId]);
  }

  inProgress[agentId] = new Sequence();
}

//If observed sequences exceeded capacity of memory buffer, placed into buffer
//Here inserted in dataset by removing old/less important seqs (see function)
//Also, algorithms might want to adap size of memory buffer
//Concurrently, I also update the value of the standard deviation of the reward
void MemoryBuffer::updateActiveBuffer()
{
  if(Buffered.size() || adapt_TotSeqNum<Set.size())
    insertBufferedSequences();

  #ifdef importanceSampling
    updateImportanceWeights();
  #endif
}

void MemoryBuffer::insertBufferedSequences()
{
  // 4 steps:
  //1) if Set.size() < adapt_TotSeqNum, add buffer to Set
  //2) optional, sort transitions in Set
  //3) if Set.size() > adapt_TotSeqNum remove those at the back
  //this implies that adaptive Set size is only suppoted if i sort the sequences
  //4) add the buffered transitions
  while( Set.size()<adapt_TotSeqNum && Buffered.size()>0 ) {
    const auto bufTransition = Buffered.back();
    assert(bufTransition not_eq nullptr);
    pushBackSequence(bufTransition);

    Buffered.back() = nullptr;
    Buffered.pop_back();
  }

  #ifndef RESORT_SEQS
  if(Set.size() > adapt_TotSeqNum)
  #endif
    sortSequences();

  //If algorithm specifies a certain memory buffer size and never changes it
  // this would be never called:
  if(Set.size() > adapt_TotSeqNum) {
    while(Set.size() > adapt_TotSeqNum) popBackSequence();

    Set.reserve(maxTotSeqNum); //just to make sure, should be unnecessary
    assert(nSequences<=adapt_TotSeqNum);
    //tell code that buffered seqs will be added at the end
    iOldestSaved = nSequences - Buffered.size();
  }

  if( Buffered.size() == 0 ) return;
  nTransitionsInBuf=0; nTransitionsDeleted=0;
  for(Uint j=Buffered.size(); j>0; j--) {
    assert(Buffered.size() == j);
    const Uint ind = (iOldestSaved >= Set.size()) ? 0 : iOldestSaved;
    iOldestSaved++;
    nTransitionsDeleted += Set[ind]->ndata();
    nTransitionsInBuf += Buffered.back()->ndata();
    removeSequence(ind);
    addSequence(ind, Buffered.back());
    Buffered.back() = nullptr;
    Buffered.pop_back();
  }
}

// Sort sequences based on SquaredError array. Places large errors first
void MemoryBuffer::sortSequences()
{
  #ifndef NDEBUG
    printf("Sorting %lu sequences\n", Set.size());
    for(Uint i=0; i<Set.size(); i++) {
      assert(*(Set.begin()+i) not_eq nullptr);
      cout<<i<<" "<<Set[i]->MSE<<" "<<Set[i]->ndata()<<endl;
    }
  #endif

  #pragma omp parallel for schedule(dynamic)
  for(Uint i=0; i<Set.size(); i++) {
    assert(Set[i] not_eq nullptr);
    Set[i]->MSE = 0.;
    for(Uint j=0; j<Set[i]->ndata(); j++)
     #if 0 //sort by max error
        Set[i]->MSE = std::max(Set[i]->MSE, Set[i]->tuples[j]->SquaredError);
     #else //sort by mean error: penalizes long sequences
        Set[i]->MSE += Set[i]->SquaredError[j]/Set[i]->ndata();
     #endif
  }

  const auto compare = [&] (const Sequence*const a, const Sequence*const b) {
    assert(a->MSE>=0 && b->MSE>=0);
    return a->MSE<1e-16? true : (b->MSE<1e-16? false : (a->MSE > b->MSE) ); };
  //__gnu_parallel::
  std::sort(Set.begin(), Set.begin()+nSequences, compare);
  assert(Set.front()->MSE >= Set.back()->MSE || Set.front()->MSE<1e-16);

  iOldestSaved = nSequences - Buffered.size();
}

// remove sequences from Seq if more than maxFrac of samples have importance
// weight (from vetor offPol_weight) either <1/CmaxRho or >CmaxRho
Uint MemoryBuffer::prune(const Real maxFrac, const Real CmaxRho)
{
  assert(CmaxRho>1);

  int _ID = Set[0]->ID;
  Uint ret = 0;

  #pragma omp parallel for schedule(dynamic) reduction(+:ret) reduction(min:_ID)
  for(Uint i = 0; i < Set.size(); i++)
  {
    Real numOver = 0;
    for(Uint j=0; j<Set[i]->ndata(); j++)
    {
      if( Set[i]->offPol_weight[j] > CmaxRho )   numOver += 1;
      else
      if( 1/CmaxRho > Set[i]->offPol_weight[j] ) numOver += 1;
    }
    _ID = std::min(Set[i]->ID, _ID);
    // overwrite MSE as `boolean` depending on whether seq should be removed
    if(numOver/(Real)Set[i]->ndata() > maxFrac) {
      Set[i]->MSE = 1;
      ret++;
    } else
      Set[i]->MSE = 0;
  }

  assert(_ID>=0);
  nPruned+=ret;
  minInd = _ID;

  for(int i = (int)Set.size()-1; i >= 0; i--)
    if( Set[i]->MSE > 0.5 ) {
      std::swap(Set[i], Set.back());
      popBackSequence();
    }
  return ret;
}

Real MemoryBuffer::prune2(const Real CmaxRho, const Uint maxN)
{
  assert(CmaxRho>1);
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < Set.size(); i++) {
    Real numOver = 0, opcW = 0;
    for(Uint j=0; j<Set[i]->ndata(); j++) {
      const Real obsOpcW = Set[i]->offPol_weight[j];
      const Real obsDist = std::max(obsOpcW, 1/obsOpcW);
      assert(obsOpcW > 0 && obsDist >= 1);
      if( obsDist > CmaxRho ) numOver += 1;
      opcW += obsDist;
    }
    Set[i]->OPW = opcW/Set[i]->ndata();
    Set[i]->MSE = numOver;
  }

  const auto isAbeforeB = [&] (const Sequence*const a, const Sequence*const b) {
    return a->OPW < b->OPW;
  };
  std::sort(Set.begin(), Set.end(), isAbeforeB);
  assert(Set.front()->OPW <= Set.back()->OPW);

  const Uint nB4 = Set.size();
  Uint cntN = 0;
  Real nOffPol = 0;
  int _ID = Set[0]->ID;

  for(Uint i = 0; i < Set.size(); i++) {
    if(cntN<maxN) {
      cntN += Set[i]->ndata();
      nOffPol += Set[i]->MSE;
      _ID = std::min(Set[i]->ID, _ID);
    } else { //not really smart
      std::swap(Set[i], Set.back());
      popBackSequence();
    }
  }

  assert(_ID>=0 && cntN == nTransitions);
  minInd = _ID;
  nPruned += nB4-Set.size();
  return nOffPol;
}

Real MemoryBuffer::prune3(const Real CmaxRho, const Uint maxN)
{
  assert(CmaxRho>1);
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < Set.size(); i++) {
    Real numOver = 0;
    for(Uint j=0; j<Set[i]->ndata(); j++) {
      const Real obsOpcW = Set[i]->offPol_weight[j];
      const Real obsDist = std::max(obsOpcW, 1/obsOpcW);
      assert(obsOpcW > 0 && obsDist >= 1);
      if( obsDist > CmaxRho ) numOver += 1;
    }
    Set[i]->MSE = numOver;
  }

  const auto isAbeforeB = [&] (const Sequence*const a, const Sequence*const b) {
    return a->ID > b->ID;
  };
  std::sort(Set.begin(), Set.end(), isAbeforeB);

  const Uint nB4 = Set.size();
  Uint cntN = 0;
  Real nOffPol = 0;

  Uint i = 0;
  for(i=0; i < Set.size(); i++)
    if(cntN<maxN) {
      cntN += Set[i]->ndata();
      nOffPol += Set[i]->MSE;
    } else break;
  assert(i<=Set.size());
  while(i not_eq Set.size()) popBackSequence();
  assert(cntN == nTransitions);
  minInd = Set.back()->ID;
  nPruned += nB4-Set.size();
  return nOffPol;
}

void MemoryBuffer::updateImportanceWeights()
{
  const Uint ndata = bSampleSeq ? nSequences : nTransitions;
  vector<Uint> inds(ndata);
  std::iota(inds.begin(), inds.end(), 0);
  vector<Real> errors(ndata), Ps(ndata), Ws(ndata);

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
      if(bSampleSeq) Set[i]->offPol_weight[j] = Ws[i];
      else Set[i]->offPol_weight[j] = Ws[k++];
}

void MemoryBuffer::getMetrics(ostringstream&fileOut, ostringstream&screenOut)
{
  fileOut<<nSequences<<" "<<nTransitions<<" "<<nSeenSequences<<" "
         <<nSeenTransitions<<" "<<adapt_TotSeqNum<<" "<<1./invstd_reward<<" ";
         //<<nSequencesInBuf<<" "<<nSequencesDeleted<<endl;
  screenOut<<" nSeq:"<<nSequences<<" nObs:"<<nTransitions
  <<" (seen Seq:"<<nSeenSequences<<" Obs:"<<nSeenTransitions
  <<") maxSeq:"<<adapt_TotSeqNum<<" stdRew:"<<1./invstd_reward
  <<" nPruned:"<<nPruned<<" minInd:"<<minInd;
  nPruned=0;
  //<<nSequencesInBuf<<" "<<nSequencesDeleted<<endl;
}

void MemoryBuffer::restart()
{
  return;
  const Uint writesize = 3 +sI.dim +aI.dim +policyVecDim;
  int agentID = 0, info = 0, sampID = 0;
  vector<Real> policy(policyVecDim), action(aI.dim), state(sI.dim);
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
      if(info == 2) push_back_sequential(0);
    }
    if(_agents[0]->getStatus() not_eq 2) push_back_sequential(0); //(agentID is 0)
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
  #ifndef importanceSampling
    std::uniform_int_distribution<int> distObs(0, nTransitions-1);
    const Uint ind = distObs(generators[thrID]);
  #else
    const Uint ind = (*dist)(generators[thrID]);
  #endif
  indexToSample(ind, seq, obs);
}

void MemoryBuffer::sampleSequence(Uint& seq, const int thrID)
{
  #ifndef importanceSampling
    std::uniform_int_distribution<int> distSeq(0, nSequences-1);
    seq = distSeq(generators[thrID]);
  #else
    seq = (*dist)(generators[thrID]);
  #endif
}

vector<Uint> MemoryBuffer::sampleSequences(const Uint N)
{
  vector<Uint> inds(nSequences);
  std::iota(inds.begin(), inds.end(), 0);
  std::shuffle(inds.begin(), inds.end(), generators[0]);
  return vector<Uint>(&inds[0], &inds[N]);
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
