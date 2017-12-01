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
  mastersComm(_s.mastersComm), env(_env), bNormalize(_s.bNormalize),
  bTrain(_s.bTrain), bWriteToFile(!(_s.samplesFile=="none")),
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

void MemoryBuffer::sortSequences()
{
  #ifndef NDEBUG
    printf("Sorting %lu sequences\n", Set.size());
    for(Uint i=0; i<Set.size(); i++) {
      assert(*(Set.begin()+i) not_eq nullptr);
      printf("%u %f %u\n",i,Set[i]->MSE, Set[i]->ndata());
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

void MemoryBuffer::terminate_seq(const Agent&a)
{
  assert(a.Status==2);
  assert(inProgress[a.ID]->tuples.back()->mu.size() == 0);
  assert(inProgress[a.ID]->tuples.back()->a.size()  == 0);
  inProgress[a.ID]->add_action(vector<Real>(aI.dim,0), vector<Real>(policyVecDim,0));
  push_back(a.ID);
}

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

  if(inProgress[a.ID]->tuples.size() >= maxSeqLen) { //upper limit to seq length
    const Tuple* const last = inProgress[a.ID]->tuples.back();
    Tuple * t = new Tuple(last); //copy last state
    push_back(a.ID); //create new sequence
    inProgress[a.ID]->tuples.push_back(t);
  }
  env->pickReward(a);
  inProgress[a.ID]->ended = a.Status==2;
  inProgress[a.ID]->add_state(a.s->copy_observed(), a.r);
  return ret;
}

void MemoryBuffer::add_action(const Agent& a, vector<Real> pol) const
{
  if(pol.size() == 0) pol.resize(policyVecDim, 0);
  assert(pol.size() == policyVecDim);
  inProgress[a.ID]->add_action(a.a->vals, pol);
  if(bWriteToFile || a.ID == 0 ) a.writeData(learn_rank, pol);
}

void MemoryBuffer::updateRewardsStats()
{
  if(!bTrain || !bNormalize) return; //if not training, keep the stored values

  long double count = 0, newstdvr = 0;
  #pragma omp parallel for schedule(dynamic) reduction(+ : count, newstdvr)
  for(Uint i=0; i<Set.size(); i++)
    for(Uint j=0; j<Set[i]->ndata(); j++) {
      newstdvr += std::pow(Set[i]->tuples[j+1]->r, 2);
      count++;
    }

  //add up gradients across nodes (masters)
  if (learn_size > 1) {
    long double global[2] = {count, newstdvr};
    MPI_Allreduce(MPI_IN_PLACE, global,2,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
    count = global[0]; newstdvr = global[1];
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

void MemoryBuffer::updateActiveBuffer()
{
  //If observed sequences exceeded capacity of memory buffer, placed into buffer
  //Here inserted in dataset by removing old/less important seqs (see function)
  //Also, algorithms might want to adap size of memory buffer
  if(Buffered.size() || adapt_TotSeqNum<Set.size()) {
    updateRewardsStats();
    insertBufferedSequences();
  } else if (nTransitions not_eq old_ndata)
    updateRewardsStats();

  old_ndata = nTransitions;
  #ifndef importanceSampling
    const Uint ndata = (bSampleSeq) ? nSequences : nTransitions;
    inds.resize(ndata);
    std::iota(inds.begin(), inds.end(), 0);
    std::random_shuffle(inds.begin(), inds.end(), *(gen));
  #else
    updateImportanceWeights();
  #endif
}

Uint MemoryBuffer::prune(const Real maxFrac, const Real CmaxRho)
{
  //this assumes that sequences with importance weight closer to 1
  // are places at the beginning of the QUEUE
  //we use sortSequences() which, due to legacy reasons, sorts by sequences
  // average MSerror. therefore we placed -max(rho, 1/rho) in MSEfield
  // (therefore samples with rho = 0.2 are treated same as those with rho=5)
  // samples with rho farther from 1 are later in the Set vector
  assert(CmaxRho>1);
  int ret = 0, minInd = std::numeric_limits<int>::max();
  for(int i = (int)Set.size()-1; i >= 0; i--) {
    Real numOver = 0;
    minInd = std::min(minInd, Set[i]->ID);
    for(Uint j=0; j<Set[i]->ndata(); j++)
      if( Set[i]->offPol_weight[j] > CmaxRho ) numOver += 1;

    if( numOver/(Real)Set[i]->ndata() > maxFrac ) {
      std::swap(Set[i], Set.back());
      popBackSequence();
      ret++;
    }
  }
  printf("Minimum sequence id:%u\n",minInd);
  // sequence is removed if more than maxFrac of samples have importance
  // weight either <1/CmaxRho or >CmaxRho
  return ret;
}

void MemoryBuffer::updateImportanceWeights()
{
  const Uint ndata = bSampleSeq ? nSequences : nTransitions;
  inds.resize(ndata);
  std::iota(inds.begin(), inds.end(), 0);
  vector<Real> errors(ndata), Ps(ndata), Ws(ndata);

  for(Uint i=0, k=0; i<Set.size(); i++)
    for(Uint j=0; j<Set[i]->ndata(); j++)
      //if(bSampleSeq) errors[i] = std::max(maxerr, Set[i]->SquaredError[j]);
      if(bSampleSeq) errors[i] += Set[i]->SquaredError[j]/Set[i]->ndata();
      else errors[k++] = Set[i]->SquaredError[j];

  #if 1
    const auto comp=[&](const Uint a,const Uint b){return errors[a]>errors[b];};
    __gnu_parallel::sort(inds.begin(), inds.end(), comp);
    assert(errors[inds.front()] >= errors[inds.back()]);

    //sort in decreasing order of the error. Points with zero error
    //(which means that they are not yet processed) are put at the top:
    #pragma omp parallel for
    for(Uint i=0; i<ndata; i++)
      Ps[inds[i]] = errors[inds[i]]>0 ? std::sqrt(1./(i+1.)) : 1;
  #else
    Ps = errors;
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

void MemoryBuffer::getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
{
  fileOut<<nSequences<<" "<<nTransitions<<" "<<nSeenSequences<<" "
         <<nSeenTransitions<<" "<<adapt_TotSeqNum<<" "<<1./invstd_reward<<" ";
         //<<nSequencesInBuf<<" "<<nSequencesDeleted<<endl;
  screenOut<<" nSeq:"<<nSequences<<" nObs:"<<nTransitions
  <<" (seen Seq:"<<nSeenSequences<<" Obs:"<<nSeenTransitions
  <<") maxSeq:"<<adapt_TotSeqNum<<" stdRew:"<<1./invstd_reward;
  //<<nSequencesInBuf<<" "<<nSequencesDeleted<<endl;
}

void MemoryBuffer::restart()
{
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
      if(sampID!=_agents[0]->transitionID && info!=1) die(" transitionID");
      for(Uint i=0; i<sI.dim; i++) state[i]  = buf[k++];
      for(Uint i=0; i<aI.dim; i++) action[i] = buf[k++];
      Real reward = buf[k++];
      for(Uint i=0; i<policyVecDim; i++) policy[i] = buf[k++];
      assert(k == writesize);
      _agents[0]->update(info, state, reward);
      _agents[0]->act(action);
      //add(0, *(_agents[0]), policy);
    }
    fclose(pFile); free(buf);
    agentID++;
  }
  if(agentID==0) { printf("Couldn't restart transition data.\n"); } //return 1;
  //push_back(0);
  printf("Found %d broken seq out of %d/%d.\n",nBroken,nSequences,nTransitions);
  //return 0;
}

Uint MemoryBuffer::sampleTransitions(vector<Uint>& seq, vector<Uint>& trans)
{
  const Uint batch_size = seq.size(); assert(trans.size() == batch_size);
  vector<Uint> load(batch_size), sort(batch_size), s(batch_size), t(batch_size);
  for (Uint i=0; i<batch_size; i++) {
    const int ind = sample();
    if(ind<0) die("not enough data");
    indexToSample(ind, s[i], t[i]);
    sort[i] = i;
    //work per transition (applies to algos with off policy corrections):
    load[i] = Set[s[i]]->ndata() - t[i];
    //load[i] = data->Set[k]->tuples.size()-1; // ~ this would be for RNN
  }

  //sort elements of sorting according to load for each transition:
  const auto compare = [&] (Uint a, Uint b) { return load[a] < load[b]; };
  std::sort(sort.begin(), sort.end(), compare);
  assert(load[sort[0]] <= load[sort[batch_size-1]]);
  //sort vectors passed to learning algo:
  for (Uint i=0; i<batch_size; i++) {
    trans[i] = t[sort[i]];
    seq[i] = s[sort[i]];
  }
  return batch_size; //always add one grad per transition
}

int MemoryBuffer::sample(const int thrID)
{
  #ifndef importanceSampling
    if(inds.size() == 0) return -1;
    const Uint ind = inds.back();
    inds.pop_back();
  #else
    const Uint ind = (*dist)(generators[thrID]);
  #endif
  return ind;
}

Uint MemoryBuffer::sampleSequences(vector<Uint>& seq)
{
  Uint batch_size = seq.size(), _nAddedGradients = 0;
  for (Uint i=0; i<batch_size; i++) {
    const int ind = sample();
    if(ind<0) die("not enough data");
    seq[i]  = ind;
    _nAddedGradients += Set[ind]->ndata();
  }
  //sort them such that longer ones are started first, reducing overhead
  const auto compare = [this] (Uint a, Uint b) {
    return Set[a]->tuples.size() < Set[b]->tuples.size();
  };
  std::sort(seq.begin(), seq.end(), compare);
  assert( Set[seq.front()]->ndata() <= Set[seq.back()]->ndata() );
  return _nAddedGradients;
}
