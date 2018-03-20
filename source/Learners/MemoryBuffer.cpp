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
void MemoryBuffer::add_state(const Agent&a)
{
  if(a.Status < TERM_COMM) {
    #pragma omp atomic
    nSeenTransitions ++;
  }

  #if 1
    if (inProgress[a.ID]->tuples.size() && a.Status == INIT_COMM) {
      //prev sequence not empty, yet received an initial state, push back prev
      warn("Unexpected termination of sequence");
      push_back(a.ID);
    } else if(inProgress[a.ID]->tuples.size()==0) {
      if(a.Status not_eq INIT_COMM) die("Missing initial state");
    }
  #endif

  #ifndef NDEBUG // check that last new state and new old state are the same
    if(inProgress[a.ID]->tuples.size()) {
      bool same = true;
      const Rvec vecSold = a.sOld->copy_observed();
      const auto memSold = inProgress[a.ID]->tuples.back()->s;
      for (Uint i=0; i<sI.dimUsed && same; i++) //scaled vec only has used dims:
        same = same && std::fabs(memSold[i]-vecSold[i]) < 1e-4;
      if (!same) { //create new sequence
        warn("Unexpected termination of sequence");
        push_back(a.ID);
      }
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  env->pickReward(a);
  inProgress[a.ID]->ended = a.Status==TERM_COMM;
  inProgress[a.ID]->add_state(a.s->copy_observed(), a.r);
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
  assert(a.Status>=TERM_COMM);
  assert(inProgress[a.ID]->tuples.back()->mu.size() == 0);
  assert(inProgress[a.ID]->tuples.back()->a.size()  == 0);
  a.a->set(Rvec(aI.dim,0));
  add_action(a, Rvec(policyVecDim, 0) );
  push_back(a.ID);
}

// update the second order moment of the rewards in the memory buffer
void MemoryBuffer::updateRewardsStats(unsigned long nStep)
{
  _nStep = nStep;
  if(!bTrain) return; //if not training, keep the stored values

  long double count = 0, newstdvr = 0;
  #pragma omp parallel reduction(+ : count, newstdvr)
  {
    #pragma omp for schedule(dynamic)
    for(Uint i=0; i<Set.size(); i++) {
      count += Set[i]->ndata();
      for(Uint j=0; j<Set[i]->ndata(); j++)
        newstdvr += std::pow(Set[i]->tuples[j+1]->r, 2);
    }
  }

  //add up gradients across nodes (masters)
  if (learn_size > 1) {
    const bool firstUpdate = rewRequest == MPI_REQUEST_NULL;
    if(not firstUpdate) MPI_Wait(&rewRequest, MPI_STATUS_IGNORE);
    else {
      rew_reduce_result.resize(nReduce);
      partial_sum.resize(nReduce);
    }

    // prepare an allreduce with the current data:
    partial_sum[0] = count;
    partial_sum[1] = newstdvr;
    //use result from prev reduce to update rewards (before calling new reduce)
    count = rew_reduce_result[0];
    newstdvr = rew_reduce_result[1];

    MPI_Iallreduce(partial_sum.data(), rew_reduce_result.data(), nReduce,
                    MPI_LONG_DOUBLE, MPI_SUM, mastersComm, &rewRequest);
    // if no reduction done, partial sums are meaningless
    if(firstUpdate) return;
  }

  if(count<batchSize) return;
  //const Real stdev_reward=std::sqrt((newstdvr-newmeanr*newmeanr/count)/count);
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
  if(inProgress[agentId]->tuples.size() > 2 ) //at least s0 and sT
  {
    inProgress[agentId]->finalize( readNSeenSeq() );

    #pragma omp atomic
    nSeenSequences++;

    #pragma omp atomic
    nCmplTransitions += inProgress[agentId]->ndata();

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
  assert(CmaxRho>1);
  // vector indicating location of sequence to delete
  vector<pair<int, Real>> farpol_location(nThreads, {-1, -1});
  vector<pair<int, Real>> maxerr_location(nThreads, {-1, -1});
  vector<pair<int, Real>> minerr_location(nThreads, {-1,9e9});
  // vector indicating location of oldest sequence
  vector<pair<int,  int>> oldest_sequence(nThreads, {-1, nSeenSequences});
  Real _nOffPol = 0, _totMSE = 0, invC = 1/CmaxRho;
  #pragma omp parallel reduction(+ : _nOffPol, _totMSE)
  {
    const int thrID = omp_get_thread_num();
    #pragma omp for schedule(dynamic)
    for(Uint i = 0; i < Set.size(); i++)
    {
      if(Set[i]->just_sampled >= 0) {
        Set[i]->nOffPol = 0; Set[i]->MSE = 0;
        for(Uint j=0; j<Set[i]->ndata(); j++) {
          Set[i]->MSE += Set[i]->SquaredError[j];
          const Real W = Set[i]->offPol_weight[j];
          assert(Set[i]->SquaredError[j]>=0 && W>=0);
          // sequence is off policy if offPol W is out of 1/C : C
          if(W>CmaxRho || W<invC) Set[i]->nOffPol += 1;
        }
        Set[i]->just_sampled = -1;
      }
      
      const Real W_MSE = Set[i]->MSE    /Set[i]->ndata();
      const Real W_FAR = Set[i]->nOffPol/Set[i]->ndata();
      _nOffPol += Set[i]->nOffPol; _totMSE += Set[i]->MSE;

      // TODO: to avoid overfitting only keep "unexpected" transition in buffer
      if(Set[i]->ID < oldest_sequence[thrID].second) {
        oldest_sequence[thrID].second = Set[i]->ID;
        oldest_sequence[thrID].first = i;
      }
      if(W_FAR > farpol_location[thrID].second) {
        farpol_location[thrID].second = W_FAR;
        farpol_location[thrID].first = i;
      }
      if(W_MSE > maxerr_location[thrID].second) {
        maxerr_location[thrID].second = W_MSE;
        maxerr_location[thrID].first = i;
      }
      if(W_MSE < minerr_location[thrID].second) {
        minerr_location[thrID].second = W_MSE;
        minerr_location[thrID].first = i;
      }
    }
  }

  nOffPol = _nOffPol; totMSE = _totMSE/nTransitions;
  const int nB4 = Set.size();
  int old_ptr = -1, far_ptr = -1, mse_ptr = -1, fit_ptr = -1, del_ptr = -1;
  Real mse_val = -1, far_val = -1, fit_val = 9e9, old_ind = nSeenSequences;

  for(const auto&P: oldest_sequence)
   if(P.second<old_ind && P.first>=0) { old_ptr = P.first; old_ind = P.second; }

  for(const auto&P: farpol_location)
   if(P.second>far_val && P.first>=0) { far_ptr = P.first; far_val = P.second; }

  for(const auto&P: maxerr_location)
   if(P.second>mse_val && P.first>=0) { mse_ptr = P.first; mse_val = P.second; }

  for(const auto&P: minerr_location)
   if(P.second<fit_val && P.first>=0) { fit_ptr = P.first; fit_val = P.second; }

  minInd = old_ind;
  assert( old_ptr < nB4 && far_ptr < nB4 && mse_ptr < nB4 && fit_ptr < nB4 );
  switch(ALGO) {
      case OLDEST:     del_ptr = old_ptr; break;
      case FARPOLFRAC: del_ptr = far_ptr; break;
      case MAXERROR:   del_ptr = mse_ptr; break;
      case MINERROR:   del_ptr = fit_ptr; break;
  }
  // safety measure to avoid trajectories lingering too much in mem buffer
  if(Set[old_ptr]->ID + (int)Set.size() < Set[del_ptr]->ID) del_ptr = old_ptr;
  // safety measure: do not delete trajectory if Nobs > Ntarget
  // but if N > Ntarget even if we remove the trajectory
  // done to avoid problems if a sequence is longer than maxTotObsNum
  // negligible effect if hyperparameters are chosen wisely
  if(nTransitions-Set[del_ptr]->ndata() > maxTotObsNum) {
    std::swap(Set[del_ptr], Set.back());
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
  buff<<" "<<std::setw(5)<<nSequences;
  buff<<" "<<std::setw(6)<<nTransitions;
  buff<<" "<<std::setw(7)<<nSeenSequences;
  buff<<" "<<std::setw(8)<<nSeenTransitions;
  buff<<" "<<std::setw(7)<<minInd;
  buff<<" "<<std::setw(6)<<(int)nOffPol;
  buff<<" "<<std::setw(6)<<std::setprecision(2)<<1./invstd_reward;
  buff<<" "<<std::setw(6)<<std::setprecision(3)<<totMSE;
  nPruned=0;
}
void MemoryBuffer::getHeaders(ostringstream& buff)
{
  buff <<
  "| nEp | nObs | totEp | totObs | oldEp |nOffP | stdR | tMSE ";
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

void MemoryBuffer::sampleSequence(Uint& seq, const int thrID)
{
  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<int> distSeq(0, readNSeq()-1);
    seq = distSeq(generators[thrID]);
  #else
    seq = (*dist)(generators[thrID]);
  #endif
}

void MemoryBuffer::sampleTransitions_OPW(vector<Uint>&seq, vector<Uint>&obs)
{
  assert(seq.size() == obs.size());
  vector<Uint> s = seq, o = obs;
  int nThr_loc = nThreads;
  while(seq.size() % nThr_loc) nThr_loc--;
  const int stride = seq.size()/nThr_loc, N = seq.size();
  assert(nThr_loc>0 && N % nThr_loc == 0 && stride > 0);

  // Perf tweak: sort by offPol weight to aid load balance of algos like
  // Racer/PPO. "Far Policy" obs are discarded due to opcW being out of range
  // and will trigger a resampling. Sorting here affects tasking order.
  const auto isAbeforeB = [&] (const pair<Uint,Real> a, const pair<Uint,Real> b)
  { return a.second > b.second; };

  vector<pair<Uint, Real>> load(N);
  #pragma omp parallel num_threads(nThr_loc)
  {
    const int thrI = omp_get_thread_num(), start = thrI*stride;
    assert(nThr_loc == omp_get_num_threads());

    sampleMultipleTrans(&s[start], &o[start], stride, thrI);
    for(int i=start; i<start+stride; i++) {
      const Real W = Set[s[i]]->offPol_weight[o[i]], invW = 1/W;
      load[i].first = i; load[i].second = std::max(W, invW);
    }
    std::sort(load.begin()+start, load.begin()+start+stride, isAbeforeB);

    // additional parallel partial sort if batchsize makes it worthwhile
    if(stride % nThr_loc == 0) {
      vector<pair<Uint, Real>> load_loc(stride);
      const int preSortChunk = stride / nThr_loc;

      #pragma omp barrier

      for(int i=0, k=0; i<nThr_loc; i++) // avoid cache thrashing
       for(int j=0; j<preSortChunk; j++)
         load_loc[k++] = load[thrI*preSortChunk + i*stride + j];

      std::sort(load_loc.begin(), load_loc.end(), isAbeforeB);

      #pragma omp barrier
      for(int i=0; i<stride; i++) load[i+start] = load_loc[i];
    }
  }
  //for (Uint i=0; i<seq.size(); i++) cout<<load[i].second<<endl; cout<<endl;
  std::sort(load.begin(), load.end(), isAbeforeB);
  for (Uint i=0; i<seq.size(); i++) {
    obs[i] = o[load[i].first];
    seq[i] = s[load[i].first];
  }

}

void MemoryBuffer::sampleSequences(vector<Uint>& seq)
{
  const Uint N = seq.size();
  if( readNSeq() > N*5 ) {
    for(Uint i=0; i<N; i++) sampleSequence(seq[i], 0);
  } else { // if N is large, make sure we do not repeat indices
    seq.resize(readNSeq());
    std::iota(seq.begin(), seq.end(), 0);
    std::shuffle(seq.begin(), seq.end(), generators[0]);
    seq.resize(N);
  }
  const auto compare = [&](Uint a, Uint b) {
    return Set[a]->ndata() > Set[b]->ndata();
  };
  std::sort(seq.begin(), seq.end(), compare);
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

void MemoryBuffer::sampleMultipleTrans(Uint* seq, Uint* obs, const Uint N, const int thrID)
{
  vector<Uint> samples(N);

  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<Uint> distObs(0, readNData()-1);
    for (Uint k=0; k<N; k++) samples[k] = distObs(generators[thrID]);
  #else
    for (Uint k=0; k<N; k++) samples[k] = (*dist)(generators[thrID]);
  #endif

  std::sort(samples.begin(), samples.end());

  Uint cntO = 0, samp = 0;
  for (Uint k=0; k<Set.size(); k++) {
    for(Uint i=samp; i<N; i++) {
      if(samples[i] < cntO+Set[k]->ndata()) { // cannot sample last state in seq
        seq[i] = k;
        obs[i] = samples[i]-cntO;
        samp = i+1; // next iteration remember first samp-1 were already found
      }
    }
    if(samp == N) break;
    cntO += Set[k]->ndata();
    if(k+1 == Set.size()) assert(cntO == nTransitions && samp == N);
  }
}
