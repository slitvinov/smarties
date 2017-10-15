/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 */

#include "Transitions.h"
#include <dirent.h>
#include <iterator>
#include <parallel/algorithm>

Transitions::Transitions(MPI_Comm comm, Environment* const _env, Settings & _s):
  mastersComm(comm), env(_env), bNormalize(_s.bNormalize), bTrain(_s.bTrain),
  bWriteToFile(!(_s.samplesFile=="none")), bSampleSeq(_s.bSampleSequences),
  maxTotSeqNum(_s.maxTotSeqNum),maxSeqLen(_s.maxSeqLen),minSeqLen(_s.minSeqLen),
  nAppended(_s.appendedObs),batchSize(_s.batchSize),learn_rank(_s.learner_rank),
  learn_size(_s.learner_size), path(_s.samplesFile), gamma(_s.gamma),
  generators(_s.generators), sI(_env->sI), aI(_env->aI), _agents(_env->agents)
{
  mean.resize(sI.dimUsed, 0);
  std.resize(sI.dimUsed, 1);
  invstd.resize(sI.dimUsed, 1);

  if (sI.mean.size()){
    Uint k = 0;
    for (Uint i=0; i<sI.dim; i++){
      if (sI.inUse[i]) {
        mean[k] = sI.mean[i];
        std[k] = sI.scale[i];
        invstd[k] = 1./sI.scale[i];
        k++;
      }
    }
    assert(k == sI.dimUsed);
  }
  assert(_s.nAgents>0);
  Tmp.resize(_s.nAgents);
  for (Uint i=0; i<static_cast<Uint>(_s.nAgents); i++)
    Tmp[i] = new Sequence();

  gen = new Gen(&generators[0]);
  Set.reserve(maxTotSeqNum);
}

Uint Transitions::restartSamples(const Uint polDim)
{
  if(path == "none") return false;
  const Uint writesize = 3+sI.dim+aI.dim+polDim;
  int agentID = 0, info = 0, sampID = 0;
  vector<Real> policy(polDim), action(aI.dim);
  vector<double> state(sI.dim);
  Real reward = 0;
  char asciipath[256];

  while (true)
  {
    sprintf(asciipath, "obs_rank%02d_agent%03d.raw", learn_rank, agentID);
    FILE*pFile = fopen(asciipath, "rb");
    if(pFile==NULL){ printf("Couldnt open file %s.\n",asciipath); break; }

    float* buf = (float*) malloc(writesize*sizeof(float));

    while(true) {
      size_t ret = fread(buf, sizeof(float), writesize, pFile);
      if (ret == 0) break;
      if (ret != writesize) _die("Error reading datafile %s", asciipath);
      Uint k = 0; info = buf[k++]; sampID = buf[k++];

      if((sampID==0) != (info==1)) die("Mismatch in transition counter\n");
      if(sampID!=_agents[0]->transitionID && info!=1) die(" transitionID");

      for(Uint i=0; i<sI.dim; i++) state[i]  = buf[k++];
      for(Uint i=0; i<aI.dim; i++) action[i] = buf[k++];
      reward = buf[k++];
      for(Uint i=0; i<polDim; i++) policy[i] = buf[k++];
      assert(k == writesize);
      _agents[0]->update(info, state, reward);
      _agents[0]->a->vals = action;
      add(0, *(_agents[0]), policy);
    }
    fclose(pFile);
    free(buf);
    agentID++;
  }
  if(agentID==0)  {
    printf("Couldn't restart transition data.\n");
    return 1;
  }
  push_back(0);
  printf("Found %d broken chains out of %d / %d.\n",
      nBroken, nSequences, nTransitions);
  return 0;
}

void Transitions::writeData(const int agentId, const Agent&a, const vector<Real>mu)
{
  char asciipath[256];
  sprintf(asciipath, "obs_rank%02d_agent%03d.raw", learn_rank, agentId);
  FILE * pFile = fopen (asciipath, "ab");
  const Uint writesize = (3 + sI.dim + aI.dim + mu.size())*sizeof(float);
  float* buf = (float*) malloc(writesize);
  memset(buf, 0, writesize);
  Uint k=0;
  buf[k++] = a.Status + 0.1;
  buf[k++] = a.transitionID + 0.1;
  for (Uint i=0; i<sI.dim;    i++) buf[k++] = (float) a.s->vals[i];
  for (Uint i=0; i<aI.dim;    i++) buf[k++] = (float) a.a->vals[i];
  buf[k++] = a.r;
  for (Uint i=0; i<mu.size(); i++) buf[k++] = (float) mu[i];
  assert(k*sizeof(float) == writesize);
  fwrite (buf, sizeof(float), writesize/sizeof(float), pFile);
  fflush(pFile); fclose(pFile);  free(buf);
}

int Transitions::passData(const int agentId,const Agent&a,const vector<Real>mu)
{
  const int ret = add(agentId, a, mu);

  if(bWriteToFile || !agentId) writeData(agentId, a, mu);

  return ret;
}

int Transitions::add(const int agentId, const Agent&a, const vector<Real>mu)
{
  //return value is 1 if the agent states buffer is empty or on initial state
  int ret = 0;

  const Uint sApp = nAppended*sI.dimUsed;
  if (Tmp[agentId]->tuples.size() && a.Status == 1) {
    //previous sequence not empty, yet received an initial state, push back prev
    warn("Detected partial sequence\n");
    push_back(agentId);
    ret = 1;
  } else if(Tmp[agentId]->tuples.size()==0) {
    if(a.Status not_eq 1) die("Missing initial state\n");
    ret = 1; //status is 1
  }
  /*
    if(Tmp[agentId]->tuples.size()!=0) {
      const Tuple*const last = Tmp[agentId]->tuples.back();
      printf("Continue %d[%s]=[%s][%s][%s],%g\n",agentId,sOld._print().c_str(),
      print(last->s).c_str(),aNew._print().c_str(),sNew._print().c_str(),rNew);
    }
    else
      printf("Start chain %d[%s][%s][%s],%g\n",agentId, sOld._print().c_str(),
      aNew._print().c_str(),sNew._print().c_str(),rNew);
   */
  if(Tmp[agentId]->tuples.size()) {
    // check that last new state and new old state are the same
    bool same = true;
    const vector<Real> vecSold = a.sOld->copy_observed();
    const Tuple*const last = Tmp[agentId]->tuples.back();
    for (Uint i=0; i<sI.dimUsed && same; i++) //scaled vec only has used dims:
      same = same && std::fabs(last->s[i]-vecSold[i])<1e-8;
    if (!same) {
      warn("Detected partial sequence");
      push_back(agentId); //create new sequence
      ret = 1;
    }
  }

  if (Tmp[agentId]->tuples.size() >= maxSeqLen) {
    //upper limit to how long a sequence can be
    //printf("Sequence is too long!\n");
    const Tuple* const l = Tmp[agentId]->tuples.back();
    Tuple * t = new Tuple(); //backup last state
    t->s = l->s; t->a = l->a; t->r = l->r, t->mu = l->mu;
    t->SquaredError = l->SquaredError;
    #ifdef importanceSampling
      t->weight = l->weight;
    #endif
    push_back(agentId); //create new sequence
    Tmp[agentId]->tuples.push_back(t);
  }

  //we can add sNew:
  Tuple * t = new Tuple();
  t->s = a.s->copy_observed();
  if (sApp>0) {
    if(Tmp[agentId]->tuples.size()==0)
      t->s.insert(t->s.end(), sApp, 0);
    else {
      const Tuple * const last = Tmp[agentId]->tuples.back();
      t->s.insert(t->s.end(), last->s.begin(), last->s.begin() +sApp);
      assert(last->s.size()==t->s.size());
    }
  }

  const bool end_seq = env->pickReward(a);
  assert((a.Status==2)==end_seq); //alternative not supported
  t->a = a.a->vals;
  t->r = a.r;
  t->mu = mu;

  Tmp[agentId]->tuples.push_back(t);
  if (end_seq) {
    Tmp[agentId]->ended = true;
    push_back(agentId);
  }

  return ret;
}

void Transitions::clearFailedSim(const int agentOne, const int agentEnd)
{
  for (int i = agentOne; i<agentEnd; i++) {
    _dispose_object(Tmp[i]);
    Tmp[i] = new Sequence();
  }
}

void Transitions::pushBackEndedSim(const int agentOne, const int agentEnd)
{
  for (int i = agentOne; i<agentEnd; i++)
    if(Tmp[i]->tuples.size()) push_back(i);
}

void Transitions::push_back(const int & agentId)
{
  if(Tmp[agentId]->tuples.size() > minSeqLen )
  {
    Tmp[agentId]->ID = nSeenSequences++;
    nSeenTransitions += Tmp[agentId]->ndata();
    assert(nSequences == Set.size());

    lock_guard<mutex> lock(dataset_mutex);
    if (nSequences >= adapt_TotSeqNum)
      Buffered.push_back(Tmp[agentId]);
    else
      pushBackSequence(Tmp[agentId]);
  } else {
    printf("Trashing %lu obs.\n",Tmp[agentId]->tuples.size());
    fflush(0);
    _dispose_object(Tmp[agentId]);
  }

  Tmp[agentId] = new Sequence();
}

/*
void Transitions::update_samples_mean(const Real alpha)
{
  if(!bTrain || !bNormalize) return; //if not training, keep the stored values

  long double count = 0;
  vector<long double> newStd(sI.dimUsed,0), newMean(sI.dimUsed,0);

  #pragma omp parallel
  {
    //local sum and counter
    vector<long double> sum(sI.dimUsed,0), sum2(sI.dimUsed,0);
    Uint cnt = 0;

    #pragma omp for schedule(dynamic)
    for(Uint i=0; i<Set.size(); i++)
      for(const auto & t : Set[i]->tuples) {
        assert(t->s.size() == sI.dimUsed*(1+nAppended));
        cnt++;
        for (Uint j=0; j<sI.dimUsed; j++) {
          sum2[j] += t->s[j]*t->s[j];
          sum[j]  += t->s[j];
        }
      }

    #pragma omp critical
    {
      count += cnt;
      for (Uint i=0; i<sI.dimUsed; i++) {
        newMean[i] += sum[i];
        newStd[i] += sum2[i];
      }
    }
  }

  //add up gradients across nodes (masters)
  if (learn_size > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &count, 1,
        MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, newMean.data(), sI.dimUsed,
        MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, newStd.data(), sI.dimUsed,
        MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
  }

  if(count<batchSize) return;
  for (Uint i=0; i<sI.dimUsed; i++) {
    newStd[i] = std::sqrt((newStd[i]-newMean[i]*newMean[i]/count)/count);
    newStd[i] = std::max(newStd[i],(long double)1e-8);
    newMean[i] /= count;
  }

  if (sI.mean.size()) {
    Uint k=0;
    for (Uint i=0; i<sI.dim; i++)
      if (sI.inUse[i]) {
        mean[k] = sI.mean[i]*(1-alpha) + alpha*newMean[k];
        std[k] = sI.scale[i]*(1-alpha) + alpha*newStd[k];
        invstd[k] = 1./(std[i]+1e-8);
        k++;
      }
    assert(k==sI.dimUsed);
  } else {
    for (Uint i=0; i<sI.dimUsed; i++) {
      mean[i] = mean[i]*(1.-alpha) + alpha*newMean[i];
      std[i] = std[i]*(1.-alpha) + alpha*newStd[i];
      invstd[i] = 1./(std[i]+1e-8);
    }
  }
}
*/

void Transitions::update_rewards_mean()
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

Uint Transitions::prune(const Real maxFrac, const Real CmaxRho)
{
  //this assumes that sequences with importance weight closer to 1
  // are places at the beginning of the QUEUE
  //we use sortSequences() which, due to legacy reasons, sorts by sequences
  // average MSerror. therefore we placed -max(rho, 1/rho) in MSEfield
  // (therefore samples with rho = 0.2 are treated same as those with rho=5)
  // samples with rho farther from 1 are later in the Set vector
  assert(CmaxRho>1);
  Uint ret = 0;
  for(int i = (int)Set.size()-1; i >= 0; i--) {
    Real numOver = 0;
    for(Uint j=0; j<Set[i]->ndata(); j++)
      if( Set[i]->tuples[j]->SquaredError > CmaxRho ) numOver += 1;

    if( numOver/(Real)Set[i]->ndata() > maxFrac ) {
      std::swap(Set[i], Set.back());
      popBackSequence();
      ret++;
    }
  }
  // sequence is removed if more than maxFrac of samples have importance
  // weight either <1/CmaxRho or >CmaxRho

  #ifndef NDEBUG
  Uint cntSamp = 0;
  for(Uint i=0; i<Set.size(); i++) {
    assert(Set[i] not_eq nullptr);
    cntSamp += Set[i]->ndata();
  }
  assert(cntSamp==nTransitions);
  #endif

  return ret;
}

void Transitions::sortSequences()
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
        Set[i]->MSE += Set[i]->tuples[j]->SquaredError;
      Set[i]->MSE /= Set[i]->ndata();
     #endif
  }
  //printf("%lu %u %u %lu %lu\n", Set.size(), adapt_TotSeqNum, maxTotSeqNum, Set.capacity(), Set.end()-Set.begin()); fflush(0);

  struct Compare
  {
    vector<Sequence*> Set;
    Compare(const vector<Sequence*>& S) : Set(S) {}
    bool operator()(const Sequence*const a, const Sequence*const b) const {
      //printf("a:%lu b:%lu\n", a-Set.begin(), b-Set.begin()); fflush(0);
      assert(a not_eq nullptr);
      assert(b not_eq nullptr);
      return a->MSE==0 ? true : (b->MSE==0 ? false : (a->MSE>b->MSE) );
    }
  };
  Compare comparer(Set);
  assert(Set.size()>0 && Set.size() == nSequences);
  //__gnu_parallel::
  std::sort(Set.begin(), Set.begin()+nSequences, comparer);
  assert(Set.front()->MSE > Set.back()->MSE || Set.front()->MSE == 0);

  #ifndef NDEBUG
  for(Uint i=0; i<Set.size(); i++) {
    //printf("%u %f %u\n",i,Set[i]->MSE, Set[i]->ndata());
    assert(*(Set.begin()+i) not_eq nullptr);
  }
  #endif
}

void Transitions::synchronize()
{
  // 4 steps:
  //1) if Set.size() < adapt_TotSeqNum, add buffer to Set
  //2) optional, sort transitions in Set
  //3) if Set.size() > adapt_TotSeqNum remove those at the back
  //this implies that adaptive Set size is only suppoted if i sort the sequences
  //4) add the buffered transitions
  while( (Set.size()<adapt_TotSeqNum) && Buffered.size()>0 )
  {
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

  if(Set.size() > adapt_TotSeqNum)
  {
    while(Set.size() > adapt_TotSeqNum) popBackSequence();

    Set.reserve(maxTotSeqNum);
    assert(nSequences<=adapt_TotSeqNum);
    iOldestSaved = nSequences - Buffered.size();
  }

  #ifndef NDEBUG
    Uint cntSamp = 0;
    for(Uint i=0; i<Set.size(); i++) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions);
  #endif

  if( Buffered.size() == 0 ) return;

  Uint nTransitionsInBuf=0, nTransitionsDeleted=0;
  for(Uint j=Buffered.size(); j>0; j--)
  {
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

  const string fname = "transitions.log";
  FILE * f = fopen(fname.c_str(), "a");
  if (f == NULL) die("Save fail\n");

  fprintf(f,"Removing %lu sequences (avg length %f) associated with small MSE"
      "error in favor of new ones (avg lendth %f).\n",
      Buffered.size(), nTransitionsDeleted/(Real)Buffered.size(),
      nTransitionsInBuf/(Real)Buffered.size());
  fflush(f); fclose(f);
}

Uint Transitions::updateSamples(const Real annealFac)
{
  bool update_meanstd_needed = false;
  const string fname = "transitions.log";
  FILE * f = fopen(fname.c_str(), "a");
  if (f == NULL) die("Save fail\n");

  printCount++;
  // when do I need to sort and refresh dataset vector?
  // 1- if i have buffered sequences
  // 2- if i have to remove some of the samples
  if(Buffered.size()>0 || adapt_TotSeqNum < Set.size())
  {
    if(!learn_rank) // && printCount%10 == 0
    fprintf(f,"nSeq %d (%lu)  >maxTotSeqNum %d (nObs=%d/%lu, avgSeqLen=%f).\n",
      nSequences, nSeenSequences, adapt_TotSeqNum, nTransitions,
      nSeenTransitions, nTransitions/(Real)nSequences);
    synchronize();
    update_meanstd_needed = true;
    old_ndata = nTransitions;
  }
  else
  {
    if(!learn_rank) // && printCount%100 == 0
    fprintf(f,"nSeq %d (%lu) =<maxTotSeqNum %d (nObs=%d/%lu, avgSeqLen=%f).\n",
      nSequences, nSeenSequences, adapt_TotSeqNum, nTransitions,
      nSeenTransitions, nTransitions/(Real)nSequences);
    update_meanstd_needed = nTransitions not_eq old_ndata;
    old_ndata = nTransitions;
  }

  if(update_meanstd_needed) update_rewards_mean();
  update_meanstd_needed = update_meanstd_needed && bNormalize && annealFac>0;

  fflush(f); fclose(f);
  #ifndef importanceSampling
    const Uint ndata = (bSampleSeq) ? nSequences : nTransitions;
    inds.resize(ndata);
    std::iota(inds.begin(), inds.end(), 0);
    //__gnu_parallel::
    std::random_shuffle(inds.begin(), inds.end(), *(gen));
  #else
    updateP();
  #endif

  return update_meanstd_needed ? 1 : 0;
}

int Transitions::sample(const int thrID)
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

void Transitions::save(std::string fname)
{
  if(learn_rank) return;
  string nameBackup = fname + "_data_stats";
  FILE * f = fopen(nameBackup.c_str(), "w");
  if (f != NULL) {
    for (Uint i=0; i<sI.dimUsed; i++)
      fprintf(f, "%9.9e %9.9e\n", mean[i], std[i]);
    fprintf(f, "%9.9e %9.9e\n", invstd_reward, mean_reward);
  }

  fclose(f);
}

void Transitions::restart(std::string fname)
{
  string nameBackup = fname + "_data_stats";
  ifstream in(nameBackup.c_str());
  debugT("Reading from %s\n", nameBackup.c_str());
  if (!in.good()) {
    debugT("File not found %s\n", nameBackup.c_str());
    #ifndef NDEBUG //if debug, you might want to do this
      if(!bTrain) {die("...and I'm not training\n");}
    #endif
    return;
  }

  for (Uint i=0; i<sI.dimUsed; i++) {
    in >> mean[i] >> std[i];
    printf("Read: %9.9e %9.9e\n", mean[i], std[i]);
  }
  in >> invstd_reward >> mean_reward;
  printf("Read: %9.9e %9.9e\n", invstd_reward, mean_reward);
  in.close();
}

#ifdef importanceSampling
//Sample sequences: same procedure with importance weights computed from maximun error?
void Transitions::updateP()
{
  const Uint ndata = bSampleSeq ? nSequences : nTransitions;
  inds.resize(ndata);
  std::iota(inds.begin(), inds.end(), 0);
  vector<Real> errors(ndata), Ps(ndata), Ws(ndata);

  {
    Uint k = 0;
    for(Uint i=0; i<Set.size(); i++)
    {
      Real maxerr = 0;
      for(Uint j=0; j<Set[i]->ndata(); j++)
      {
        if(bSampleSeq) //sample based on max error of the sequence
          maxerr = std::max(maxerr,Set[i]->tuples[j]->SquaredError);
        else //sample based on transition's last error
          errors[k++] = Set[i]->tuples[j]->SquaredError;
      }
      if(bSampleSeq) errors[k++] = maxerr;
    }
    assert(k==ndata);
  }

  const auto comp=[&](const Uint a,const Uint b) {return errors[a]>errors[b];};
  __gnu_parallel::sort(inds.begin(), inds.end(), comp);
  assert(errors[inds.front()] >= errors[inds.back()]);

  //sort in decreasing order of the error. Points with zero error
  //(which means that they are not yet processed)
  //are put at the top:
  #pragma omp parallel for
  for(Uint i=0; i<ndata; i++)
    Ps[inds[i]] = errors[inds[i]]>0 ? std::sqrt(1./(i+1.)) : 1;

  //const Real minP = Ps[inds.back()];
  //const Real sumP = __gnu_parallel::accumulate(Ps.begin(), Ps.end(), 0);
  Real minP = 2, sumP = 0;
  #pragma omp parallel for reduction(min: minP) reduction(+: sumP)
  for(Uint i=0; i<ndata; i++) {
    minP = std::min(minP, Ps[i]);
    sumP += Ps[i];
  }
  assert(minP<=1 && sumP>0);

  #pragma omp parallel for
  for(Uint i=0; i<ndata; i++) {
    Ws[i] = minP/Ps[i];
    Ps[i] = Ps[i]/sumP;
  }

  if(dist not_eq nullptr) delete dist;
  dist = new std::discrete_distribution<Uint>(Ps.begin(), Ps.end());

  {
    Uint k = 0;
    for(Uint i=0; i<Set.size(); i++)
    {
      for(Uint j=0; j<Set[i]->ndata(); j++)
      {
        if(bSampleSeq) //sample based on max error of the sequence
          Set[i]->tuples[j]->weight = Ws[k];
        else //sample based on transition's last error
          Set[i]->tuples[j]->weight = Ws[k++];
      }
      if(bSampleSeq) k++;
    }
    assert(k==ndata);
  }
}
#endif
