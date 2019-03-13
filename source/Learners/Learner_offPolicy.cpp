//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_offPolicy.h"
#include "../Network/Optimizer.h"

Learner_offPolicy::Learner_offPolicy(Environment*const E, Settings & S) :
Learner(E, S)
{
  if(not bSampleSequences && nObsB4StartTraining < batchSize)
    die("Parameter minTotObsNum is too low for given problem");
}

bool Learner_offPolicy::blockDataAcquisition() const
{
  //if there is not enough data for training, need more data
  if( data->readNData() < nObsB4StartTraining ) return false;

  // block data if we have observed too many observations
  // here we add one to concurrently gather data and compute gradients
  // 'freeze if there is too much data for the  next gradient step'
  return nLocTimeStepsTrain() > (nGradSteps()+1) * obsPerStep_loc;
}

bool Learner_offPolicy::unblockGradientUpdates() const
{
  // almost the same of the function before
  if( data->readNData() < nObsB4StartTraining ) return false;
  // 'freeze if there is too little data for the current gradient step'
  return nLocTimeStepsTrain() >= nGradSteps() * obsPerStep_loc;
}

void Learner_offPolicy::spawnTrainTasks()
{
  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  vector<Uint> samp_seq = vector<Uint>(batchSize, -1);
  vector<Uint> samp_obs = vector<Uint>(batchSize, -1);
  data->sample(samp_seq, samp_obs);

  for(Uint i=0; i<batchSize && bSampleSequences; i++)
    assert( samp_obs[i] == data->get(samp_seq[i])->ndata() - 1 );

  if(bSampleSequences) {
  #pragma omp parallel for collapse(2) schedule(static) num_threads(nThreads)
    for (Uint wID=0; wID<ESpopSize; wID++)
      for (Uint bID=0; bID<batchSize; bID++) {
        const Uint thrID = omp_get_thread_num();
        TrainBySequences(samp_seq[bID], wID, bID, thrID);
        input->gradient(thrID);
      }
  } else {
  static const Uint CS = batchSize / nThreads;
  #pragma omp parallel for collapse(2) schedule(static,CS) num_threads(nThreads)
    for (Uint wID=0; wID<ESpopSize; wID++)
      for (Uint bID=0; bID<batchSize; bID++) {
        const Uint thrID = omp_get_thread_num();
        Train(samp_seq[bID], samp_obs[bID], wID, bID, thrID);
        input->gradient(thrID);
      }
  }
}

void Learner_offPolicy::processMemoryBuffer()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this

  profiler->stop_start("PRNE");
  //shift data / gradient counters to maintain grad stepping to sample
  // collection ratio prescirbed by obsPerStep

  CmaxRet = 1 + annealRate(CmaxPol, currStep, epsAnneal);
  CinvRet = 1 / CmaxRet;
  if(CmaxRet<=1 and CmaxPol>0)
    die("Either run lasted too long or epsAnneal is wrong.");
  data_proc->prune(ERFILTER, CmaxRet);

  // use result from prev AllReduce to update rewards (before new reduce).
  // Assumption is that the number of off Pol trajectories does not change
  // much each step. Especially because here we update the off pol W only
  // if an obs is actually sampled. Therefore at most this fraction
  // is wrong by batchSize / nTransitions ( ~ 0 )
  // In exchange we skip an mpi implicit barrier point.
  ReFER_reduce.update({(long double)data_proc->nFarPol(),
                       (long double)data->readNData()});
  const LDvec nFarGlobal = ReFER_reduce.get();
  const Real fracOffPol = nFarGlobal[0] / nFarGlobal[1];

  if(fracOffPol>ReFtol) beta = (1-1e-4)*beta; // iter converges to 0
  else beta = 1e-4 +(1-1e-4)*beta; //fixed point iter converge to 1
  if(std::fabs(ReFtol-fracOffPol)<0.001) alpha = (1-1e-4)*alpha;
  else alpha = 1e-4 + (1-1e-4)*alpha;
}

void Learner_offPolicy::updateRetraceEstimates()
{
  profiler->stop_start("QRET");
  const std::vector<Uint>& sampled = data->listSampled();
  const Uint setSize = sampled.size();
  #pragma omp parallel for schedule(dynamic, 1)
  for(Uint i = 0; i < setSize; i++) {
    Sequence * const S = data->get(sampled[i]);
    assert(std::fabs(S->Q_RET[S->ndata()]) < 1e-16);
    assert(std::fabs(S->action_adv[S->ndata()]) < 1e-16);
    if( S->isTerminal(S->ndata()) )
      assert(std::fabs(S->state_vals[S->ndata()]) < 1e-16);
    for(int j=S->just_sampled-1; j>0; j--) backPropRetrace(S, j);
  }
}

void Learner_offPolicy::finalizeMemoryProcessing()
{
  const Uint currStep = nGradSteps()+1; //base class will advance this
  profiler->stop_start("FIND");
  data_proc->finalize();

  profiler->stop_start("PRE");
  if(currStep%1000==0) // update state mean/std with net's learning rate
    data_proc->updateRewardsStats(1, 1e-3*(OFFPOL_ADAPT_STSCALE>0));
}

void Learner_offPolicy::initializeLearner()
{
  const Uint currStep = nGradSteps();

  ReFER_reduce.update({(long double)data_proc->nFarPol(),
                       (long double)data->readNData()});

  if ( currStep > 0 ) {
    warn("Skipping initialization for restartd learner.");
    return;
  }
  // shift counters after initial data is gathered
  nDataGatheredB4Startup = data->readNSeen_loc();

  debugL("Compute state/rewards stats from the replay memory");
  profiler->stop_start("PRE");
  data_proc->updateRewardsStats(1, 1, true);
  //data_proc->updateRewardsStats(1, 1e-3, true);
  if( learn_rank == 0 )
    std::cout << "Initial reward std " << 1/data->scaledReward(1) << std::endl;

  data->initialize();

  if( not computeQretrace ) return;
  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  const Uint setSize = data->readNSeq();
  #pragma omp parallel for schedule(dynamic)
  for(Uint i=0; i<setSize; i++)
    for(Uint j=data->get(i)->ndata(); j>0; j--) backPropRetrace(data->get(i),j);
}

void Learner_offPolicy::save()
{
  const long int currStep = nGradSteps()+1;
  Learner::save();
  const Real freqSave = freqPrint * PRFL_DMPFRQ;
  const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = currStep % freqBackup == 0;
  if(not bBackup) return;

  std::ostringstream ss;
  ss<<learner_name<<"rank_"<<std::setfill('0')<<std::setw(3)<<learn_rank
    <<"_"<<std::setw(9)<<currStep<<"_learner.raw";
  FILE * f = fopen( ss.str().c_str(), "wb" );

  const Uint nObs=data->readNData(), tObs=data->readNSeen_loc();
  const Uint nSeqs=data->readNSeq(), tSeqs=data->readNSeenSeq_loc();
  fwrite(&nSeqs, sizeof(Uint), 1, f);
  fwrite(&nObs, sizeof(Uint), 1, f);
  fwrite(&tObs, sizeof(Uint), 1, f);
  fwrite(&tSeqs, sizeof(Uint), 1, f);
  fwrite(&nDataGatheredB4Startup, sizeof(long), 1, f);
  fwrite(&currStep, sizeof(long int), 1, f);
  fwrite(&beta, sizeof(Real), 1, f);
  fwrite(&CmaxRet, sizeof(Real), 1, f);

  for(Uint i = 0; i <nSeqs; i++)
    data->get(i)->save(f, sInfo.dimUsed, aInfo.dim, aInfo.policyVecDim);
}

void Learner_offPolicy::restart()
{
  Learner::restart();
  if(settings.restart == "none") return;
  std::ostringstream ss;
  ss<<"rank_"<<std::setfill('0')<<std::setw(3)<<learn_rank<<"_learner.raw";
  FILE * f = fopen((learner_name+ss.str()).c_str(), "rb");
  if(f == NULL) {
   _warn("Learner restart file %s not found\n",(learner_name+ss.str()).c_str());
   return;
  }

  Uint nSeqs;
  if(fread(&nSeqs,sizeof(Uint),1,f) != 1) die("");
  data->setNSeq(nSeqs);

  {
    Uint nObs;
    if(fread(&nObs, sizeof(Uint),1,f) != 1) die("");
    data->setNData(nObs);
  }
  {
    Uint tObs;
    if(fread(&tObs, sizeof(Uint),1,f) != 1) die("");
    data->setNSeen_loc(tObs);
  }
  {
    Uint tSeqs;
    if(fread(&tSeqs,sizeof(Uint),1,f) != 1) die("");
    data->setNSeenSeq_loc(tSeqs);
  }
  {
    long tB4;
    if(fread(&tB4, sizeof(long), 1, f) != 1) die("");
    nDataGatheredB4Startup = tB4;
  }
  {
    long int currStep;
    if(fread(&currStep, sizeof(long int), 1, f) != 1) die("");
    _nGradSteps = currStep;
    if(input->opt not_eq nullptr) input->opt->nStep = currStep;
    for(auto & net : F) net->opt->nStep = currStep;
  }
  if(fread(&beta, sizeof(Real), 1, f) != 1) die("");
  if(fread(&CmaxRet, sizeof(Real), 1, f) != 1) die("");

  for(Uint i = 0; i < nSeqs; i++) {
    assert(data->get(i) == nullptr);
    Sequence* const S = new Sequence();
    if( S->restart(f, sInfo.dimUsed, aInfo.dim, aInfo.policyVecDim) )
      _die("Unable to find sequence %u\n", i);
    data->set(S, i);
  }
}

void Learner_offPolicy::getMetrics(ostringstream& buff) const {
  real2SS(buff, alpha, 6, 1); real2SS(buff, beta, 6, 1);
}
void Learner_offPolicy::getHeaders(ostringstream& buff) const {
  buff << "| alph | beta ";
}
