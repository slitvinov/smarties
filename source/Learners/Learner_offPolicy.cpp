//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_offPolicy.h"

Learner_offPolicy::Learner_offPolicy(Environment*const _env, Settings & _s) :
Learner(_env,_s) {
  if(not bSampleSequences && nObsPerTraining < batchSize)
    die("Parameter minTotObsNum is too low for given problem");
}

bool Learner_offPolicy::readyForTrain() const
{
  //const Uint nTransitions = data->readNTransitions();
  //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
  //  die("I do not have enough data for training. Change hyperparameters");
  //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
  const bool ready = bTrain && data->readNData() >= nObsPerTraining;

  if(not ready && bTrain && learn_rank==0)
  {
    lock_guard<mutex> lock(buffer_mutex);
    const int currPerc = data->readNData() * 100. / (Real) nObsPerTraining;
    if(currPerc>=percData+5) {
      percData = currPerc;
      printf("\rCollected %d%% of data required to begin training. ", percData);
      fflush(0); //otherwise no show on some platforms
    }
  }
  return ready;
}

bool Learner_offPolicy::lockQueue() const
{
  // lockQueue tells scheduler that has stopped receiving states from workers
  // whether should start communication again.
  // for off policy learning, there is a ratio between gradient steps
  // and observed transitions to be kept (approximatively) constant

  //if there is not enough data for training, need more data
  if( not readyForTrain() ) return false;

  //const Real _nData = (Real)data->readNConcluded() - nData_b4Startup;
  const long int mynData = data->readNSeen() - nData_b4Startup.load();
  const long int mynStep = nStep() - nStep_b4Startup.load();
  // Lock the queue if we have !added to the training set! more observations
  // than (grad_step * obsPerStep) or if the update is ready.
  // The distinction between "added to set" and "observed" allows removing
  // some load inbalance, with only has marginal effects on algorithms.
  // Load imb. is reduced by minimizing pauses in either data or grad stepping.
  const bool tooMuchData = mynData > mynStep*obsPerStep;
  return tooMuchData;
}

void Learner_offPolicy::spawnTrainTasks_par()
{
  // it should be impossible to get here before starting batch update was ready
  if(updateComplete || updateToApply) die("undefined behavior");

  if( not readyForTrain() ) {
    warn("spawnTrainTasks_par called with not enough data, wait next call");
    // This can happen if data pruning algorithm is allowed to delete a lot of
    // data from the mem buffer, which could cause training to pause
    return; // Do not prepare an update
  }

  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");
  debugL("Sample the replay memory and compute the gradients");
  vector<Uint> samp_seq = vector<Uint>(batchSize, -1);
  vector<Uint> samp_obs = vector<Uint>(batchSize, -1);
  if(bSampleSequences) data->sampleSequences(samp_seq);
  else data->sampleTransitions(samp_seq, samp_obs);

  nAddedGradients = 0;
  for (Uint i=0; i<batchSize && bSampleSequences; i++) {
    samp_obs[i] = data->Set[samp_seq[i]]->ndata() - 1;
    nAddedGradients += ESpopSize * data->Set[samp_seq[i]]->ndata();
  }
  if(not bSampleSequences) nAddedGradients = ESpopSize * batchSize;

  if(bSampleSequences) {
  #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(nThreads)
    for (Uint wID=0; wID<ESpopSize; wID++)
      for (Uint bID=0; bID<batchSize; bID++) {
        const Uint thrID = omp_get_thread_num();
        TrainBySequences(samp_seq[bID], wID, bID, thrID);
        input->gradient(thrID);
      }
  } else {
  #pragma omp parallel for collapse(2) schedule(static, 1) num_threads(nThreads)
    for (Uint wID=0; wID<ESpopSize; wID++)
      for (Uint bID=0; bID<batchSize; bID++) {
        const Uint thrID = omp_get_thread_num();
        Train(samp_seq[bID], samp_obs[bID], wID, bID, thrID);
        input->gradient(thrID);
      }
  }

  for(Uint i=0;i<batchSize;i++) data->Set[samp_seq[i]]->setSampled(samp_obs[i]);
  updateComplete = true;
}

bool Learner_offPolicy::bNeedSequentialTrain() {return false;}
void Learner_offPolicy::spawnTrainTasks_seq() { }

void Learner_offPolicy::prepareGradient()
{
  Learner::prepareGradient();

  const Uint currStep = nStep()+1; //base class will advance this with this func
  if(updateToApply)
  {
    debugL("Prune the Replay Memory for old/stale episodes, advance counters");
    //put here because this is called after workers finished gathering new data
    profiler->stop_start("PRNE");
    //shift data / gradient counters to maintain grad stepping to sample
    // collection ratio prescirbed by obsPerStep

    if(CmaxPol>0) // assume ReF-ER
    {
      #ifdef PRIORITIZED_ER
        die("ReFER and Prioritized ER are incompatible. Set CmaxPol to 0");
      #endif
      CmaxRet = 1 + annealRate(CmaxPol, currStep, epsAnneal);
      CinvRet = 1 / CmaxRet;
      if(CmaxRet<=1) die("Either run lasted too long or epsAnneal is wrong.");
      data->prune(ERFILTER, CmaxRet);
      Real fracOffPol = data->nOffPol / (Real) data->readNData();

      if (learn_size > 1) {
        vector<Real> partial_data {(Real)data->nOffPol,(Real)data->readNData()};
        // use result from prev AllReduce to update rewards (before new reduce).
        // Assumption is that the number of off Pol trajectories does not change
        // much each step. Especially because here we update the off pol W only
        // if an obs is actually sampled. Therefore at most this fraction
        // is wrong by batchSize / nTransitions ( ~ 0 )
        // In exchange we skip an mpi implicit barrier point.
        const bool skipped = reductor.sync(partial_data);
        fracOffPol = partial_data[0] / partial_data[1];
        if(skipped and partial_data[0]>nnEPS)
          die("If skipping it must be 1st step, with nothing far policy");
      }

      if(fracOffPol>ReFtol) beta = (1-1e-4)*beta; // iter converges to 0
      else beta = 1e-4 +(1-1e-4)*beta; //fixed point iter converge to 1
      if(std::fabs(ReFtol-fracOffPol)<0.01) alpha = (1-1e-4)*alpha;
      else alpha = 1e-4 + (1-1e-4)*alpha;
    }
    else
    {
      data->prune(ERFILTER);
    }
  }
}

void Learner_offPolicy::applyGradient()
{
  const Uint currStep = nStep()+1; //base class will advance this with this func
  if(updateToApply)
  {
    debugL("Finalize pruning of dataset");
    data->finalize();
  }
  else
  {
    if( not readyForTrain() ) die("undefined behavior");
    warn("Pruning at prev grad step removed too much data and training was paused: shift training counters");
    // Prune at prev grad step removed too much data and training was paused.
    // ApplyGradient was surely called by Scheduler after workers finished
    // gathering new data enabling training to continue ( after workers.join() )
    // Therefore we should shift these counters to restart gradient stepping:
    nData_b4Startup = data->readNConcluded();
    nStep_b4Startup = nStep();
  }

  if( readyForTrain() )
  {
    debugL("Compute state/rewards stats from the replay memory");
    // placed here because this occurs after workers.join() so we have new data
    profiler->stop_start("PRE");
    if(currStep%1000==0) { // update state mean/std with net's learning rate
      const Real WS = annealRate(learnR, currStep, epsAnneal);
      data->updateRewardsStats(currStep, 1, WS*(OFFPOL_ADAPT_STSCALE>0));
    }
  }
  else
  {
    warn("Pruning removed too much data from buffer: will have to wait one scheduler loop before training can continue");
  }

  Learner::applyGradient();
}

void Learner_offPolicy::initializeLearner()
{
  const Uint currStep = nStep();
  if ( not readyForTrain() ) die("undefined behavior");
  if ( currStep > 0 ) {
    warn("Skipping initialization for restartd learner.");
    return;
  }
  // shift counters after initial data is gathered
  nData_b4Startup = data->readNConcluded();
  nStep_b4Startup = currStep;

  debugL("Compute state/rewards stats from the replay memory");
  profiler->stop_start("PRE");
  data->updateRewardsStats(currStep, 1, 1);
  if( learn_rank == 0 )
    cout<<"Initial reward std "<<1/data->invstd_reward<<endl;

  Learner::initializeLearner();
}

void Learner_offPolicy::save()
{
  const long int currStep = nStep()+1;
  Learner::save();
  static constexpr Real freqSave = 1000*PRFL_DMPFRQ;
  const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = currStep % freqBackup == 0;
  if(not bBackup) return;

  ostringstream ss; ss << std::setw(9) << std::setfill('0') << currStep;
  FILE* f = fopen((learner_name+ss.str()+"_learner.raw").c_str(), "wb");
  Uint val;
  val = data->Set.size(); fwrite(&val, sizeof(Uint), 1, f);
  val = data->nSequences.load(); fwrite(&val, sizeof(Uint), 1, f);
  val = data->nTransitions.load(); fwrite(&val, sizeof(Uint), 1, f);
  val = data->nSeenSequences.load(); fwrite(&val, sizeof(Uint), 1, f);
  val = data->nSeenTransitions.load(); fwrite(&val, sizeof(Uint), 1, f);
  val = data->nCmplTransitions.load(); fwrite(&val, sizeof(Uint), 1, f);
  val = data->iOldestSaved.load(); fwrite(&val, sizeof(Uint), 1, f);
  fwrite(&currStep, sizeof(long int), 1, f);
  fwrite(&beta, sizeof(Real), 1, f);
  fwrite(&CmaxRet, sizeof(Real), 1, f);

  for(Uint i = 0; i < data->Set.size(); i++)
    data->Set[i]->save(f, sInfo.dimUsed, aInfo.dim, aInfo.policyVecDim);
}

void Learner_offPolicy::restart()
{
  Learner::restart();
  if(settings.restart == "none") return;
  const string fname = learner_name+"learner.raw";
  FILE* f = fopen(fname.c_str(), "rb");
  if(f == NULL) {
    _warn("Did not find learner state file %s\n", fname.c_str()); return;
  }
  Uint val;
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->Set.resize(val, nullptr);
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->nSequences = val;
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->nTransitions = val;
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->nSeenSequences = val;
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->nSeenTransitions = val;
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->nCmplTransitions = val;
  if(fread(&val,sizeof(Uint),1,f) != 1) die(""); data->iOldestSaved = val;

  long int cnter;
  if(fread(&cnter, sizeof(long int), 1, f) != 1) die(""); _nStep = cnter;
  if(input->opt not_eq nullptr) input->opt->nStep = cnter;
  for(auto & net : F) net->opt->nStep = cnter;

  if(fread(&beta,            sizeof(Real), 1, f) != 1) die("");
  if(fread(&CmaxRet,         sizeof(Real), 1, f) != 1) die("");

  for(Uint i = 0; i < data->Set.size(); i++) {
    assert(data->Set[i] == nullptr);
    data->Set[i] = new Sequence();
    if( data->Set[i]->restart(f, sInfo.dimUsed, aInfo.dim, aInfo.policyVecDim) )
      _die("Unable to find sequence %u\n", i);
  }
}

void Learner_offPolicy::getMetrics(ostringstream& buff) const {
  real2SS(buff, alpha, 6, 1); real2SS(buff, beta, 6, 1);
}
void Learner_offPolicy::getHeaders(ostringstream& buff) const {
  buff << "| alph | beta ";
}
