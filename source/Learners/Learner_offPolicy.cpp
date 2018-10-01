//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_offPolicy.h"
#include "../Network/Optimizer.h"

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

bool Learner_offPolicy::blockDataAcquisition() const
{
  // lockQueue tells scheduler that has stopped receiving states from workers
  // whether should start communication again.
  // for off policy learning, there is a ratio between gradient steps
  // and observed transitions to be kept (approximatively) constant

  //if there is not enough data for training, need more data
  if( readyForTrain() not_eq true ) return false;

  //const Real _nData = (Real)data->readNConcluded() - nData_b4Startup;
  const long int mynData = data->readNSeen_loc() - nData_b4Startup;
  const long int mynStep = _nStep.load();
  // Lock the queue if we have !added to the training set! more observations
  // than (grad_step * obsPerStep) or if the update is ready.
  // The distinction between "added to set" and "observed" allows removing
  // some load inbalance, with only has marginal effects on algorithms.
  // Load imb. is reduced by minimizing pauses in either data or grad stepping.
  return mynData > mynStep * obsPerStep_loc;
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
    const auto* const S = data->get(samp_seq[i]);
    samp_obs[i] = S->ndata() - 1;
    nAddedGradients += ESpopSize * S->ndata();
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

  for(Uint i=0;i<batchSize;i++) data->get(samp_seq[i])->setSampled(samp_obs[i]);
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

    #ifdef PRIORITIZED_ER
      die("ReFER and Prioritized ER are incompatible. Set CmaxPol to 0");
    #endif
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
    ReFER_reduce.update({(Real)data_proc->nFarPol(), (Real)data->readNData()});
    const vector<Real> nFarGlobal = ReFER_reduce.get();
    const Real fracOffPol = nFarGlobal[0] / nFarGlobal[1];

    if(fracOffPol>ReFtol) beta = (1-1e-4)*beta; // iter converges to 0
    else beta = 1e-4 +(1-1e-4)*beta; //fixed point iter converge to 1
    if(std::fabs(ReFtol-fracOffPol)<0.01) alpha = (1-1e-4)*alpha;
    else alpha = 1e-4 + (1-1e-4)*alpha;
  }
}

void Learner_offPolicy::applyGradient()
{
  const Uint currStep = nStep()+1; //base class will advance this with this func
  if(updateToApply)
  {
    debugL("Finalize pruning of dataset");
    data_proc->finalize();
  }
  else die("undefined behavior");

  if( readyForTrain() )
  {
    debugL("Compute state/rewards stats from the replay memory");
    // placed here because this occurs after workers.join() so we have new data
    profiler->stop_start("PRE");
    if(currStep%1000==0) { // update state mean/std with net's learning rate
      const Real WS = annealRate(learnR, currStep, epsAnneal);
      data_proc->updateRewardsStats(1, WS*(OFFPOL_ADAPT_STSCALE>0));
    }
  }
  else die("Pruning removed too much data: no longer supported");

  Learner::applyGradient();
}

void Learner_offPolicy::initializeLearner()
{
  const Uint currStep = nStep();
  if ( not readyForTrain() ) die("undefined behavior");

  ReFER_reduce.update({(Real)data_proc->nFarPol(), (Real)data->readNData()});

  if ( currStep > 0 ) {
    warn("Skipping initialization for restartd learner.");
    return;
  }
  // shift counters after initial data is gathered
  nData_b4Startup = data->readNData();

  debugL("Compute state/rewards stats from the replay memory");
  profiler->stop_start("PRE");
  data_proc->updateRewardsStats(1, 1, true);
  if( learn_rank == 0 )
    std::cout << "Initial reward std " << 1/data->scaledReward(1) << std::endl;

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

  Uint nObs, tObs, nSeqs, tSeqs;
  if(fread(&nSeqs,sizeof(Uint),1,f) != 1) die(""); data->setNSeq(nSeqs);
  if(fread(&nObs, sizeof(Uint),1,f) != 1) die(""); data->setNData(nObs);
  if(fread(&tObs, sizeof(Uint),1,f) != 1) die(""); data->setNSeen_loc(tObs);
  if(fread(&tSeqs,sizeof(Uint),1,f) != 1) die(""); data->setNSeenSeq_loc(tSeqs);

  long int currStep;
  if(fread(&currStep, sizeof(long int), 1, f) != 1) die(""); _nStep = currStep;
  if(input->opt not_eq nullptr) input->opt->nStep = currStep;
  for(auto & net : F) net->opt->nStep = currStep;

  if(fread(&beta,            sizeof(Real), 1, f) != 1) die("");
  if(fread(&CmaxRet,         sizeof(Real), 1, f) != 1) die("");

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
