//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the “CC BY-SA 4.0” license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_offPolicy.h"

Learner_offPolicy::Learner_offPolicy(Environment*const _env, Settings & _s) :
Learner(_env,_s), obsPerStep_orig(_s.obsPerStep), nObsPerTraining(
_s.minTotObsNum>_s.batchSize? _s.minTotObsNum : _s.maxTotObsNum) {
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

  if(not ready && bTrain && !learn_rank) {
    lock_guard<mutex> lock(buffer_mutex);
    const int currPerc = data->readNData() * 100. / (Real) nObsPerTraining;
    if(currPerc>=percData+5) {
      percData = currPerc;
      printf("\rCollected %d%% of data required to begin training. ", percData);
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
  const Real _nData = data->readNSeen() - nData_b4Startup;
  const Real dataCounter = _nData - (Real)nData_last;
  const Real stepCounter =  nStep - (Real)nStep_last;
  // Lock the queue if we have !added to the training set! more observations
  // than (grad_step * obsPerStep) or if the update is ready.
  // The distinction between "added to set" and "observed" allows removing
  // some load inbalance, with only has marginal effects on algorithms.
  // Load imb. is reduced by minimizing pauses in either data or grad stepping.
  const bool tooMuchData = dataCounter > stepCounter*obsPerStep;
  return tooMuchData;
}

void Learner_offPolicy::spawnTrainTasks_par()
{
  // it should be impossible to get here before starting batch update was ready
  if(updateComplete || updateToApply) die("undefined behavior");

  if( ! readyForTrain() ) return; // Do not prepare an update

  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  vector<Uint> samp_seq = vector<Uint>(batchSize, -1);
  vector<Uint> samp_obs = vector<Uint>(batchSize, -1);
  if(bSampleSequences) data->sampleSequences(samp_seq);
  else data->sampleTransitions_OPW(samp_seq, samp_obs);

  profiler->stop_start("SLP"); // so we see inactive time during parallel loop
  #pragma omp parallel for schedule(dynamic)
  for (Uint i=0; i<batchSize; i++)
  {
    Uint seq = samp_seq[i], obs = samp_obs[i];
    const int thrID = omp_get_thread_num();
    //printf("Thread %d done %u %u %f\n",thrID,seq,obs,data->Set[seq]->offPolicImpW[obs]); fflush(0);
    if(bSampleSequences)
    {
      obs = data->Set[seq]->ndata()-1;
      TrainBySequences(seq, thrID);
      #pragma omp atomic
      nAddedGradients += data->Set[seq]->ndata();
    }
    else
    {
      //data->sampleTransition(seq, obs, thrID);
      Train(seq, obs, thrID);
      #pragma omp atomic
      nAddedGradients++;
    }

    input->gradient(thrID);
    data->Set[seq]->setSampled(obs);
    if(thrID==0) profiler->stop_start("SLP");
  }

  updateComplete = true;
}

bool Learner_offPolicy::bNeedSequentialTrain() {return false;}
void Learner_offPolicy::spawnTrainTasks_seq() { }

void Learner_offPolicy::applyGradient()
{
  if(not updateToApply)
  {
    nData_b4Startup = data->readNConcluded();
    nData_last = 0;
  }
  else
  {
    profiler->stop_start("PRNE");
    advanceCounters();
    if(CmaxRet>0) // assume ReF-ER
    {
      CmaxRet = 1 + annealRate(CmaxPol, nStep, epsAnneal);
      if(CmaxRet<=1) die("Either run lasted too long or epsAnneal is wrong.");
      data->prune(FARPOLFRAC, CmaxRet);
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
        if(skipped) // it must be the first step: nothing is far policy yet
          assert(partial_data[0] < nnEPS);
      }

      if(fracOffPol>ReFtol) beta = (1-learnR)*beta; // iter converges to 0
      else beta = learnR +(1-learnR)*beta; //fixed point iter converge to 1

      if( beta <= 10*learnR && nStep % 1000 == 0)
      warn("beta too low. Lower lrate, pick bounded nnfunc, or incr net size.");
    }
    else
    {
      data->prune(MEMBUF_FILTER_ALGO);
    }
  }

  Learner::applyGradient();

  if( readyForTrain() )
  {
    profiler->stop_start("PRE");
    if(nStep%1000==0) { // update state mean/std with net's learning rate
      //const Real WS = nStep? annealRate(learnR, nStep, epsAnneal) : 1;
      const Real WS = nStep? 0 : 1;
      data->updateRewardsStats(nStep, 1, WS*(LEARN_STSCALE>0));
    }
    if(nStep == 0 && !learn_rank)
      cout<<"Initial reward std "<<1/data->invstd_reward<<endl;
  }
}
