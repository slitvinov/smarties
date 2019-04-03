//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_pytorch.h"
#include <chrono>

Learner_pytorch::Learner_pytorch(Environment*const E, Settings&S): Learner(E, S)
{

}

Learner_pytorch::~Learner_pytorch() {
  // _dispose_object(input);
}

void Learner_pytorch::spawnTrainTasks()
{
  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  std::vector<Uint> samp_seq = std::vector<Uint>(batchSize, -1);
  std::vector<Uint> samp_obs = std::vector<Uint>(batchSize, -1);
  data->sample(samp_seq, samp_obs);

  for(Uint i=0; i<batchSize && bSampleSequences; i++)
    assert( samp_obs[i] == data->get(samp_seq[i])->ndata() - 1 );


}

void Learner_pytorch::getMetrics(std::ostringstream& buf) const
{
  Learner::getMetrics(buf);
}
void Learner_pytorch::getHeaders(std::ostringstream& buf) const
{
  Learner::getHeaders(buf);
}

void Learner_pytorch::restart()
{
  // if(settings.restart == "none") return;
  // if(!learn_rank) printf("Restarting from saved policy...\n");

  // for(auto & net : F) net->restart(settings.restart+"/"+learner_name);
  // input->restart(settings.restart+"/"+learner_name);

  // for(auto & net : F) net->save("restarted_"+learner_name, false);
  // input->save("restarted_"+learner_name, false);

  Learner::restart();

  // if(input->opt not_eq nullptr) input->opt->nStep = _nGradSteps;
  // for(auto & net : F) net->opt->nStep = _nGradSteps;
}

void Learner_pytorch::save()
{
  // const Uint currStep = nGradSteps()+1;
  // const Real freqSave = freqPrint * PRFL_DMPFRQ;
  // const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  // const bool bBackup = currStep % freqBackup == 0;
  // for(auto & net : F) net->save(learner_name, bBackup);
  // input->save(learner_name, bBackup);

  Learner::save();
}


