//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Approximator.h"
#include "Aggregator.h"
#include "Builder.h"

namespace smarties
{

Approximator::Approximator(const string _name, Settings&S, Encapsulator*const E,
  MemoryBuffer* const data_ptr, const Aggregator* const r) :
settings(S), name(_name), input(E), data(data_ptr), relay(r) { }

Approximator::~Approximator()
{
  Utilities::dispose_object(relay);
}

if (m_blockInpGrad or not preprocessing)
{
  if(auxInputAttachLayer<0)
  Uint layBckPrpInp = 1, nInputs = input->nOutputs();
  // make sure that we are computing relay gradient
  if(auxInputAttachLayer>0) { //then network attaches to a separate layer
    layBckPrpInp = 3;
    if(not net->layers[1]->bInput) die("should not be possible"); //relay
    if(net->layers[2]->bInput) die("should not be possible"); //joining
  }
  if (not auxInputNet) {
    if(net->layers.size() < layBckPrpInp)
      if(net->layers[layBckPrpInp]->spanCompInpGrads!=nInps)
        die("should not be possible");
  } else
    if(net->layers[layBckPrpInp]->spanCompInpGrads!=nInps+relay->nOutputs())
      die("should not be possible");

  if(net->layers.size() < layBckPrpInp) {
    net->layers[layBckPrpInp]->spanCompInpGrads -= nInps;
    net->layers[layBckPrpInp]->startCompInpGrads = nInps;
  }
}



void prepare_seq(Sequence*const traj, const Uint thrID,
  const Uint wghtID) const;

void prepare_one(Sequence*const traj, const Uint samp,
    const Uint thrID, const Uint wghtID) const;
void prepare(Sequence*const traj, const Uint samp, const Uint N,
    const Uint thrID, const Uint wghtID) const;









//const bool bRecurrent = settings.bRecurrent;
//const Uint mpisize = settings.learner_size;
//const Uint nMaxBPTT = settings.nnBPTTseq;
//Uint extraAlloc = 0;
} // end namespace smarties
