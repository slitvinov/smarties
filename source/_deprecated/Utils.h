//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once
struct MiniBatchThreadContext
{
  const MiniBatch * batch;
  Uint nAddedSamples;
  Uint batchSize;
  std::vector<std::vector<Sint>> lastGradTstep;
  std::vector<Sint> weightID;
  std::vector< //vector over evaluations (eg. target/current or many samples)
    std::vector< // vector over mini batch size
      std::vector<Activation*>>> activations; // vector over time

  MiniBatchContext();

  load(MiniBatch&B, Uint nAdded=0)
  {
    batch = & B;
    nAddedSamples = nAdded;
    batchSize = batch->size;
    lastGradTstep.resize(1+nAddedSamples, std::vector<Sint>(batchSize, -1) );
    weightID = std::vector<Sint>(batchSize, -1);
    activations.resize(1+nAddedSamples);
    for(Uint i=0; i<=nAddedSamples; ++i) {
      activations[i].resize(batchSize);
      for(Uint b=0; b<batchSize; ++b)
        Network::allocTimeSeries(activations[i][b], batch->getNumSteps(b) );
    }
  }

  Sint endBackPropStep(const Uint b, const Uint sample = 0) const
  {
    assert(lastGradTstep.size() > sample);
    assert(lastGradTstep[sample].size() > b);
    return lastGradTstep[sample][b];
  }
  Sint usedWeightID(const Uint b) const
  {
    assert(weightID.size() > b);
    return weightID[b];
  }
  Activation * net(const Uint b, const Uint t, const Uint sample = 0) const
  {
    assert(weightID.size() > b);
    return weightID[b];
  }
  Uint mapTime2Ind(const Uint b, const Uint t) const
  {
    assert(batch not_eq nullptr);
    return batch->mapTime2Ind(b, t);
  }
};
