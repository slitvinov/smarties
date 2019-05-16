//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Network_h
#define smarties_Network_h

#include "Layers/Layers.h"

namespace smarties
{

class Builder;

class Network
{
public:
  const std::vector<std::unique_ptr<Layer>> layers;
  const Uint nInputs, nOutputs, nLayers = layers.size();
  const std::shared_ptr<Parameters> weights;
  Uint getnOutputs() const { return nOutputs; }
  Uint getnInputs()  const { return nInputs;  }
  Uint getnLayers()  const { return nLayers;  }

  static std::shared_ptr<Parameters> allocParameters(
        const std::vector<std::unique_ptr<Layer>>& layers, const Uint mpiSize)
  {
    std::vector<Uint> nWeight, nBiases;
    for(const auto & l : layers) l->requiredParameters(nWeight, nBiases);
    return std::make_shared<Parameters>(nWeight, nBiases, mpiSize);
  }
  static std::unique_ptr<Activation> allocActivation(
        const std::vector<std::unique_ptr<Layer>>& layers)
  {
    std::vector<Uint> sizes, output, input;
    for(const auto & l : layers) l->requiredActivation(sizes, output, input);
    return std::make_unique<Activation>(sizes, output, input);
  }
  std::unique_ptr<Activation> allocActivation() const
  {
    return allocActivation(layers);
  }
  std::shared_ptr<Parameters> allocParameters() const
  {
    return weights->allocateEmptyAlike();
  }

  void allocTimeSeries(std::vector<std::unique_ptr<Activation>>& series,
                       const Uint N) const
  {
    if (series.size() < N)
      for(Uint j=series.size(); j<N; ++j)
        series.emplace_back( allocActivation() );
    assert(series.size()>=N);

    for(Uint j=0; j<series.size(); ++j) {
      series[j]->clearErrors();
      series[j]->written = false;
    }

    #ifndef NDEBUG
    for(Uint j=0; j<series.size(); j++) assert(not series[j]->written);
    #endif
  }

  Network(const Uint _nInp, const Uint _nOut,
          std::vector<std::unique_ptr<Layer>>& _layers,
          const std::shared_ptr<Parameters>& _weights);

  std::vector<Real> predict(const std::vector<Real>& _inp,
    const std::vector<Activation*>& timeSeries, const Uint step,
    const Parameters*const _weights = nullptr) const
  {
    assert(timeSeries.size() > step);
    const Activation*const currStep = timeSeries[step];
    const Activation*const prevStep = step==0 ? nullptr : timeSeries[step-1];
    return predict(_inp, prevStep, currStep, _weights);
  }

  std::vector<Real> predict(const std::vector<Real>& _inp,
    const Activation* const currStep,
    const Parameters*const _weights = nullptr) const
  {
    return predict(_inp,  nullptr, currStep, _weights);
  }

  std::vector<Real> predict(const std::vector<Real>& _inp,
    const Activation* const prevStep, const Activation* const currStep,
    const Parameters*const _weights = nullptr) const;

  void backProp(const Activation*const currStep, const Parameters*const _grad,
                const Parameters*const _weights = nullptr) const
  {
    return backProp(nullptr, currStep, nullptr, _grad, _weights);
  }

  void backProp(const std::vector<Real>& _errors, const Activation*const currStep,
    const Parameters*const _grad, const Parameters*const _weights=nullptr) const
  {
    currStep->clearErrors();
    currStep->setOutputDelta(_errors);
    assert(currStep->written);
    _grad->written = true;
    backProp(nullptr, currStep, nullptr, _grad, _weights);
  }

  std::vector<Real> backPropToLayer(const std::vector<Real>& gradient,
                                    const Uint toLayerID,
                                    const Activation*const activation,
                                    const Parameters*const _weights) const;

  void backProp(const std::vector<std::unique_ptr<Activation>>& timeSeries,
                const Uint stepLastError,
                const Parameters*const _gradient,
                const Parameters*const _weights = nullptr) const;

  void backProp(const Activation*const prevStep,
                const Activation*const currStep,
                const Activation*const nextStep,
                const Parameters*const _gradient,
                const Parameters*const _weights = nullptr) const;

  void checkGrads();
  //void dump(const int agentID);
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
