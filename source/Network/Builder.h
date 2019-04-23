//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Builder_h
#define smarties_Builder_h

namespace smarties
{

class Builder
{
public:
  void addInput(const int size);

  /*
    addLayer adds fully conn. layer:
      - nNeurons: simply the size of the layer (for LSTM is number of cells)
      - funcType: non-linearity applied to the matrix-vector mul
                  (for LSTM is function applied cell input, gates have sigmoid)
      - bOutput: whether layer is output and therefore copied into return
                 vector when calling Network:predict
      - layerType: LSTM, RNN, else assumed MLP
      - iLink: how many layers back should layer take the input from.
               iLink=1 means that input is previous layer
               iLink=2 means input is *only* the output of 2 layers below
               This allows networks with multiple heads, but always each
               layer has only one input layer (+ eventual recurrent connection).
  */
  void addLayer(const int nNeurons,
                const std::string funcType,
                const bool bOutput=false,
                const std::string layerType="",
                const int iLink = 1);

  void setLastLayersBias(std::vector<Real> init_vals);

  void addParamLayer(int size, std::string funcType="Linear", Real init=0);

  void addParamLayer(int size, std::string funcType, std::vector<Real> init);

  void addConv2d(const bool bOutput=false, const int iLink = 1);

  // Function that initializes and constructs net and optimizer.
  // Once this is called number of layers or weights CANNOT be modified.
  Network* build(const bool isInputNet = false);

  // stackSimple reads from the settings file the amount of fully connected
  // layers (nnl1, nnl2, ...) and builds a network with given number of nInputs
  // and nOutputs. Supports LSTM, RNN and MLP (aka InnerProduct or Dense).
  //void stackSimple(Uint ninps,Uint nouts) { return stackSimple(ninps,{nouts}); }
  void stackSimple(const Uint ninps, const vector<Uint> nouts);

private:
  bool bBuilt = false;
public:
  const Settings & settings;
  const Uint nThreads = settings.nThreads;
  const Uint CMApopSize = settings.ESpopSize;
  const Uint mpisize = settings.learner_size;
  Uint nInputs=0, nOutputs=0, nLayers=0;
  Real gradClip = 1;
  std::vector<std::mt19937>& generators = settings.generators;
  Parameters *weights, *tgt_weights;
  std::vector<Parameters*> Vgrad;
  std::vector<Parameters*> popW;
  std::vector<Layer*> layers;

  Network* net = nullptr;
  Optimizer* opt = nullptr;

  Builder(const Settings& _sett);
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
