/*
 *  Functions.h
 *  rl
 *
 *  Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Functions.h"

struct Mem //Memory light recipient for prediction on agents
{
  Mem(Uint _nNeurons, Uint _nStates): nNeurons(_nNeurons), nStates(_nStates),
    outvals(initClean(nNeurons)), ostates(initClean(nStates)) {  }

  ~Mem()
  {
    _myfree(outvals);
    _myfree(ostates);
  }
  const Uint nNeurons, nStates;
  nnReal*const outvals;
  nnReal*const ostates;
};

struct Activation //All the network signals. TODO: vector of activations, one per layer, allowing classes of activations
{
  Activation(Uint _nNeurons,Uint _nStates):
    nNeurons(_nNeurons),nStates(_nStates),
    //contains all inputs to each neuron (inputs to network input layer is empty)
    in_vals(init(nNeurons)),
    //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
    outvals(init(nNeurons)),
    //deltas for each neuron
    errvals(initClean(nNeurons)),
    //memory and inputs to gates (cell into in_vals)
    ostates(init(nNeurons)), iIGates(init(nNeurons)),
    iFGates(init(nNeurons)), iOGates(init(nNeurons)),
    //output of gates and LSTM cell
    oMCell(init(nNeurons)), oIGates(init(nNeurons)),
    oFGates(init(nNeurons)), oOGates(init(nNeurons)),
    //errors of gates and LSTM cell
    eMCell(init(nNeurons)), eIGates(init(nNeurons)),
    eFGates(init(nNeurons)), eOGates(init(nNeurons))
  { }

  ~Activation()
  {
    _myfree(in_vals);
    _myfree(outvals);
    _myfree(errvals);
    _myfree(ostates);

    _myfree(iIGates);
    _myfree(iFGates);
    _myfree(iOGates);

    _myfree(oMCell);
    _myfree(oIGates);
    _myfree(oFGates);
    _myfree(oOGates);

    _myfree(eMCell);
    _myfree(eIGates);
    _myfree(eFGates);
    _myfree(eOGates);
  }

  inline void clearOutput()
  {
    std::memset(outvals,0.,nNeurons*sizeof(nnReal));
    std::memset(ostates,0.,nStates*sizeof(nnReal));
    std::memset(oMCell, 0.,nStates*sizeof(nnReal));
    std::memset(oIGates,0.,nStates*sizeof(nnReal));
    std::memset(oFGates,0.,nStates*sizeof(nnReal));
    std::memset(oOGates,0.,nStates*sizeof(nnReal));
  }

  inline void clearErrors()
  {
    std::memset(errvals,0.,nNeurons*sizeof(nnReal));
    std::memset(eOGates,0.,nStates*sizeof(nnReal));
    std::memset(eIGates,0.,nStates*sizeof(nnReal));
    std::memset(eFGates,0.,nStates*sizeof(nnReal));
    std::memset(eMCell,0.,nStates*sizeof(nnReal));
  }

  inline void clearInputs()
  {
    std::memset(in_vals,0.,nNeurons*sizeof(nnReal));
    std::memset(iIGates,0.,nStates*sizeof(nnReal));
    std::memset(iFGates,0.,nStates*sizeof(nnReal));
    std::memset(iOGates,0.,nStates*sizeof(nnReal));
  }

  inline void loadMemory(Mem*const _M)
  {
    assert(_M->nNeurons == nNeurons);
    assert(_M->nStates == nStates);
    for (Uint j=0; j<nNeurons; j++) outvals[j] = _M->outvals[j];
    for (Uint j=0; j<nStates;  j++) ostates[j] = _M->ostates[j];
  }

  inline void storeMemory(Mem*const _M)
  {
    assert(_M->nNeurons == nNeurons);
    assert(_M->nStates == nStates);
    for (Uint j=0; j<nNeurons; j++) _M->outvals[j] = outvals[j];
    for (Uint j=0; j<nStates;  j++) _M->ostates[j] = ostates[j];
  }

  const Uint nNeurons, nStates;
  nnReal*const in_vals;
  nnReal*const outvals;
  nnReal*const errvals;
  nnReal*const ostates;
  nnReal*const iIGates;
  nnReal*const iFGates;
  nnReal*const iOGates;
  nnReal*const oMCell;
  nnReal*const oIGates;
  nnReal*const oFGates;
  nnReal*const oOGates;
  nnReal*const eMCell;
  nnReal*const eIGates;
  nnReal*const eFGates;
  nnReal*const eOGates;
};

struct Grads
{
  Grads(Uint _nWeights, Uint _nBiases): nWeights(_nWeights), nBiases(_nBiases),
    _W(initClean(_nWeights)), _B(initClean(_nBiases)) { }

  ~Grads()
  {
    _myfree(_W);
    _myfree(_B);
  }
  inline void clear()
  {
    std::memset(_W,0.,nWeights*sizeof(nnReal));
    std::memset(_B,0.,nBiases*sizeof(nnReal));
  }
  const Uint nWeights, nBiases;
  nnReal*const _W;
  nnReal*const _B;
};

inline void circle_region(Grads*const trust, Grads*const grad, const Real delta, const int ngrads)
{
  #if 1
    assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
    Real norm = 0;
    for(Uint j=0; j<trust->nWeights; j++)
      norm += std::pow((grad->_W[j]+trust->_W[j])/ngrads, 2);
    for(Uint j=0; j<trust->nBiases; j++)
      norm += std::pow((grad->_B[j]+trust->_B[j])/ngrads, 2);

    const Real nG = std::sqrt(norm), softclip = delta/(nG+delta);
    //printf("grad norm %f\n",nG);
    for(Uint j=0; j<trust->nWeights; j++)
      grad->_W[j] = (grad->_W[j]+trust->_W[j])*softclip -trust->_W[j];
    for(Uint j=0; j<trust->nBiases; j++)
      grad->_B[j] = (grad->_B[j]+trust->_B[j])*softclip -trust->_B[j];
  #else
    Real dot=0, norm = numeric_limits<Real>::epsilon();
    for(Uint j=0; j<trust->nWeights; j++) {
      norm += std::pow(trust->_W[j]/ngrads, 2);
      dot += grad->_W[j]*trust->_W[j]/(ngrads*ngrads);
    }
    for(Uint j=0; j<trust->nBiases; j++)  {
      norm += std::pow(trust->_B[j]/ngrads, 2);
      dot += grad->_B[j]*trust->_B[j]/(ngrads*ngrads);
    }
    const Real proj = std::max( (Real)0, (dot - delta)/norm );
    //printf("grad norm %f %f %f\n", proj, dot, norm);
    for(Uint j=0; j<trust->nWeights; j++)
      grad->_W[j] = grad->_W[j] -proj*trust->_W[j];
    for(Uint j=0; j<trust->nBiases; j++)
      grad->_B[j] = grad->_B[j] -proj*trust->_B[j];
  #endif
  trust->clear();
}

inline void circle_region(Grads*const grad, Grads*const trust, Grads*const dest, const Real delta)
{
  assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
  Real norm = 0;
  for(Uint j=0;j<trust->nWeights;j++)
    norm += std::pow(grad->_W[j]+trust->_W[j],2);
  for(Uint j=0;j<trust->nBiases; j++)
    norm += std::pow(grad->_B[j]+trust->_B[j],2);

  const Real nG = std::sqrt(norm), softclip = delta/(nG+delta);

  for(Uint j=0;j<trust->nWeights;j++)
    dest->_W[j] += (grad->_W[j]+trust->_W[j])*softclip -trust->_W[j];
  for(Uint j=0;j<trust->nBiases; j++)
    dest->_B[j] += (grad->_B[j]+trust->_B[j])*softclip -trust->_B[j];
  trust->clear();
  grad->clear();
}

inline void fullstats(Grads*const grad, Grads*const trust, Grads*const dest, const Real delta)
{
  assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
  Real EO1 = 0, EO2 = 0, EO3 = 0, EO4 = 0;
  Real EL1 = 0, EL2 = 0, EL3 = 0, EL4 = 0;
  Real EC1 = 0, EC2 = 0, EC3 = 0, EC4 = 0;

  Real dot=0, norm=numeric_limits<Real>::epsilon(), sum=0,  dotL=0, dotC=0;
  for(Uint j=0; j<trust->nWeights; j++) {
    sum += std::pow(grad->_W[j]+trust->_W[j],2);
    norm += trust->_W[j]*trust->_W[j];
    dot +=   grad->_W[j]*trust->_W[j];
  }
  for(Uint j=0; j<trust->nBiases; j++)  {
    sum += std::pow(grad->_B[j]+trust->_B[j],2);
    norm += trust->_B[j]*trust->_B[j];
    dot +=   grad->_B[j]*trust->_B[j];
  }
  const Real proj = std::max( (Real)0, (dot - delta)/norm );
  const Real nG = std::sqrt(sum), clip = delta/(nG+delta);

  for(Uint j=0; j<trust->nWeights; j++) {
    const long double linear = grad->_W[j] -proj*trust->_W[j];
    const long double circle = (grad->_W[j]+trust->_W[j])*clip -trust->_W[j];
    dotL += linear*trust->_W[j];
    dotC += circle*trust->_W[j];
    dest->_W[j] += circle;
  }
  for(Uint j=0; j<trust->nBiases; j++)  {
    const long double linear = grad->_B[j] -proj*trust->_B[j];
    const long double circle = (grad->_B[j]+trust->_B[j])*clip -trust->_B[j];
    dotL += linear*trust->_B[j];
    dotC += circle*trust->_B[j];
    dest->_B[j] += circle;
  }

  if(omp_get_thread_num() == 1) {
    EO1 =          dot/std::sqrt(norm);      // to compute E[grad_proj_dkl]
    EO2 = std::pow(dot/std::sqrt(norm), 2);  //higher order statistics
    EO3 = std::pow(dot/std::sqrt(norm), 3);
    EO4 = std::pow(dot/std::sqrt(norm), 4);
    EL1 =          dotL/std::sqrt(norm);      // to compute E[grad_proj_dkl]
    EL2 = std::pow(dotL/std::sqrt(norm), 2);  //higher order statistics
    EL3 = std::pow(dotL/std::sqrt(norm), 3);
    EL4 = std::pow(dotL/std::sqrt(norm), 4);
    EC1 =          dotC/std::sqrt(norm);      // to compute E[grad_proj_dkl]
    EC2 = std::pow(dotC/std::sqrt(norm), 2);  //higher order statistics
    EC3 = std::pow(dotC/std::sqrt(norm), 3);
    EC4 = std::pow(dotC/std::sqrt(norm), 4);
    ofstream fs;
    fs.open("gradproj_dist.txt", ios::app);
    fs<<EO1<<"\t"<<EO2<<"\t"<<EO3<<"\t"<<EO4<<"\t"
      <<EL1<<"\t"<<EL2<<"\t"<<EL3<<"\t"<<EL4<<"\t"
      <<EC1<<"\t"<<EC2<<"\t"<<EC3<<"\t"<<EC4<<endl;
    fs.close(); fs.flush();
  }

  trust->clear();
  grad->clear();
}
