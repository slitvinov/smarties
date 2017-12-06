/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

class LSTMLayer: public Layer
{
  const Uint nInputs, nNeurons, link;
  const Function* const cell;

 public:
  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override
  {
    //cell, input, forget, output gates all linked to inp and prev LSTM output
    nWeight.push_back(4*nNeurons * (nInputs + nNeurons) );
    nBiases.push_back(4*nNeurons);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs) const override
  {
    sizes.push_back(4*nNeurons);
    bOutputs.push_back(bOutput);
  }
  virtual void biasInitialValues(const vector<nnReal> init) {}

  ~LSTMLayer() { _dispose_object(cell); }

  LSTMLayer(Uint _ID, Uint _nInputs, Uint _nNeurons, string funcType,
    bool bOut, Uint iLink) :  Layer(_ID, _nNeurons, bOut), nInputs(_nInputs),
    nNeurons(_nNeurons), link(iLink), cell(makeFunction(funcType))
  {
    printf("(%u) %s %sLSTM Layer of size:%u linked to Layer:%u of size:%u.\n",
    ID, funcType.c_str(), bOutput? "output ":"", nNeurons, ID-link, nInputs);
    fflush(0);
  }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    // suminp contains input to all cell inputs and gates
    // only one matrix-vector multiplication
    nnReal* const suminp = curr->X(ID);
    memcpy(suminp, para->B(ID), 4*nNeurons*sizeof(nnReal));
    {
      const nnReal* const inputs = curr->Y(ID-link);
      const nnReal* const weight = para->W(ID);
      for (Uint i = 0; i < nInputs; i++)
        for (Uint o = 0; o < 4*nNeurons; o++)
          suminp[o] += inputs[i] * weight[o + (4*nNeurons) * i];
    }

    if(prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
      const nnReal* const weight = para->W(ID) +(4*nNeurons)*nInputs;
      //first input loop, here input only prev step LSTM's output
      for (Uint i = 0; i < nNeurons; i++)
        for (Uint o = 0; o < 4*nNeurons; o++)
          suminp[o] += inputs[i] * weight[o + (4*nNeurons) * i];
    }
    {
      //now we computed prenonlinearity gates and cells first apply nonlinearity
      // cell output is written onto output work memory shifted by 2*nN
      cell->eval(suminp, curr->Y(ID)+ 2*nNeurons, nNeurons);
      // Input, forget, output gates output overwrite their input
      Sigm::_eval(suminp +1*nNeurons, suminp +1*nNeurons, nNeurons);
      Sigm::_eval(suminp +2*nNeurons, suminp +2*nNeurons, nNeurons);
      Sigm::_eval(suminp +3*nNeurons, suminp +3*nNeurons, nNeurons);

      // state is placed onto output work mem, shifted by nN
      const nnReal*const prevSt = prev==nullptr? nullptr : prev->Y(ID)+nNeurons;
            nnReal*const output = curr->Y(ID)+ 0*nNeurons;
            nnReal*const currSt = curr->Y(ID)+ 1*nNeurons;
      const nnReal*const cellOp = curr->Y(ID)+ 2*nNeurons;
      const nnReal*const inputG = curr->X(ID)+ 1*nNeurons;
      const nnReal*const forgtG = curr->X(ID)+ 2*nNeurons;
      const nnReal*const outptG = curr->X(ID)+ 3*nNeurons;

      for (Uint o=0; o<nNeurons; o++) {
       const nnReal oldStatePass = prev==nullptr? 0 : prevSt[o] * forgtG[o];
       currSt[o] = cellOp[o] * inputG[o] + oldStatePass;
       output[o] = outptG[o] * currSt[o];
      }
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
    const Uint nC = nNeurons; //lighten notation, number of cells
    const nnReal*const cellIn = curr->X(ID); //cell input before func
          nnReal*const deltas = curr->E(ID); //error signal from above/future
    const nnReal*const currState  = curr->Y(ID) + 1*nC;
    // Will also need to copy the state's error signal, use last available slot:
          nnReal*const stateDelta = curr->Y(ID) + 3*nC;
    // Also need output of input cell, and the 3 gates
    const nnReal*const cellInput  = curr->Y(ID) + 2*nC;
    const nnReal*const IGate = curr->X(ID)+ 1*nC;
    const nnReal*const FGate = curr->X(ID)+ 2*nC;
    const nnReal*const OGate = curr->X(ID)+ 3*nC;
    // prevState, nextState's delta and next output of forget gate
    const nnReal*const prvState = prev==nullptr? nullptr : prev->Y(ID) + 1*nC;
    const nnReal*const nxtStErr = next==nullptr? nullptr : next->Y(ID) + 3*nC;
    const nnReal*const nxtFGate = next==nullptr? nullptr : next->X(ID) + 2*nC;

    for (Uint o=0; o<nC; o++) {
      const nnReal D = deltas[o]; //before overwriting it
      // Compute state's error signal
      stateDelta[o] = D * OGate[o] +(next==nullptr? 0: nxtStErr[o]*nxtFGate[o]);
      // Compute deltas for cell input and gates
      deltas[o+0*nC] = cell->evalDiff(cellIn[o],D) * IGate[o] * stateDelta[o];
      deltas[o+1*nC] = IGate[o]*(1-IGate[o]) * cellInput[o]   * stateDelta[o];
      if(prev not_eq nullptr)
      deltas[o+2*nC] = FGate[o]*(1-FGate[o]) *  prvState[o]   * stateDelta[o];
      else deltas[o+2*nC] = 0;
      deltas[o+3*nC] = OGate[o]*(1-OGate[o]) * D * currState[o];
    }
    { // now that all is loaded in place, we can treat it like normal RNN layer
      nnReal* const grad_b = grad->B(ID);
      for(Uint o=0; o<4*nNeurons; o++) grad_b[o] += deltas[o];
    }
    {
      const nnReal* const inputs = curr->Y(ID-link);
            nnReal* const errors = curr->E(ID-link);
      const nnReal* const weight = para->W(ID);
            nnReal* const grad_w = grad->W(ID);

      for(Uint i=0; i<nInputs;  i++)
        for(Uint o=0; o<4*nNeurons; o++)
          grad_w[o +(4*nNeurons)*i] += inputs[i] * deltas[o];

      for(Uint o=0; o<4*nNeurons; o++)
        for(Uint i=0; i<nInputs;  i++)
          errors[i] += weight[o +(4*nNeurons)*i] * deltas[o];
    }
    if(prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
            nnReal* const errors = prev->E(ID);
      const nnReal* const weight = para->W(ID) +(4*nNeurons)*nInputs;
            nnReal* const grad_w = grad->W(ID) +(4*nNeurons)*nInputs;

      for(Uint i=0; i<nInputs;  i++)
        for(Uint o=0; o<4*nNeurons; o++)
          grad_w[o +(4*nNeurons)*i] += inputs[i] * deltas[o];

      for(Uint o=0; o<4*nNeurons; o++)
        for(Uint i=0; i<nInputs;  i++)
          errors[i] += weight[o +(4*nNeurons)*i] * deltas[o];
    }
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * cell->initFactor(nInputs, nNeurons);
    uniform_real_distribution<nnReal> dis(-init, init);
    { // forget gate starts open, inp/out gates are closed
     nnReal* const BB = para->B(ID);
     for(Uint o=0*nNeurons; o<1*nNeurons; o++) BB[o]=dis(*gen);
     for(Uint o=1*nNeurons; o<2*nNeurons; o++) BB[o]=dis(*gen)-LSTM_PRIME_FAC;
     for(Uint o=2*nNeurons; o<3*nNeurons; o++) BB[o]=dis(*gen)+LSTM_PRIME_FAC;
     for(Uint o=3*nNeurons; o<4*nNeurons; o++) BB[o]=dis(*gen)-LSTM_PRIME_FAC;
    }
    {
     nnReal* const weight = para->W(ID);
     for(Uint w=0; w<4*nNeurons*(nInputs+nNeurons); w++) weight[w] = dis(*gen);
    }
  }
};
