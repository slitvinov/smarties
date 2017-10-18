/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Learner.h"

#if 0
class Input_layer
{
  /*
    Skeleton. Class will take user's builder specification and create feature procesing layers. THis will be a common layer between higher network layers and the raw input.
    Assumptions: no target weights, no recurrencies.
    Challenges: How to allow an user to specify this from client process. I do not think it makes sense. Should there be a separate settings file for custom network settings? One line per layer, number of features and stride.
    Unsupervised aux tasks will backprop onto this, higher level network will backprop onto this, dqn two nets will backprop onto this. Threrefore inserting errors and actual backprop should be separate functions. Errors are additive.
   */
  Network* net;
  Optimizer* opt;
  mutable vector<vector<Activation*>*> series;

  vector<Real> prepare_features(const Uint len, const Uint thrID) const
  {
    net->prepForBackProp(series_1[thrID], len);
  }
  vector<Real> input_features(const vector<Real>& obs, const Uint thrID, const Uint samp) const
  {
    vector<Activation*>& series = *(series_1[thrID]);
    net->predict(obs, ret, series[samp]);
  }
  void backprop(const vector<Real>& error, const Uint thrID, const Uint samp) const
  {
    vector<Activation*>& series = *(series_1[thrID]);
    net->backProp(error, series, net->Vgrad[thrID]);
  }
}
#endif

class Learner_utils: public Learner
{
protected:
  mutable vector<long double> cntGrad;
  mutable vector<vector<long double>> avgGrad, stdGrad;
  trainData stats;
  vector<trainData*> Vstats;
  mutable vector<Activation*> currAct, prevAct;
  mutable vector<vector<Activation*>*> series_1, series_2;

public:
  Learner_utils(MPI_Comm mcom,Environment*const _e, Settings&sett, Uint ngrads)
  : Learner(mcom, _e, sett), cntGrad(nThreads+1,0),
  avgGrad(nThreads+1,vector<long double>(ngrads,0)),
  stdGrad(nThreads+1,vector<long double>(ngrads,0))
  {
    stdGrad[0] = vector<long double>(ngrads,1000);
    assert(avgGrad.size()==nThreads+1 && cntGrad.size()==nThreads+1);
    for (Uint i=0; i<nThreads; i++) Vstats.push_back(new trainData());
  }
  virtual ~Learner_utils()
  {
    for (auto & trash : Vstats) _dispose_object(trash);
    for(Uint i=0; i<series_1.size(); i++) {
      net->deallocateUnrolledActivations(series_1[i]);
      delete series_1[i];
    }
    for(Uint i=0; i<series_2.size(); i++) {
      net->deallocateUnrolledActivations(series_2[i]);
      delete series_2[i];
    }
  }

  void dumpPolicy() override;

  void stackAndUpdateNNWeights() override;

  void updateTargetNetwork() override;

  virtual void buildNetwork(const vector<Uint> nouts, Settings & settings,
      vector<Uint> addedInputs = vector<Uint>() );

  void finalize_network(Builder& build)
  {
    opt = build.finalSimple();
    assert(nOutputs == net->getnOutputs() && nInputs == net->getnInputs());
    for (Uint i = 0; i < nThreads; i++) {
      series_1.push_back(new vector<Activation*>());
      series_2.push_back(new vector<Activation*>());
    }
    net->prepForFwdProp(&currAct, nThreads);
    net->prepForFwdProp(&prevAct, nThreads);
  }

  vector<Real> output_stochastic_policy(const int agentId, const Agent& agent) const;
  vector<Real> output_value_iteration(  const int agentId, const Agent& agent) const;

  inline void dumpStats(trainData*const _st, const Real&Q, const Real&err) const
  {
    _st->MSE += err*err;
    _st->avgQ += Q;
    _st->stdQ += Q*Q;
    _st->minQ = std::min(_st->minQ,static_cast<long double>(Q));
    _st->maxQ = std::max(_st->maxQ,static_cast<long double>(Q));
    _st->dCnt++;
  }

  virtual void processStats() override;
  virtual void processGrads();

  inline int clip_gradient(vector<Real>& grad, const vector<long double>& std,
    const Uint seq, const Uint samp) const
  {
    int ret = 0;
    for (Uint i=0; i<grad.size(); i++) {
      #ifdef importanceSampling
        assert(data->Set[seq]->tuples[samp]->weight>0);
        grad[i] *= data->Set[seq]->tuples[samp]->weight;
      #endif
      #ifdef ACER_GRAD_CUT
        if(grad[i] >  ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
        {
          //printf("Cut! %u was:%f is:%LG\n", i, grad[i], ACER_GRAD_CUT*std[i]);
          grad[i] =  ACER_GRAD_CUT*std[i];
          ret = 1;
        }
        else
        if(grad[i] < -ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
        {
          //printf("Cut! %u was:%f is:%LG\n",i, grad[i],-ACER_GRAD_CUT*std[i]);
          grad[i] = -ACER_GRAD_CUT*std[i];
          ret = 1;
        }
        //else printf("Not cut\n");
      #endif
    }
    return ret;
  }

  void statsVector(vector<vector<long double>>& sum, vector<vector<long double>>& sqr, vector<long double>& cnt);

  void dumpNetworkInfo(const int agentId) const;
};
