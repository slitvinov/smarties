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

class Learner_utils: public Learner
{
protected:
  mutable vector<long double> cntGrad;
  mutable vector<vector<long double>> avgGrad, stdGrad;
  trainData stats;
  vector<trainData*> Vstats;
  mutable vector<Activation*> currAct, prevAct;
  mutable vector<vector<Activation*>*> series_1, series_2;

  static vector<Uint> count_indices(const vector<Uint> outs)
  {
    vector<Uint> ret(outs.size(), 0); //index 0 is 0
    for(Uint i=1; i<outs.size(); i++) ret[i] = ret[i-1] + outs[i-1];
    return ret;
  }

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

  inline void clip_gradient(vector<Real>& grad, const vector<long double>& std,
    const Uint seq, const Uint samp) const
  {
    for (Uint i=0; i<grad.size(); i++) {
      #ifdef importanceSampling
        assert(data->Set[seq]->tuples[samp]->weight>0);
        grad[i] *= data->Set[seq]->tuples[samp]->weight;
      #endif
      #ifdef ACER_GRAD_CUT
        if(grad[i] >  ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
        {
          //printf("Cut! was:%f is:%LG\n",grad[i], ACER_GRAD_CUT*std[i]);
          grad[i] =  ACER_GRAD_CUT*std[i];
        }
        else
        if(grad[i] < -ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
        {
          //printf("Cut! was:%f is:%LG\n",grad[i],-ACER_GRAD_CUT*std[i]);
          grad[i] = -ACER_GRAD_CUT*std[i];
        }
        //else printf("Not cut\n");
      #endif
    }
  }

  void statsVector(vector<vector<long double>>& sum, vector<vector<long double>>& sqr, vector<long double>& cnt);

  void dumpNetworkInfo(const int agentId) const;
};
