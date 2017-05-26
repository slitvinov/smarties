/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../Agent.h"
#include "../Environments/Environment.h"
#include "Transitions.h"
#include "../Network/Builder.h"
#include "../Network/Network.h"
#include "../Network/Optimizer.h"

class Master;

#include "../Scheduler.h"
#include <list>

using namespace std;

struct trainData
{
    trainData() : weight(1), MSE(0), avgQ(0), minQ(1e5), maxQ(-1e5), relE(0), dumpCount(0), epochCount(0) {}
    Real weight, MSE, avgQ, minQ, maxQ, relE;
    Uint dumpCount, epochCount;
};

class Learner
{
protected:
    const MPI_Comm mastersComm;
    Environment * const env;
    const Uint nAgents, batchSize, tgtUpdateDelay, nThreads, nInputs, nOutputs, nAppended;
    bool bRecurrent;
    const bool bTrain;
    const Real tgtUpdateAlpha, gamma, greedyEps, epsAnneal;
    Real dataUsage = 0;
    Uint cntUpdateDelay = 0, taskCounter;
    unsigned long mastersNiter_b4PolUpdates = 0;
    ActionInfo aInfo;
    StateInfo  sInfo;
    mt19937* const gen;  //only ok if only thread 0 accesses
    Profiler* profiler;
    Network* net;
    Optimizer* opt;
    Transitions* data;
    trainData stats;
    vector<trainData*> Vstats;
	//mutable vector<Real> meanGain1;
	//mutable vector<Real> meanGain2;
	//mutable vector<Real>  stdGain1;
	//mutable vector<Real>  stdGain2;
    virtual void Train_BPTT(const Uint seq, const Uint thrID=0) const = 0;
    virtual void Train(const Uint seq, const Uint samp, const Uint thrID=0) const = 0;
    Uint sampleSequences(vector<Uint>& sequences);
    Uint sampleTransitions(vector<Uint>& sequences, vector<Uint>& transitions);
    void dumpStats(const Real& Q, const Real& err, const vector<Real>& Qs);
    void dumpStats(trainData* const _stats, const Real& Q, const Real& err, const vector<Real>& Qs) const;
    virtual void processStats(vector<trainData*> _stats, const Real avgTime);
    virtual void updateTargetNetwork();
    virtual void stackAndUpdateNNWeights(const Uint nAddedGradients);
    virtual void updateNNWeights(const Uint nAddedGradients);
public:
    Learner(MPI_Comm mastersComm, Environment*const env, Settings & settings);

    virtual ~Learner()
    {
        _dispose_object(profiler);
        _dispose_object(net);
        _dispose_object(opt);
        _dispose_object(data);
        for (auto & trash : Vstats) _dispose_object(trash);
    }
    inline unsigned nData()
	{
		return data->nTransitions;
	}
    inline Real annealingFactor() const
    {
      //number that goes from 1 to 0 with optimizer's steps
      assert(epsAnneal>1.);
    	if(opt->nepoch >= epsAnneal) return 0;
    	else return 1 - opt->nepoch/epsAnneal;
    }

    inline Real annealedGamma() const
    {
      if (opt->nepoch>epsAnneal) return gamma;
      const Real anneal = Real(opt->nepoch)/epsAnneal;
      //const Real fac = 1 + anneal*(1./(1-gamma) -1);
      //return 1 - 1./fac;
      return anneal*gamma;
    }

    inline Real sequenceR(const Uint t0, const Uint seq) const
    {
      Real R = 0, G = 1;
      assert(t0+1 < data->Set[seq]->tuples.size());
      for(Uint i=t0+1; i<data->Set[seq]->tuples.size(); i++) {
        R += G*data->Set[seq]->tuples[i]->r;
        G *= gamma;
      }
      return R;
    }

    inline vector<Real> pickState(const vector<vector<Real>>& bins, Uint k)
    {
    	vector<Real> state(bins.size());
    	for (Uint i=0; i<bins.size(); i++) {
    		state[i] = bins[i][ k % bins[i].size() ];
    		k /= bins[i].size();
    	}
    	return state;
    }

    virtual void select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r) = 0;
    void clearFailedSim(const int agentOne, const int agentEnd);
    void pushBackEndedSim(const int agentOne, const int agentEnd);
    virtual void dumpPolicy(const vector<Real> lower, const vector<Real>& upper, const vector<Uint>& nbins);
    bool checkBatch(unsigned long mastersNiter);
    void TrainBatch();
    void TrainTasking(Master* const master);
    void save(string name);
    void restart(string fname);
    void buildNetwork(Network*& _net , Optimizer*& _opt, const vector<Uint> nouts,
      Settings & settings, const vector<Uint> addedInputs = vector<Uint>(0));
};
