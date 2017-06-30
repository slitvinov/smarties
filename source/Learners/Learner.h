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
	trainData() : weight(1), MSE(0), avgQ(0), stdQ(0), minQ(1e5), maxQ(-1e5), relE(0), dumpCount(0), epochCount(0) {}
	long double weight, MSE, avgQ, stdQ, minQ, maxQ, relE;
	Uint dumpCount, epochCount;
};

class Learner
{
protected:
	const MPI_Comm mastersComm;
	Environment * const env;
	const Uint tgtUpdateDelay, nAgents, batchSize, nThreads, nAppended, maxTotSeqNum, totNumSteps;
	Uint nInputs, nOutputs;
	const bool bRecurrent, bTrain, bSampleSequences;
	const Real tgtUpdateAlpha, gamma, greedyEps, epsAnneal, obsPerStep;
	unsigned long batchUsage = 0, dataUsage = 0;
	Uint cntUpdateDelay = 0, taskCounter, epochCounter = 0, policyVecDim = 0;
	unsigned long mastersNiter_b4PolUpdates = 0;
	ActionInfo aInfo;
	StateInfo  sInfo;
	mt19937* const gen;  //only ok if only thread 0 accesses
	Profiler* profiler;
	Network* net;
	Optimizer* opt;
	Transitions* data;

	virtual void Train_BPTT(const Uint seq, const Uint thrID=0) const = 0;
	virtual void Train(const Uint seq, const Uint samp, const Uint thrID=0) const = 0;
	virtual void processStats(const Real avgTime) = 0;
	virtual void stackAndUpdateNNWeights(const Uint nAddedGradients) = 0;
	virtual void updateTargetNetwork() = 0;

	Uint sampleSequences(vector<Uint>& sequences);
	Uint sampleTransitions(vector<Uint>& sequences, vector<Uint>& transitions);

public:
	Learner(MPI_Comm mastersComm, Environment*const env, Settings & settings);

	virtual ~Learner()
	{
		_dispose_object(profiler);
		_dispose_object(net);
		_dispose_object(opt);
		_dispose_object(data);
	}

	inline unsigned nData()
	{
		return data->nTransitions;
	}

	inline Real annealingFactor() const
	{
		//number that goes from 1 to 0 with optimizer's steps
		assert(epsAnneal>1.);
		if(opt->nepoch >= epsAnneal || !bTrain) return 0;
		else return 1 - opt->nepoch/epsAnneal;
	}
	/*
	inline Real annealedGamma() const
	{
		assert(epsAnneal>1. && bTrain && gamma>0.5);
		if (opt->nepoch > epsAnneal) return gamma;
		const Real anneal = 0.5 + 0.5*opt->nepoch/epsAnneal;
		return anneal*gamma;
	}
	/*/
	inline Real annealedGamma() const
	{
		assert(epsAnneal>1. && bTrain);
		if (opt->nepoch > epsAnneal) return gamma;
		const Real anneal = opt->nepoch/epsAnneal;
		const Real fac = 10 + anneal*(1./(1-gamma) -10);
		return 1 - 1./fac;
	}
	//*/

	virtual void select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r) = 0;
	void clearFailedSim(const int agentOne, const int agentEnd);
	void pushBackEndedSim(const int agentOne, const int agentEnd);
	virtual void dumpPolicy() = 0;
	bool checkBatch(unsigned long mastersNiter);
	//void TrainBatch();
	void run(Master* const master);
	void save(string name);
	void restart(string fname);
	void buildNetwork(Network*& _net , Optimizer*& _opt, const vector<Uint> nouts,
			Settings & settings, const vector<Uint> addedInputs = vector<Uint>(0));
};
