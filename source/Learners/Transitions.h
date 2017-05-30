/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "../StateAction.h"
#include "../Settings.h"
#include "../Environments/Environment.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>

struct Tuple
{
	vector<Real> s;
	vector<Real> a;
	vector<Real> mu;
	Real r;

	Real SquaredError;
	Tuple(): r(0), SquaredError(0) {}
};

struct Sequence
{
	Sequence() : ended(false), MSE(0.) {}

	vector<Tuple*> tuples;
	bool ended;
	Real MSE;

	~Sequence()
	{
		for (auto & trash : tuples) _dispose_object( trash);
	}
};

struct Gen
{
	mt19937 * g;
	Gen(mt19937 * gen) : g(gen) { }
	size_t operator()(size_t n)
	{
		std::uniform_int_distribution<size_t> d(0, n ? n-1 : 0);
		return d(*g);
	}
};

class Transitions
{
protected:
	const MPI_Comm mastersComm;
	Environment * const env;
	Uint iOldestSaved=0;
	const bool bSampleSeq, bWriteToFile, bNormalize, bTrain;
	const string path;
	vector<Real> std, mean, invstd;
	vector<Uint> curr_transition_id;
	discrete_distribution<Uint> * dist;
	int add(const int agentId, const int info, const State & sOld,
			const Action & a, const vector<Real>& mu, const State& s, Real r);
	int add(const int agentId, const int info, const State& sOld,
			const Action& a, const State& sNew, const Real reward);

	void push_back(const int & agentId);
	void clear(const int & agentId);
	void synchronize();

	inline bool needed_samples_mean(const bool something_changed)
	{
		//if env has specified scale and mean of state components, do not compute from data
		if(sI.mean.size()) return false;
		//user can still specify in settings if state should be standardized
		if(!bNormalize) return false;
		//else, if something changed in any of the Master ranks, then update is needed
		return syncBoolOr(something_changed);
	}

public:
	const Uint nAppended, batchSize, maxSeqLen, minSeqLen, maxTotSeqNum;
	bool bRecurrent;
	Uint anneal=0, nBroken=0, nTransitions=0, nSequences=0, old_ndata=0, nSeenSequences=0;
	const StateInfo sI;
	const ActionInfo aI;
	std::vector<std::mt19937>& generators;
	Gen * gen;
	vector<Sequence*> Set, Tmp, Buffered;
	vector<Uint> inds;

	Transitions(MPI_Comm comm, Environment*const env, Settings & settings);

	~Transitions()
	{
		_dispose_object(gen);
		_dispose_object(dist);
		for (auto & trash : Set) _dispose_object( trash);
		for (auto & trash : Tmp) _dispose_object( trash);
		for (auto & trash : Buffered) _dispose_object( trash);
	}
	void clearFailedSim(const int agentOne, const int agentEnd);
	void pushBackEndedSim(const int agentOne, const int agentEnd);
	void update_samples_mean(const Real alpha = 0.01);
	int syncBoolOr(int needed) const;
	vector<Real> standardize(const vector<Real>& state, const Real noise=-1, const Uint thrID=0) const;
#ifdef _Priority_
	void updateP();
#endif
	void save(std::string fname);
	void restart(std::string fname);
	void updateSamples(const Real alpha = 0.01);
	Uint sample();
	void restartSamples();
	void restartSamplesNew(const bool bContinuous);
	void saveSamples();

	int passData(const int agentId, const int info, const State& sOld,
			const Action & a, const vector<Real>& mu, const State & s, const Real r);
	int passData(const int agentId, const int info, const State& sOld,
			const Action & a, const State & sNew, const Real r);
};
