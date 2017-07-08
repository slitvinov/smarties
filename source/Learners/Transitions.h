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
	vector<Real> s, a, mu;
	Real r = 0, SquaredError = 0;
	#ifdef importanceSampling
		Real weight = 0;
	#endif
};

struct Sequence
{
	vector<Tuple*> tuples;
	bool ended = false;
	Real MSE = 0;

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
	const bool bNormalize, bTrain, bWriteToFile, bSampleSeq;
	const Uint maxTotSeqNum, maxSeqLen, minSeqLen, nAppended, batchSize;
	const int learn_rank, learn_size;
	const string path;
	const StateInfo sI;
	const ActionInfo aI;
	std::vector<std::mt19937>& generators;

	Uint iOldestSaved = 0;
	vector<Real> std, mean, invstd;
	vector<Uint> curr_transition_id, inds;
	discrete_distribution<Uint> * dist = nullptr;

	int add(const int agentId, const int info, const State & sOld,
			const Action & a, const vector<Real> mu, const State& s, Real r);

	void push_back(const int & agentId);
	void clear(const int & agentId);
	void synchronize();
	void sortSequences();

public:
	//bool bRecurrent;
	Uint anneal=0, nBroken=0, nTransitions=0, nSequences=0, old_ndata=0, nSeenSequences=0;
	Gen * gen;
	vector<Sequence*> Set, Tmp, Buffered;

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

	vector<Real> standardize(const vector<Real>& state, const Real noise=-1, const Uint thrID=0) const;
#ifdef importanceSampling
	void updateP();
#endif
	void save(std::string fname);
	void restart(std::string fname);
	Uint updateSamples(const Real annealFac);
	Uint sample(const int thrID = 0);
	Uint restartSamples(const Uint polDim = 0);
	void saveSamples();

	int passData(const int agentId, const int info, const State& sOld,
		const Action&a, const State&s, const Real r, const vector<Real>mu = vector<Real>());

	inline bool requestUpdateSamples() const
	{
		return inds.size()<batchSize;
	}
};
