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
mutable vector<Real> cntGrad;
	mutable vector<vector<Real>> avgGrad, stdGrad;
	trainData stats;
	vector<trainData*> Vstats;

public:
	Learner_utils(MPI_Comm mcom,Environment*const _e, Settings&sett, Uint ngrads)
	: Learner(mcom, _e, sett), cntGrad(nThreads+1,0),
	avgGrad(nThreads+1,vector<Real>(ngrads,0)),
	stdGrad(nThreads+1,vector<Real>(ngrads,1000))
	{
		assert(avgGrad.size()==nThreads+1 && cntGrad.size()==nThreads+1);
		for (Uint i=0; i<nThreads; i++) Vstats.push_back(new trainData());
	}
	virtual ~Learner_utils()
	{
		for (auto & trash : Vstats) _dispose_object(trash);
	}

	void dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
			const vector<Uint>& nbins) override;

	void stackAndUpdateNNWeights(const Uint nAddedGradients) override;

	void updateNNWeights(const Uint nAddedGradients) override;

	void updateTargetNetwork() override;

	void buildNetwork(Network*& _net , Optimizer*& _opt,
			const vector<Uint> nouts, Settings & settings,
			vector<Real> weightInitFactors = vector<Real>(),
			const vector<Uint> addedInputs = vector<Uint>());

	vector<Real> output_stochastic_policy(const int agentId, State& s, Action& a,
			State& sOld, Action& aOld, const int info, Real r);

	vector<Real> output_value_iteration(const int agentId, State& s, Action& a,
		State& sOld, Action& aOld, const int info, Real r);

	inline void dumpStats(trainData*const _st, const Real&Q, const Real&err) const
	{
		_st->MSE += err*err;
		//_st->relE += fabs(err)/(max_Q-min_Q);
		_st->avgQ += Q;
		_st->stdQ += Q*Q;
		_st->minQ = std::min(_st->minQ,static_cast<long double>(Q));
		_st->maxQ = std::max(_st->maxQ,static_cast<long double>(Q));
		_st->dumpCount++;
	}

	virtual void processStats(const Real avgTime) override;
	virtual void processGrads();
	inline vector<Real> pickState(const vector<vector<Real>>& bins, Uint k)
  {
		vector<Real> state(bins.size());
		for (Uint i=0; i<bins.size(); i++) {
			state[i] = bins[i][ k % bins[i].size() ];
			k /= bins[i].size();
		}
		return state;
  }

	void dumpNetworkInfo(const int agentId);
};
