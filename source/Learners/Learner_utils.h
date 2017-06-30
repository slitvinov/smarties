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

public:
	Learner_utils(MPI_Comm mcom,Environment*const _e, Settings&sett, Uint ngrads)
	: Learner(mcom, _e, sett), cntGrad(nThreads+1,0),
	avgGrad(nThreads+1,vector<long double>(ngrads,0)),
	stdGrad(nThreads+1,vector<long double>(ngrads,0))
	{
		stdGrad[0] = vector<long double>(ngrads,100);
		assert(avgGrad.size()==nThreads+1 && cntGrad.size()==nThreads+1);
		for (Uint i=0; i<nThreads; i++) Vstats.push_back(new trainData());
	}
	virtual ~Learner_utils()
	{
		for (auto & trash : Vstats) _dispose_object(trash);
	}

	void dumpPolicy() override;

	void stackAndUpdateNNWeights(const Uint nAddedGradients) override;

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

	void dumpNetworkInfo(const int agentId);
};
