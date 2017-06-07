/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_utils.h"
#include "../Math/FeatureControlTasks.h"
#include "../Math/Quadratic_advantage.h"

class RACER : public Learner_utils
{
	const Real truncation, delta;
	const Uint nA, nL;
	std::vector<std::mt19937>& generators;
	#if defined ACER_RELAX // output V(s), P(s), pol(s), prec(s)
		vector<Uint> net_outputs = {1, nL, nA, nA};
		vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA};
	#elif defined ACER_SAFE // output V(s), P(s), pol(s), mu(s)
		vector<Uint> net_outputs = {1, nL, nA, nA};
		vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA};
	#else // output V(s), P(s), pol(s), prec(s), mu(s)
		vector<Uint> net_outputs = {1, nL, nA, nA, nA};
		vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA, 1+nL+2*nA};
	#endif
	#ifdef FEAT_CONTROL
	const ContinuousSignControl* task;
	#endif

	inline Gaussian_policy prepare_policy(const vector<Real>& out) const
	{
		#if defined ACER_SAFE
		return Gaussian_policy(net_indices[2], nA, out);
		#else
		return Gaussian_policy(net_indices[2], net_indices[3], nA, out);
		#endif
	}
	inline Quadratic_advantage prepare_advantage(const vector<Real>& out,
			const Gaussian_policy*const pol) const
	{
		#if defined ACER_RELAX
		return Quadratic_advantage(net_indices[1], nA, nL, out, pol);
		#else
		#if defined ACER_SAFE
		return Quadratic_advantage(net_indices[1],net_indices[3],nA,nL,out,pol);
		#else
		return Quadratic_advantage(net_indices[1],net_indices[4],nA,nL,out,pol);
		#endif
		#endif
	}

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;
	/*
		vector<Real> basicNetOut(Action& a, const vector<Real> mu, const vector<Real> var)
		{
			assert(mu.size()==nA);
			assert(var.size()==nA);
			vector<Real> beta(2*nA, 0);
			const Real anneal = annealingFactor();
			//const Real eps = max(anneal, greedyEps);
			if( positive(anneal) || bTrain ) {
				for(Uint i=0; i<nA; i++) {
					const Real policy_std = std::sqrt(var[i]);
					const Real anneal_std = greedyEps +(1-anneal)*policy_std;
					const Real annealed_mean = (1-anneal*anneal)*mu[i];
					std::normal_distribution<Real> dist_cur(annealed_mean, anneal_std);
					beta[i] = annealed_mean; //to save correct mu
					beta[nA+i] = 1./std::pow(anneal_std, 2); //to save correct mu
					a.vals[i] = dist_cur(*gen);
				}
			}
			else if ( positive(greedyEps) ) { //still want to sample policy.
				for(Uint i=0; i<nA; i++) {
					std::normal_distribution<Real> dist_cur(mu[i], std::sqrt(var[i]));
					a.vals[i] = dist_cur(*gen);
					beta[i] = mu[i]; //to save correct mu
					beta[nA+i] = 1./var[i]; //to save correct mu
				}
			}
			else {//load computed policy into a
				for(Uint i=0; i<nA; i++) {
					a.vals[i] = mu[i];
					beta[i] = mu[i]; //to save correct mu
					beta[nA+i] = 1./var[i]; //to save correct mu
				}
			}
			return beta;
		}
	*/
public:
	RACER(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s,Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;

	void test();

	static Uint getnOutputs(const Uint NA)
	{
#if defined ACER_RELAX
		// I output V(s), P(s), pol(s), prec(s) (and variate)
		return 1+compute_nL(NA)+NA+NA;
#elif defined ACER_SAFE
		// I output V(s), P(s), pol(s), mu(s) (and variate)
		return 1+compute_nL(NA)+NA+NA;
#else //full formulation
		// I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
		return 1+compute_nL(NA)+NA+NA+NA;
#endif
	}
	static inline Uint compute_nL(const Uint NA)
	{
		return (NA*NA + NA)/2;
	}
};
