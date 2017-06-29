/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Utils.h"

struct Discrete_policy
{
	const Uint start_prob, start_vals, nA;
	const vector<Real>& netOutputs;
	const vector<Real> unnorm, vals;
	const Real normalization;
	const vector<Real> probs;

	Discrete_policy(Uint _startP, Uint _startV, Uint _nA,const vector<Real>&out) :
			start_prob(_startP), start_vals(_startV), nA(_nA), netOutputs(out),
			unnorm(extract_unnorm()), vals(extract_values()),
			normalization(compute_norm()), probs(extract_probabilities())
			{
				//printf("Discrete_policy: %u %u %u %lu %lu %lu %lu\n",
				//start_prob,start_vals,nA,netOutputs.size(),
				//unnorm.size(),vals.size(),probs.size());
			}

private:
	inline vector<Real> extract_values() const
	{
		assert(netOutputs.size()>=start_vals+nA);
		return vector<Real>(&(netOutputs[start_vals]),&(netOutputs[start_vals])+nA);
	}

	inline vector<Real> extract_unnorm() const
	{
		assert(netOutputs.size()>=start_prob+nA);
		vector<Real> ret(nA);
		for (Uint j=0; j<nA; j++) ret[j] = prob_func(netOutputs[start_prob+j]);
		return ret;
	}

	inline Real compute_norm() const
	{
		assert(unnorm.size()==nA);
		Real ret = 0;
		for (Uint j=0; j<nA; j++) ret += unnorm[j];
		return ret;
	}

	inline vector<Real> extract_probabilities() const
	{
		assert(unnorm.size()==nA);
		vector<Real> ret(nA);
		for (Uint j=0; j<nA; j++) ret[j] = unnorm[j]/normalization;
		return ret;
	}

	inline Real expectedAdvantage() const
	{
		Real ret = 0;
		for (Uint j=0; j<nA; j++) ret += probs[j]*vals[j];
		return ret;
	}

	static inline Real prob_func(const Real val)
	{
		//return safeExp(val) + ACER_MIN_PROB;
		return 0.5*(val + std::sqrt(val*val+1)) + ACER_MIN_PROB;
	}

	static inline Real prob_diff(const Real val)
	{
		//return safeExp(val);
		return 0.5*(1.+val/std::sqrt(val*val+1));
	}

public:
	void test(const Uint act, const Discrete_policy*const pol_hat);

	inline Real probability(const Uint act) const
	{
		assert(act<=nA);
		return probs[act];
	}
	inline Real evalLogProbability(const Uint act) const
	{
		assert(act<=nA && probs.size()==nA);
		return std::log(probs[act]);
	}

	inline Uint sample(mt19937*const gen) const
	{
		std::discrete_distribution<Uint> dist(probs.begin(), probs.end());
		return dist(*gen);
	}

	inline Real advantageVariance() const
	{
		const Real base = expectedAdvantage();
		Real ret = 0;
		for (Uint j=0; j<nA; j++) ret += probs[j]*(vals[j]-base)*(vals[j]-base);
		return ret;
	}

	inline Real kl_divergence(const Discrete_policy*const pol_hat) const
	{
		Real ret = 0;
		for (Uint i=0; i<nA; i++)
			ret += pol_hat->probs[i]*(std::log(pol_hat->probs[i]/probs[i]));
		return ret;
	}

	inline vector<Real> div_kl_grad(const Discrete_policy*const pol_hat) const
	{
		vector<Real> ret(nA, 0);
		//for (Uint i=0; i<nA; i++) ret[i] = (probs[i]-pol_hat->probs[i]);
		for (Uint j=0; j<nA; j++) {
			const Real pDiff = prob_diff(netOutputs[start_prob+j]);
			ret[j] = pDiff*(1./normalization - pol_hat->probs[j]/unnorm[j]);
		}

		return ret;
	}

	inline vector<Real> policy_grad(const Uint act, const Real factor) const
	{
		vector<Real> ret(nA);
		//for (Uint i=0; i<nA; i++) ret[i] = factor*(((i==act) ? 1 : 0) -probs[i]);
		for (Uint i=0; i<nA; i++)
			ret[i] = -factor*prob_diff(netOutputs[start_prob+i])/normalization;
		ret[act] += factor*prob_diff(netOutputs[start_prob+act])/unnorm[act];
		return ret;
	}

	inline vector<Real> control_grad(const Real eta) const
	{
		vector<Real> ret(nA, 0);
		//for (Uint j=0; j<nA; j++) for (Uint i=0; i<nA; i++)
		//	ret[i]+=eta*((i==j)?probs[i]*(1-probs[i]):-probs[i]*probs[j])*vals[j];
		for (Uint j=0; j<nA; j++) {
			const Real pDiff = prob_diff(netOutputs[start_prob+j]);
			ret[j] = eta*pDiff*(vals[j]-expectedAdvantage())/normalization;
		}
		return ret;
	}

	inline Real computeAdvantage(const Uint act) const
	{
		return vals[act]-expectedAdvantage(); //subtract expectation from advantage of action
	}

	inline void values_grad(const Uint act, const Real Qer, vector<Real>&netGradient) const
	{
		for (Uint j=0; j<nA; j++)
			netGradient[start_vals+j] = Qer*((j==act ? 1 : 0) - probs[j]);
	}

	inline void finalize_grad(const vector<Real>&grad, vector<Real>&netGradient) const
	{
		assert(netGradient.size()>=start_prob+nA && grad.size() == nA);
		for (Uint j=0; j<nA; j++) netGradient[start_prob+j] = grad[j];
	}
	inline vector<Real> getVals() const
	{
		return vals;
	}
	inline vector<Real> getProbs() const
	{
		return probs;
	}
};
/*
 inline Real diagTerm(const vector<Real>& S, const vector<Real>& mu,
			const vector<Real>& a) const
	{
		assert(S.size() == nA);
		assert(a.size() == nA);
		assert(mu.size() == nA);
		Real Q = 0;
		for (Uint j=0; j<nA; j++) Q += S[j]*std::pow(mu[j]-a[j],2);
		return Q;
	}
 */
