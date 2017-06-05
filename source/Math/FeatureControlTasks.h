/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Quadratic_advantage.h"
#include "../Learners/Learner.h"

class ContinuousFeatureControl
{
	const Uint outIndex, nA, nL;
	const Network*const net;
	const Transitions*const data;
	const vector<Uint> net_outputs = {1, nL, nA};
	const vector<Uint> net_indices = {outIndex, outIndex+1, outIndex+1+nL};
	const Uint nOutputs = 1+nL+nA;

	inline Quadratic_advantage prepare_advantage(const vector<Real>& out) const
	{
		return Quadratic_advantage(net_indices[1], net_indices[2], nA, nL, out);
	}
	static inline Uint compute_nL(const Uint NA)
	{
		return (NA*NA + NA)/2;
	}
	inline Real computeReward(const Tuple*const t0, const Tuple*const t1) const
	{
		Real ret = 0;
		const vector<Real> sold = data->standardize(t0->s);
		const vector<Real> snew = data->standardize(t1->s);
		for (Uint j=0; j<sold.size(); j++)
			ret += std::pow(sold[j]-snew[j], 2);
		return ret/sold.size();
	}

public:
	ContinuousFeatureControl(Uint oi, Uint na, const Network*const _net,
		const Transitions*const d) : outIndex(oi), nA(na), nL(compute_nL(na)),
		net(_net), data(d) { }

	inline void Train(
		const Activation*const nPrev, const Activation*const nNext,
		const vector<Real>&act, const Tuple*const t0, const Tuple*const t1,
		const Real gamma, const bool terminal, vector<Real>& grad) const
	{
		const vector<Real> outPrev = net->getOutputs(nPrev);
		const vector<Real> outNext = terminal?vector<Real>():net->getOutputs(nNext);
		const Real reward = computeReward(t0, t1);
		const Quadratic_advantage adv_sold = prepare_advantage(outPrev);
		const Real Vsold = outPrev[net_indices[0]];
		const Real Qsold = Vsold + adv_sold.computeAdvantage(act);
		const Real value = (terminal)? reward :reward+gamma*outNext[net_indices[0]];
		const Real error = value - Qsold;
		grad[net_indices[0]] = error;
		adv_sold.grad(act, error, grad);
	}
};

class DiscreteFeatureControl
{
	const Uint outIndex, nA;
	const Network*const net;
	const Transitions*const data;
	const vector<Uint> net_outputs = {1, nA};
	const vector<Uint> net_indices = {outIndex, outIndex+1};
	const Uint nOutputs = 1+nA;

	inline Real computeReward(const Tuple*const t0, const Tuple*const t1) const
	{
		Real ret = 0;
		const vector<Real> sold = data->standardize(t0->s);
		const vector<Real> snew = data->standardize(t1->s);
		for (Uint j=0; j<sold.size(); j++)
			ret += std::pow(sold[j]-snew[j], 2);
		return ret/sold.size();
	}

public:
	DiscreteFeatureControl(Uint oi, Uint na, const Network*const _net,
		const Transitions*const d) : outIndex(oi), nA(na), net(_net), data(d) {}

	inline void Train(
		const Activation*const nPrev, const Activation*const nNext,
		const Uint act, const Tuple*const t0, const Tuple*const t1,
		const Real gamma, const bool terminal, vector<Real>& grad) const
	{
		assert(act<nA);
		const vector<Real> outPrev = net->getOutputs(nPrev);
		const vector<Real> outNext = terminal?vector<Real>():net->getOutputs(nNext);
		const Real reward = computeReward(t0, t1);
		const Real Qsold = outPrev[net_indices[0]] + outPrev[net_indices[1] + act];
		const Real value = (terminal)? reward :reward+gamma*outNext[net_indices[0]];
		const Real error = value - Qsold;
		grad[net_indices[0]] = grad[net_indices[1]+act] = error;
	}
};
