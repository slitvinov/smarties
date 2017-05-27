/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Learner.h"

class DiscreteAlgorithm : public Learner
{
protected:
	const Real delta;
	const Uint nA;
	std::vector<std::mt19937>& generators;

public:
	DiscreteAlgorithm(MPI_Comm comm, Environment*const _env, Settings & settings,
	const Real _delta): Learner(comm,_env,settings), delta(_delta),
	nA(_env->aI.maxLabel), generators(settings.generators) {}

protected:

	inline vector<Real> extractValues(const vector<Real>& out) const
	{
		assert(out.size()==1+2*nA);
		return vector<Real>(&(out[1+nA]),&(out[1+nA])+nA);
	}

	inline vector<Real> extractUnnormPolicy(const vector<Real>& out) const
	{
		assert(out.size()==1+2*nA);
		vector<Real> ret(nA);
		for (Uint j=0; j<nA; j++) ret[j] = safeExp(out[1+j]);
		return ret;
	}

	inline vector<Real> extractPolicy(const vector<Real>& out) const
	{
		assert(out.size()==1+2*nA);
		const vector<Real> unpol = extractUnnormPolicy(out);
		assert(unpol.size()==nA);
		Real base = 0;
		for (Uint j=0; j<nA; j++)
			base += unpol[j];

		vector<Real> ret(nA);
		for (Uint j=0; j<nA; j++)
			ret[j] = unpol[j]/base;
		return ret;
	}

	void checkGradient()
	{

			vector<Real> hat(nOutputs), out(nOutputs);
			uniform_real_distribution<Real> out_dis(-.5,.5);
			uniform_real_distribution<Real> act_dis(0,1);
			for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
			for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
			const Uint act = nA*act_dis(*gen);

			vector<Real> mu = extractPolicy(out);
			vector<Real> mu_hat = extractPolicy(hat);
			vector<Real> val = extractValues(out);
			vector<Real> val_hat = extractValues(hat);

			vector<Real> polGrad = policyGradient(mu, act, 1);
			vector<Real> cntGrad = controlGradient(act, mu, val, 1);
			vector<Real> gradDivKL = gradDKL(mu, mu_hat);
			vector<Real> cgrad = criticGradient(act, mu, val, 1);

			for(Uint i = 0; i<nA; i++)
			{
				vector<Real> out_1 = out;
				vector<Real> out_2 = out;
				out_1[1+i] -= 0.0001;
				out_2[1+i] += 0.0001;
				vector<Real> mu_1 = extractPolicy(out_1);
				vector<Real> mu_2 = extractPolicy(out_2);
				const Real A_1 = computeAdvantage(act, mu_1, val);
				const Real A_2 = computeAdvantage(act, mu_2, val);
				const Real p1 = std::log(mu_1[act]);
			 	const Real p2 = std::log(mu_2[act]);
				const Real d1 = DivKL(mu_1, mu_hat);
				const Real d2 = DivKL(mu_2, mu_hat);
				printf("LogPol mu Gradient %d: finite differences %g analytic %g \n",
				i, (p2-p1)/0.0002, polGrad[i]);
				printf("Control mu Gradient %d: finite differences %g analytic %g \n",
				i, (A_2-A_1)/0.0002, -cntGrad[i]); //the function returns -grad
				printf("DivKL mu Gradient %d: finite differences %g analytic %g \n",
				i, (d2-d1)/0.0002, gradDivKL[i]);
			}

			for(Uint i = 0; i<nA; i++)
			{
					vector<Real> out_1 = out;
					vector<Real> out_2 = out;
					out_1[1+nA+i] -= 0.0001;
					out_2[1+nA+i] += 0.0001;
					vector<Real> val_1 = extractValues(out_1);
					vector<Real> val_2 = extractValues(out_2);
					const Real A_1 = computeAdvantage(act, mu, val_1);
					const Real A_2 = computeAdvantage(act, mu, val_2);
					printf("Value Gradient %d: finite differences %g analytic %g \n",
					i, (A_2-A_1)/0.0002, cgrad[1+i]);
			}
	}

	inline Uint samplePolicy(const vector<Real>& pol, const Uint thrID) const
	{
		assert(pol.size()==nA);
		std::discrete_distribution<Uint> dist(pol.begin(),pol.end());
		return dist(generators[thrID]);
	}

	inline Real safeExp(const Real val) const
	{
			return std::exp( std::min(9., std::max(-32.,val) ) );
	}

	inline Real expectedAdvantage(const vector<Real>& pol, const vector<Real>& val) const
	{
		Real ret = 0;
		for (Uint j=0; j<nA; j++) ret += pol[j]*val[j];
		return ret;
	}

	inline Real advantageVariance(const vector<Real>& pol, const vector<Real>& val) const
	{
		const Real base = expectedAdvantage(pol, val);
		Real ret = 0;
		for (Uint j=0; j<nA; j++) ret += pol[j]*(val[j]-base)*(val[j]-base);
		return ret;
	}

	inline Real DivKL(const vector<Real>& pol, const vector<Real>& pol_hat) const
	{
		Real ret = 0;
		for (Uint i=0; i<nA; i++)
			ret += pol_hat[i]*(std::log(pol_hat[i])-std::log(pol[i]));
		return ret;
 	}

	//warning: return grad in terms of outputs
	inline vector<Real> gradDKL(const vector<Real>& pol, const vector<Real>& pol_hat) const
	{
		vector<Real> ret(nA);
		for (Uint i=0; i<nA; i++) ret[i] = pol[i]-pol_hat[i];
		return ret;
	}

	inline vector<Real> gradAcerTrpo(const vector<Real>& DA, const vector<Real>& DKL) const
	{
		assert(DA.size() == nA);
		assert(DKL.size() == nA);
		vector<Real> gradAcer(nA);
		Real dot=0, norm=0;
		for (Uint j=0; j<nA; j++) {
			norm += DKL[j] * DKL[j];
			dot +=  DKL[j] * DA[j];
		}
		const Real proj = std::max((Real)0., (dot - delta)/norm);
		for (Uint j=0; j<nA; j++) gradAcer[j] = DA[j] - proj*DKL[j];
		return gradAcer;
	}

	inline vector<Real> policyGradient(const vector<Real>& pol, const Uint act, const Real factor) const
	{
		vector<Real> ret(nA);
		for (Uint i=0; i<nA; i++) ret[i] = factor*(((i==act) ? 1 : 0) - pol[i]);
		return ret;
	}

	inline vector<Real> controlGradient(const Uint act, const vector<Real>& pol, const vector<Real>& val, const Real eta) const
	{
		vector<Real> gradCC(nA, 0);
		for (Uint j=0; j<nA; j++)
		for (Uint i=0; i<nA; i++)
			gradCC[i] += eta * ((i==j) ? pol[i]*(1-pol[i]) : -pol[i]*pol[j]) * val[j];
		return gradCC;
	}

	inline Real computeAdvantage(const Uint act, const vector<Real>& pol, const vector<Real>& val) const
	{
		assert(pol.size() == nA);
		assert(val.size() == nA);
		return val[act]-expectedAdvantage(pol,val); //subtract expectation from advantage of action
	}

	inline vector<Real> sum3Grads(const vector<Real>& f, const vector<Real>& g,
		const vector<Real>& h) const
	{
		assert(f.size() == nA);
		assert(g.size() == nA);
		assert(h.size() == nA);
		vector<Real> ret(nA,0);
		for(Uint i=0; i<nA; i++) ret[i] = f[i]+g[i]+h[i];
		return ret;
	}

	inline vector<Real> sum2Grads(const vector<Real>& f, const vector<Real>& g) const
	{
		assert(f.size() == nA);
		assert(g.size() == nA);
		vector<Real> ret(nA,0);
		for(Uint i=0; i<nA; i++) ret[i] = f[i]+g[i];
		return ret;
	}

	inline vector<Real> criticGradient(const Uint act, const vector<Real>& pol, const vector<Real>& val, const Real Qer) const
	{
		assert(pol.size()==nA);
		assert(val.size()==nA);
		vector<Real> grad(1+nA, 0);
		grad[0] = Qer;
		for (Uint j=0; j<nA; j++)
			grad[j+1] = Qer*((j==act ? 1 : 0) - pol[j]);
		return grad;
	}

	inline Uint maxInd(const vector<Real>& pol) const
	{
		assert(pol.size()==nA);
		Real Val = -1;
		Uint Nbest = -1;
		for (Uint i=0; i<nA; ++i) {
				if (pol[i]>Val) {
					Val = pol[i];
					Nbest = i;
				}
		}
		return Nbest;
	}
};
