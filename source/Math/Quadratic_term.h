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

struct Quadratic_term
{
	const Uint start_matrix, start_mean, nA, nL;
	const vector<Real>& netOutputs;
	const vector<Real> L, mean, matrix;

	Quadratic_term(Uint _startMat, Uint _startMean, Uint _nA, Uint _nL,
			const vector<Real>& out, const vector<Real>_m=vector<Real>()) :
			start_matrix(_startMat), start_mean(_startMean), nA(_nA), nL(_nL),
			netOutputs(out), L(extract_L()), mean(extract_mean(_m)),
			matrix(extract_matrix())
			{
				//printf("Quadratic_term: %u %u %u %u %lu %lu %lu %lu\n", start_matrix,start_mean,nA,nL,
				//netOutputs.size(), L.size(), mean.size(), matrix.size());
				assert(L.size()==nA*nA && mean.size()==nA && matrix.size()==nA*nA);
				assert(netOutputs.size()>=start_matrix+nL && netOutputs.size()>=start_mean+nA);
			}

protected:
	inline Real quadMatMul(const vector<Real>& act, const vector<Real>& mat) const
	{
		assert(act.size() == nA && mat.size() == nA*nA);
		Real ret = 0;
		for (Uint j=0; j<nA; j++)
		for (Uint i=0; i<nA; i++)
			ret += (act[i]-mean[i])*mat[nA*j+i]*(act[j]-mean[j]);
		return ret;
	}

	inline Real quadraticTerm(const vector<Real>& act) const
	{
		return quadMatMul(act, matrix);
	}

	inline vector<Real> extract_L() const
	{
		assert(netOutputs.size()>=start_matrix+nL);
		vector<Real> ret(nA*nA);
		Uint kL = start_matrix;
		for (Uint j=0; j<nA; j++)
		for (Uint i=0; i<nA; i++)
			if (i<j) ret[nA*j + i] = offdiag_func(netOutputs[kL++]);
			else if (i==j) ret[nA*j + i] = diag_func(netOutputs[kL++]);
		assert(kL==start_matrix+nL);
		return ret;
	}

	inline vector<Real> extract_mean(const vector<Real>& tmp) const
	{
		if(tmp.size() == nA) return tmp;
		assert(start_mean!=0 && netOutputs.size()>=start_mean+nA);
		return vector<Real>(&(netOutputs[start_mean]),&(netOutputs[start_mean])+nA);
	}

	inline vector<Real> extract_matrix() const //fill positive definite matrix P == L * L'
	{
		assert(L.size() == nA*nA);
		vector<Real> ret(nA*nA,0);
		for (Uint j=0; j<nA; j++)
		for (Uint i=0; i<nA; i++)
		for (Uint k=0; k<nA; k++) {
			const Uint k1 = nA*j + k;
			const Uint k2 = nA*i + k;
			ret[nA*j + i] += L[k1] * L[k2];
		}
		return ret;
	}

	inline void grad_matrix(const vector<Real>&dErrdP, vector<Real>&netGradient) const
	{
		assert(netGradient.size() >= start_matrix+nL);
		for (Uint il=0; il<nL; il++)
		{
			Uint kL = 0;
			vector<Real> _dLdl(nA*nA, 0);
			for (Uint j=0; j<nA; j++)
			for (Uint i=0; i<nA; i++)
				if(i<=j) if(kL++==il) _dLdl[nA*j+i]=1;
			assert(kL==nL);

			netGradient[start_matrix+il] = 0;
			//_dPdl = dLdl' * L + L' * dLdl
			for (Uint j=0; j<nA; j++)
			for (Uint i=0; i<nA; i++)
			{
				Real dPijdl = 0;
				for (Uint k=0; k<nA; k++)
				{
					const Uint k1 = nA*j + k;
					const Uint k2 = nA*i + k;
					dPijdl += _dLdl[k1]*L[k2] + L[k1]*_dLdl[k2];
				}
				netGradient[start_matrix+il] += dPijdl*dErrdP[nA*j+i];
			}
		}
		{
			Uint kl = start_matrix;
			for (Uint j=0; j<nA; j++)
			for (Uint i=0; i<nA; i++) {
				if (i==j) netGradient[kl] *= diag_func_diff(netOutputs[kl]);
				if (i<j)  netGradient[kl] *= offdiag_func_diff(netOutputs[kl]);
				if (i<=j) kl++;
			}
			assert(kl==start_matrix+nL);
		}
	}

	static inline Real diag_func(const Real val)
	{
		//return std::exp(val) +ACER_TOL_DIAG;
		return 0.5*(val + std::sqrt(val*val+1)) +ACER_TOL_DIAG;
		//return sqrt(val + std::sqrt(val*val+1)) +ACER_TOL_DIAG;
	}
	static inline Real diag_func_diff(const Real val)
	{
		//return std::exp(val);
		return 0.5*(1 + val/std::sqrt(val*val+1));
		//const Real den = std::sqrt(val*val+1);
		//return 0.5*std::sqrt(den+val)/den;
	}
	static inline Real offdiag_func(const Real val)
	{
		//return val;
		return val/sqrt(1+std::fabs(val));
	}
	static inline Real offdiag_func_diff(Real val)
	{
		//return 1.;
		if(val<0) val = -val; //symmetric
		const Real denom = std::sqrt(val+1);
		return (.5*val+1)/(denom*denom*denom);
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
	inline Real quadraticNoise(const vector<Real>& P, const vector<Real>& var, const int thrID) const
	{
		vector<Real> q(nA,0);
		for (Uint j=0; j<nA; j++)
		{
			const Real scale = 0.1*std::sqrt(3)*std::sqrt(var[j]);
			std::uniform_real_distribution<Real> distn(-scale, scale);
			q[j] = distn(generators[thrID]);
		}

		Real Q = 0;
		for (Uint j=0; j<nA; j++) for (Uint i=0; i<nA; i++)
			Q += P[nA*j+i]*q[i]*q[j];

		return Q;
	}
 */
