/*
 *  Settings.h
 *  rl
 *
 *  Created by Guido Novati on 02.05.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "../Bund.h"

inline Real safeExp(const Real val)
{
    return std::exp( std::min(9., std::max(-32.,val) ) );
}

inline vector<Real> sum3Grads(const vector<Real>& f, const vector<Real>& g,
  const vector<Real>& h)
{
  assert(g.size() == f.size());
  assert(h.size() == f.size());
  vector<Real> ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = f[i]+g[i]+h[i];
  return ret;
}

inline vector<Real> sum2Grads(const vector<Real>& f, const vector<Real>& g)
{
  assert(g.size() == f.size());
  vector<Real> ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = f[i]+g[i];
  return ret;
}

inline vector<Real> trust_region_update(const vector<Real>& grad,
	const vector<Real>& trust, const Real delta)
{
	assert(grad.size() == trust.size());
	const Uint nA = grad.size();
	vector<Real> ret(nA);
	Real dot=0, norm=0;
	for (Uint j=0; j<nA; j++) {
		norm += trust[j] * trust[j];
		dot +=  trust[j] *  grad[j];
	}
	const Real proj = std::max((Real)0., (dot - delta)/norm);
	//#ifndef NDEBUG
	//if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
	//#endif
	for (Uint j=0; j<nA; j++) {
		ret[j] = grad[j]-proj*trust[j];
		if(ret[j]*grad[j] < 0) ret[j] = 0;
	}
	return ret;
}

inline Real clip(const Real val, const Real ub, const Real lb)
{
  assert(!isnan(val));
  assert(!isinf(val));
  assert(ub>lb);
  return std::max(std::min(val, ub), lb);
}

inline Uint maxInd(const vector<Real>& pol)
{
  Real Val = -1e9;
  Uint Nbest = 0;
  for (Uint i=0; i<pol.size(); ++i)
      if (pol[i]>Val) { Val = pol[i]; Nbest = i; }
  return Nbest;
}

inline Real minAbsValue(const Real v, const Real w)
{
  return std::fabs(v)<std::fabs(w) ? v : w;
}

inline void statsVector(vector<vector<Real>>& sum, vector<vector<Real>>& sqr,
  vector<Real>& cnt)
{
   assert(sum.size()>1);
  assert(sum.size() == cnt.size() && sqr.size() == cnt.size());

  for (Uint i=0; i<sum[0].size(); i++)
    sum[0][i] = sqr[0][i] = 0;
  cnt[0] = 0;

  for (Uint i=1; i<sum.size(); i++) {
    cnt[0] += cnt[i]; cnt[i] = 0;
    for (Uint j=0; j<sum[0].size(); j++)
    {
      sum[0][j] += sum[i][j]; sum[i][j] = 0;
      sqr[0][j] += sqr[i][j]; sqr[i][j] = 0;
    }
  }
  cnt[0] = std::max(2.2e-16, cnt[0]);
  for (Uint j=0; j<sum[0].size(); j++)
  {
    sqr[0][j] = std::sqrt((sqr[0][j]-sum[0][j]*sum[0][j]/cnt[0])/cnt[0]);
    sum[0][j] /= cnt[0];
  }
}

inline void statsGrad(vector<Real>& sum, vector<Real>& sqr, Real& cnt, vector<Real> grad)
{
  assert(sum.size() == grad.size() && sqr.size() == grad.size());
  cnt += 1;
  for (Uint i=0; i<grad.size(); i++) {
    sum[i] += grad[i];
    sqr[i] += grad[i]*grad[i];
  }
}

inline void clip_gradient(vector<Real>& grad, const vector<Real>& std)
{
	for (Uint i=0; i<grad.size(); i++)
	{
		if(grad[i] >  ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
			grad[i] =  ACER_GRAD_CUT*std[i];
		else
		if(grad[i] < -ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
			grad[i] = -ACER_GRAD_CUT*std[i];
	}
}

/*
inline void setVecMean(vector<Real>& vals)
{
   assert(vals.size()>1);
	Real mean = 0;
	for (Uint i=1; i<vals.size(); i++) //assume 0 is empty
		mean += vals[i];
	mean /= (Real)(vals.size()-1);
	for (Uint i=0; i<vals.size(); i++)
		vals[i] = mean;
}
*/
