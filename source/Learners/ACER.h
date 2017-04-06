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

using namespace std;

class ACER : public Learner
{
    const int nA, nL;
    const Real delta = 1;
    std::vector<std::mt19937>& generators;

    void Train_BPTT(const int seq, const int thrID=0) const override;
    void Train(const int seq, const int samp, const int thrID=0) const override;
    //void dumpNetworkInfo(const int agentId);

public:
    ACER(MPI_Comm comm, Environment*const env, Settings & settings);
    void select(const int agentId, State& s,Action& a, State& sOld,
                Action& aOld, const int info, Real r) override;

private:

  inline Real evaluateProbability(const vector<Real>& a, const vector<Real>& out) const
  {
    Real p = 1;
    for(int i=0; i<nA; i++)
      p *= std::exp(-.5*out[1+nL+nA+i]*std::pow(a[i]-out[1+nL+i],2))*std::sqrt(0.5*out[1+nL+nA+i]/M_PI);

    return p;
  }

  inline vector<Real> preparePmatrix(const vector<Real>& out) const
  {
    vector<Real> _L(nA*nA, 0), _P(nA*nA, 0);
    assert(out.size() == 1+nL+2*nA);
    { //fill lower diag matrix L
        int kL = 1;
        for (int j=0; j<nA; j++)
          for (int i=0; i<nA; i++)
            if (i<=j) _L[nA*j + i] = out[kL++];
        assert(kL==1+nL);
    }
    //fill positive definite matrix P == L * L'
    for (int j=0; j<nA; j++)
        for (int i=0; i<nA; i++) {
            const int ind = nA*j + i;
            for (int k=0; k<nA; k++) {
                const int k1 = nA*j + k;
                const int k2 = nA*i + k;
                _P[ind] += _L[k1] * _L[k2];
            }
        }
    return _P;
  }

  inline Real computeAdvantage(const vector<Real>& P, const vector<Real>& pi, const vector<Real>& a) const
  {
    /*
    this function returns the advantage for a given action = -.5 * (a-pi)' P (a-pi)
      Assumptions:
        - if action space is bounded, we are already receiving a that is scaled to unbounded space
    */
    assert(P.size() == nA*nA);
    assert(pi.size() == nA);
    assert(a.size() == nA);
    Real Q = 0;
    for (int j=0; j<nA; j++)
    for (int i=0; i<nA; i++) { //A = L * L'
        const int ind = nA*j + i;
        Q += P[ind]*(a[i]-pi[i])*(a[j]-pi[j]);
    }
    return -0.5*Q;
  }

  inline vector<Real> gradDKL(const vector<Real>& out, const vector<Real>& hat) const
  {
    /*
    this function return the derivative wrt to network output of the KL divergence
     from the distribution whose statistics are given by the current network weights
     and the distribution which target weights (delayed update/exponential averaging)

    Div_KL between two multiv. Gaussians N_1 and N_2 of dim=M is
    0.5*( trace(inv(Sigma_2)*Sigma_1) + (m_2 - m_1)'*inv(Sigma_2)*(m_2 - m_1) - M + ln(det(Sigma_2)/det(Sigma_1))

    assumptions:
      - we deal with diagonal covariance matrices
      - network outputs the inverse of diagonal terms of the cov matrix

    therefore divKL assumes shape
    0.5*(\sum_i( Sigma_1_i*(Sigma_2_i)^-1 + (m_2_i - m_1_i)^2*(Sigma_2_i)^-1 -M +ln(Sigma_2_i) -ln(Sigma_1_i))
    */
    assert(out.size() == 1+nL+2*nA);
    assert(hat.size() == 1+nL+2*nA);
    const vector<Real> pi_cur(&out[1+nL],&out[1+nL]+nA), C_cur(&out[1+nL+nA],&out[1+nL+nA]+nA);
    const vector<Real> pi_hat(&hat[1+nL],&hat[1+nL]+nA), C_hat(&hat[1+nL+nA],&hat[1+nL+nA]+nA);
    vector<Real> ret(2*nA);
    for (int i=0; i<nA; i++) {
      ret[i]    = (pi_cur[i]-pi_hat[i])*C_cur[i];
    }
    for (int i=0; i<nA; i++) {
      //               v from trace    v from quadratic term   v from normalization
      ret[i+nA] = 0.5*(1/C_hat[i] +pow(pi_cur[i]-pi_hat[i],2) +C_cur[i]);
    }
    return ret;
  }

  inline vector<Real> gradAcer(const vector<Real>& out, const vector<Real>& a, const Real factor) const
  {
    /*
    this function returns the off policy corrected gradient
    g_marg = factor * grad_phi log(pi(a,s))
    assumptions:
      - we deal with diagonal covariance matrices
      - network outputs the inverse of diagonal terms of the cov matrix
      - factor contains rho_i * gain_i
    Therefore log of distrib becomes:
    sum_i( -.5*log(2*M_PI*Sigma_i) -.5*(a-pi)^2*Sigma_i^-1 )
    */
    const vector<Real> pi(&out[1+nL],&out[1+nL]+nA), C(&out[1+nL+nA],&out[1+nL+nA]+nA);
    vector<Real> ret(2*nA);
    for (int i=0; i<nA; i++) {
      ret[i]    = factor*(a[i]-pi[i])*C[i];
    }
    for (int i=0; i<nA; i++) {
      ret[i+nA] = factor*(-0.5*C[i] -0.5*pow(a[i]-pi[i],2));
    }
    return ret;
  }

  inline Real advantageExpectation(const vector<Real>& pi, const vector<Real>& C,
                                  const vector<Real>& P, const vector<Real>& act) const
  {
    assert(pi.size() == nA);
    assert(C.size() == nA);
    assert(P.size() == nA*nA);
    assert(act.size() == nA);
    const Real A_s_a = computeAdvantage(P, pi, act);
    /*
    computing expectation under policy of (a-pi)'*P*(a-pi) (P non diagonal)
    where policy is defined by mean pi and diagonal covariance matrix
    which is equal to trace[(-0.5*P) * Sigma] (since E(a-pi)=0)
    */
    Real expectation = 0;
    for(int i=0; i<nA; i++) expectation += -0.5*P[nA*i+i]/C[i];
    return A_s_a - expectation; //subtract expectation from advantage of action
  }

  inline Real computeQ(const vector<Real>& act, const vector<Real>& pol_out, const vector<Real>& val_out) const
  {
    const vector<Real> pi(&pol_out[1+nL], &pol_out[1+nL]+nA);
    const vector<Real> C(&pol_out[1+nL+nA], &pol_out[1+nL+nA]+nA);
    const vector<Real> P = preparePmatrix(val_out);
    return val_out[0] + advantageExpectation(pi, C, P, act);
  }

  inline vector<Real> computeGradient(const Real error, const vector<Real>& out,
    const vector<Real>& hat, const vector<Real>& act, const Real fac) const
  {
    assert(out.size() == 1+nL+2*nA);
    assert(hat.size() == 1+nL+2*nA);
    assert(act.size() == nA);
    //assert(P.size() == 2*nA);
    vector<Real> grad(1+nL+nA*2);
    grad[0] = error;

    {
        //these are used to compute Q, so only involved in value gradient
        const vector<Real> pi_hat(&hat[1+nL], &hat[1+nL]+nA);
        const vector<Real> C_hat(&hat[1+nL+nA], &hat[1+nL+nA]+nA);
        vector<Real> _L(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA);
        int kL = 1;
        for (int j=0; j<nA; j++) {
          _u[j] = act[j] - pi_hat[j];
          for (int i=0; i<nA; i++)
            if (i<=j)
              _L[nA*j + i] = out[kL++];
        }
        assert(kL==1+nL);

        for (int il=0; il<nL; il++) {
            int kD = 0;
            for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) {
                const int ind = nA*j + i;
                _dLdl[ind] = 0;
                if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
            }
            assert(kD==nL);

            for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) {
                const int ind = nA*j + i;
                _dPdl[ind] = 0;
                for (int k=0; k<nA; k++) {
                    const int k1 = nA*j + k;
                    const int k2 = nA*i + k;
                    _dPdl[ind] += _dLdl[k1]*_L[k2] + _L[k1]*_dLdl[k2];
                }
            }

            grad[1+il] = 0.;
            for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) {
                const int ind = nA*j + i;
                grad[1+il] += -0.5*_dPdl[ind]*_u[i]*_u[j];
            }

            //add the term dependent on the estimate: applies only to diagonal terms
            for (int i=0; i<nA; i++)
                grad[1+il] -= 0.5*_dPdl[nA*i + i]/C_hat[i];

            grad[1+il] *= error;
        }
        }
    {
      //derivative wrt to statistics!
      const vector<Real> grad_acer = gradAcer(out, act, fac);
      const vector<Real> grad_divKL= gradDKL(out, hat);
      Real dot=0, norm=0;
      for (int j=0; j<nA*2; j++) {
        norm += grad_divKL[j]*grad_divKL[j];
        dot += grad_acer[j]*grad_divKL[j];
      }
      const Real proj = (dot - delta)/norm;
      for (int j=0; j<nA*2; j++)
        grad[1+nL+j] = grad_acer[j] - max((Real)0., proj)*grad_divKL[j];

      finalizeVarianceGrad(grad, out);
    }
    return grad;
  }

  /*
  	Easiest way to expand this algo with bounded action space is to assume unbounded
  	space for all intents and purposes inside the algo. When reading from data
  	scale all actions to unbounded, and scale to bounded when sampling policy
  	this should also reduce issues with sigma???
  */
  inline vector<Real> prepareAction(const vector<Real>& act) const
  {
  	//if needed: from env's bounded action space, to unbounded algo policy space
  	assert(act.size()==nA);
  	return aInfo.getInvScaled(act);
  }

  inline void finalizePolicy(Action& a) const
  {
  	a.vals = aInfo.getScaled(a.vals);
  }

  /*
  	Algorithm requires standard deviation for policy (diagonal cov matrix)
  	Therefore map linear net output to sofplus (0<std<inf) and update gradeint accordingly
  */
  inline void prepareVariance(vector<Real>& out) const
  {
  	assert(out.size()==1+nL+2*nA);
  	for (int j=0; j<nA; j++)
  	out[1+nL+nA+j] =.5*(out[1+nL+nA+j]+std::sqrt(std::pow(out[1+nL+nA+j],2)+1));
  }
  inline void finalizeVarianceGrad(vector<Real>& grad, const vector<Real>& out) const
  {
  	assert(grad.size()==1+nL+2*nA);
  	assert(out.size()==1+nL+2*nA);
  	for (int j=0; j<nA; j++)
  	grad[1+nL+nA+j]*=.5*(1+out[1+nL+nA+j]/std::sqrt(1+std::pow(out[1+nL+nA+j],2)));
  }
};
