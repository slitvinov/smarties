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

class RACER : public Learner
{
    const int nA, nL;
    const Real delta, truncation;
    std::vector<std::mt19937>& generators;

    void Train_BPTT(const int seq, const int thrID=0) const override;
    void Train(const int seq, const int samp, const int thrID=0) const override;
    //void dumpNetworkInfo(const int agentId);

public:
    RACER(MPI_Comm comm, Environment*const env, Settings & settings);
    void select(const int agentId, State& s,Action& a, State& sOld,
                Action& aOld, const int info, Real r) override;

private:

   inline Real evaluateLogProbability(const vector<Real>& a, const vector<Real>& out) const
   {
      assert(a.size()==nA);
      assert(out.size()==1+nL+3*nA);
      Real p = 0;
      for(int i=0; i<nA; i++) {
         assert(out[1+nL+nA+i]>0);
         p -= 0.5*out[1+nL+nA+i]*std::pow(a[i]-out[1+nL+i],2);
         p += 0.5*std::log(0.5*out[1+nL+nA+i]/M_PI);
      }
      return p;
   }

  inline Real evaluateLogBehavioralPolicy(const vector<Real>& a, const vector<Real>& mu) const
  {
    assert(mu.size()==nA*2);
    assert(a.size()==nA);
    Real p = 0;
    for(int i=0; i<nA; i++) {
	     assert(mu[nA+i]>0);
      p -= 0.5*mu[nA+i]*std::pow(a[i]-mu[i],2);
      p += 0.5*std::log(0.5*mu[nA+i]/M_PI);
	   }
     return p;
  }

  inline vector<Real> preparePmatrix(const vector<Real>& out) const
  {
    vector<Real> _L(nA*nA, 0), _P(nA*nA, 0);
    assert(out.size() == 1+nL+3*nA);
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
    assert(out.size() == 1+nL+3*nA);
    assert(hat.size() == 1+nL+3*nA);
    const vector<Real> pi_cur(&out[1+nL],&out[1+nL]+nA), C_cur(&out[1+nL+nA],&out[1+nL+nA]+nA);
    const vector<Real> pi_hat(&hat[1+nL],&hat[1+nL]+nA), C_hat(&hat[1+nL+nA],&hat[1+nL+nA]+nA);
    vector<Real> ret(2*nA);
    for (int i=0; i<nA; i++) {
      ret[i]    = (pi_cur[i]-pi_hat[i])*C_cur[i];
    }
    for (int i=0; i<nA; i++) {
      //               v from trace    v from quadratic term   v from normalization
      ret[i+nA] = 0.5*(1/C_hat[i] +pow(pi_cur[i]-pi_hat[i],2) -1/C_cur[i]);
    }
    return ret;
  }

  inline vector<Real> gradAcerTrpo(const vector<Real>& DA1, const vector<Real>& DA2, const vector<Real>& DKL) const
  {
    assert(DA1.size() == nA*2);
    assert(DA2.size() == nA*2);
    assert(DKL.size() == nA*2);

    vector<Real> gradAcer(nA*2);
    Real dot=0, norm=0;
    for (int j=0; j<nA*2; j++) {
      norm += DKL[j] * DKL[j];
      dot +=  DKL[j] * (DA1[j] + DA2[j]);
    }
    const Real proj = std::max((Real)0., (dot - delta)/norm);

  	//#ifndef NDEBUG
  		//if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
  	//#endif

    for (int j=0; j<nA*2; j++) {
      gradAcer[j] = (DA1[j]+DA2[j]) - proj*DKL[j];
      //if(gradAcer[j] * (DA1[j]+DA2[j]) < 0) printf("Hit DKL\n");
      //else printf("Not hit DKL\n");
      //if(gradAcer[j] * (DA1[j]+DA2[j]) < 0) gradAcer[j] = 0;
    }

    return gradAcer;
  }

  inline vector<Real> policyGradient(const vector<Real>& out, const vector<Real>& a, const Real factor) const
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

    out gives statistics and determines function whose gradient is computed
    a is action at which grad is evaluated
    factor is the advantage gain
    */
    const vector<Real> pi(&out[1+nL],&out[1+nL]+nA), C(&out[1+nL+nA],&out[1+nL+nA]+nA);
    vector<Real> ret(2*nA);
    for (int i=0; i<nA; i++) {
      ret[i]    = factor*(a[i]-pi[i])*C[i];
    }
    for (int i=0; i<nA; i++) {
      ret[i+nA] = factor*(.5/C[i] -0.5*pow(a[i]-pi[i],2));
    }
    return ret;
  }

  inline Real advantageExpectation(const vector<Real>& pi, const vector<Real>& C,
    const vector<Real>& P, const vector<Real>& mu, const vector<Real>& act) const
  {
    assert(pi.size() == nA);
    assert(mu.size() == nA);
    assert(C.size() == nA);
    assert(P.size() == nA*nA);
    assert(act.size() == nA);
    const Real A_s_a = computeAdvantage(P, mu, act);
    /*
    computing expectation under policy of (a-pi)'*P*(a-pi) (P non diagonal)
    where policy is defined by mean pi and diagonal covariance matrix
    which is equal to trace[(-0.5*P) * Sigma] (since E(a-pi)=0)
    */
    Real expectation = 0;
    for(int i=0; i<nA; i++) expectation += -.5*P[nA*i+i]/C[i];
    expectation += computeAdvantage(P, mu, pi);

    return A_s_a - expectation; //subtract expectation from advantage of action
  }

  inline Real computeQ(const vector<Real>& act, const vector<Real>& pol_out, const vector<Real>& val_out) const
  {
    const vector<Real> pi(&pol_out[1+nL], &pol_out[1+nL]+nA);
    const vector<Real> C(&pol_out[1+nL+nA], &pol_out[1+nL+nA]+nA);
    const vector<Real> mu(&val_out[1+nL+2*nA], &val_out[1+nL+2*nA]+nA);
    const vector<Real> P = preparePmatrix(val_out);
    return val_out[0] + advantageExpectation(pi, C, P, mu, act);
  }

  inline vector<Real> computeGradient(const Real Qerror, const Real Verror,
    const vector<Real>& out, const vector<Real>& hat, const vector<Real>& act,
    const vector<Real>& gradAcer) const
  {
    assert(out.size() == 1+nL+3*nA);
    assert(hat.size() == 1+nL+3*nA);
    assert(gradAcer.size() == 2*nA);
    assert(act.size() == nA);
    //assert(P.size() == 2*nA);
    vector<Real> grad(1+nL+nA*3);
    grad[0] = Qerror+Verror;

    for (int j=0; j<nA*2; j++)
      grad[1+nL+j] = gradAcer[j];
    finalizeVarianceGrad(grad, out);

    {
        //these are used to compute Q, so only involved in value gradient
        const vector<Real> pi_hat(&hat[1+nL], &hat[1+nL]+nA);
        const vector<Real> C_hat(&hat[1+nL+nA], &hat[1+nL+nA]+nA);
        //const vector<Real> pi_hat(&out[1+nL], &out[1+nL]+nA);
        //const vector<Real> C_hat(&out[1+nL+nA], &out[1+nL+nA]+nA);
        const vector<Real> mu_cur(&out[1+nL+2*nA], &out[1+nL+2*nA]+nA);
        vector<Real> _L(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA), _m(nA);
        int kL = 1;
        for (int j=0; j<nA; j++) {
          _u[j] = act[j] - mu_cur[j];
          _m[j] = pi_hat[j] - mu_cur[j];
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
                grad[1+il] -= 0.5*_dPdl[ind]*_u[i]*_u[j];
                grad[1+il] += 0.5*_dPdl[ind]*_m[i]*_m[j];
            }

            //add the term dependent on the estimate: applies only to diagonal terms
            for (int i=0; i<nA; i++)
                grad[1+il] += 0.5*_dPdl[nA*i+i]/C_hat[i];

            grad[1+il] *= Qerror;
        }

         const vector<Real> P = preparePmatrix(out);
         const Real fac = Qerror;
         for (int ia=0; ia<nA; ia++) {
            grad[1+nL+2*nA+ia] = 0.;
            for (int i=0; i<nA; i++) {
                const int ind = nA*ia + i;
                grad[1+nL+2*nA+ia] += P[ind]*(_u[i]-_m[i]);
            }
            grad[1+nL+2*nA+ia] *= Qerror;
         }
    }
    //for (int j=0; j<nA; j++)
    //  grad[1+nL+j] = 0.01*(hat[1+nL+2*nA+j]-out[1+nL+j]);

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
  	assert(out.size()==1+nL+3*nA);
  	for (int j=0; j<nA; j++)
  	out[1+nL+nA+j] =.5*(out[1+nL+nA+j]+std::sqrt(std::pow(out[1+nL+nA+j],2)+1));
  }
  inline void finalizeVarianceGrad(vector<Real>& grad, const vector<Real>& out) const
  {
  	assert(grad.size()==1+nL+3*nA);
  	assert(out.size()==1+nL+3*nA);
  	for (int j=0; j<nA; j++)
  	grad[1+nL+nA+j]*=.5*(1+out[1+nL+nA+j]/std::sqrt(1.+std::pow(out[1+nL+nA+j],2)));
  }
};
