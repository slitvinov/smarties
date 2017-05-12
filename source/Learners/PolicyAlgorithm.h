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

//#define __ACER_MAX_PREC 2500.
#define __ACER_MAX_PREC 100.
#define __ACER_MIN_PREC 0.1
#define __ACER_MAX_ACT 10.
#define __ENV_MAX_REW 100.
//#define __ENV_MAX_REW 10.
//#define __ENV_MAX_REW 1.

class PolicyAlgorithm : public Learner
{
protected:
	const Real delta;
	const int nA, nL;
	std::vector<std::mt19937>& generators;

public:
	PolicyAlgorithm(MPI_Comm comm, Environment*const _env, Settings & settings,
	const Real _delta): Learner(comm,_env,settings),delta(_delta),nA(_env->aI.dim),
	nL((_env->aI.dim*_env->aI.dim+_env->aI.dim)/2), generators(settings.generators)
	{}

protected:
	/*
  	Algorithm requires standard deviation for policy (diagonal cov matrix)
  	Therefore map linear net output to sofplus (0<std<inf) and update gradeint accordingly
	 */
	inline vector<Real> extractPrecision(const vector<Real>& out) const
	{
		//if it exists, it is always after guassian mean
		#ifdef __ACER_SAFE
			die("Attempted to extract Precision from safe Racer\n");
		#endif

		vector<Real> ret(nA);
		assert(out.size()>=1+nL+2*nA);
		for (int j=0; j<nA; j++)
			ret[j] = softPlus(out[1+nL+nA+j]);
		return ret;
	}

	inline vector<Real> extractQmean(const vector<Real>& out) const
	{
		#ifdef __ACER_RELAX
			die("Attempted to extract Precision from relaxed Racer\n");
		#endif

		#ifndef __ACER_SAFE //then if Qmean exists, it is always after Precision
			assert(out.size()>=1+nL+3*nA);
			return vector<Real>(&(out[1+nL+2*nA]),&(out[1+nL+2*nA])+nA);
		#else //then there is no precision and it is after mean of policy
			assert(out.size()>=1+nL+2*nA);
			return vector<Real>(&(out[1+nL+1*nA]),&(out[1+nL+1*nA])+nA);
		#endif
	}

	inline vector<Real> extractVariance(const vector<Real>& out) const
	{
			const vector<Real> prec = extractPrecision(out);
			vector<Real> ret(nA);
			for(int i=0; i<nA; i++) ret[i] = 1./prec[i];
			return ret;
	}

	inline vector<Real> finalizeVarianceGrad(const vector<Real>& pgrad,
		const vector<Real>& out) const
	{
		#ifdef __ACER_SAFE
			die("Attempted to get variance grad from safe Racer\n");
		#endif
		vector<Real> vargrad(nA);
		assert(out.size()>=1+nL+2*nA);
		assert(pgrad.size()==2*nA);
		for (int j=0; j<nA; j++)
			vargrad[j] = pgrad[nA+j]*diffSoftPlus(out[1+nL+nA+j]);

		return vargrad;
	}

	inline vector<Real> extractPolicy(const vector<Real>& out) const
	{
		return vector<Real>(&(out[1+nL]),&(out[1+nL])+nA);
	}

	inline Real evaluateLogBehavioralPolicy(const vector<Real>& a,
		const vector<Real>& beta) const
	{
		//assert(beta.size()==nA*2);
		assert(a.size()==nA);
		Real p = 0;
		for(int i=0; i<nA; i++) {
			assert(beta[nA+i]>0);
			p -= 0.5*beta[nA+i]*std::pow(a[i]-beta[i],2);
			p += 0.5*std::log(0.5*beta[nA+i]/M_PI);
		}
		return p;
	}

	inline Real evaluateLogProbability(const vector<Real>& act,
		const vector<Real>& mu, const vector<Real>& prec) const
	{
		assert(mu.size()==nA);
		assert(act.size()==nA);
		assert(prec.size()==nA);
		Real p = 0;
		for(int i=0; i<nA; i++) {
			assert(prec[i]>0);
			p -= 0.5*prec[i]*(act[i]-mu[i])*(act[i]-mu[i]);
			p += 0.5*std::log(0.5*prec[i]/M_PI);
		}
		return p;
	}

	void checkGradient()
	{
			vector<Real> hat(nOutputs), out(nOutputs), act(nA);
			uniform_real_distribution<Real> out_dis(-.5,.5);
			uniform_real_distribution<Real> act_dis(-.5,.5);

			for(int i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
			for(int i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
			for(int i = 0; i<nA; i++) act[i] = act_dis(*gen);

			vector<Real> mu = extractPolicy(out);
			vector<Real> mu_hat = extractPolicy(hat);
			#ifndef __ACER_SAFE
				vector<Real> var = extractVariance(out);
				vector<Real> prec = extractPrecision(out);
				vector<Real> prec_hat = extractPrecision(hat);
			#else
				vector<Real> var = vector<Real>(nA,.01); //std=.1
				vector<Real> prec = vector<Real>(nA,100); //std=.1
				vector<Real> prec_hat = vector<Real>(nA,100); //std=.1
			#endif
			#ifndef __ACER_RELAX
				vector<Real> mean = extractQmean(out);
			#else
				vector<Real> mean = mu;
			#endif

			vector<Real> P = preparePmatrix(out);
			vector<Real> polGrad = policyGradient(mu, prec, act, 1);
			vector<Real> cntGrad = controlGradient(mu, var, P, mean, 1);
			vector<Real> gradDivKL = gradDKL(mu, mu_hat, prec, prec_hat);
			for(int i = 0; i<nA; i++) {
				vector<Real> out_1 = out;
				vector<Real> out_2 = out;
				out_1[1+nL+i] -= 0.0001;
				out_2[1+nL+i] += 0.0001;
				vector<Real> mu_1 = extractPolicy(out_1);
				vector<Real> mu_2 = extractPolicy(out_2);
				const Real A_1 = computeAdvantage(act, mu_1, var, P, mean);
				const Real A_2 = computeAdvantage(act, mu_2, var, P, mean);
				const Real p1 = evaluateLogProbability(act, mu_1, prec);
			 	const Real p2 = evaluateLogProbability(act, mu_2, prec);
				const Real d1 = DivKL(mu_1, mu_hat, prec, prec_hat);
				const Real d2 = DivKL(mu_2, mu_hat, prec, prec_hat);
				printf("LogPol mu Gradient %d: finite differences %g analytic %g \n",
				i, (p2-p1)/0.0002, polGrad[i]);
				printf("Control mu Gradient %d: finite differences %g analytic %g \n",
				i, (A_2-A_1)/0.0002, -cntGrad[i]); //the function returns -grad
				printf("DivKL mu Gradient %d: finite differences %g analytic %g \n",
				i, (d2-d1)/0.0002, gradDivKL[i]);
			}

			#ifndef __ACER_SAFE
			vector<Real> varGrad = finalizeVarianceGrad(polGrad, out);
			vector<Real> trlGrad = finalizeVarianceGrad(cntGrad, out);
			vector<Real> dklGrad = finalizeVarianceGrad(gradDivKL, out);
			for(int i = 0; i<nA; i++) {
				vector<Real> out_1 = out;
				vector<Real> out_2 = out;
				out_1[1+nL+nA+i] -= 0.0001;
				out_2[1+nL+nA+i] += 0.0001;
				vector<Real> prec_1 = extractPrecision(out_1);
				vector<Real> prec_2 = extractPrecision(out_2);
				vector<Real> var_1 = extractVariance(out_1);
				vector<Real> var_2 = extractVariance(out_2);
				const Real A_1 = computeAdvantage(act, mu, var_1, P, mean);
				const Real A_2 = computeAdvantage(act, mu, var_2, P, mean);
				const Real p1 = evaluateLogProbability(act, mu, prec_1);
			 	const Real p2 = evaluateLogProbability(act, mu, prec_2);
				const Real d1 = DivKL(mu, mu_hat, prec_1, prec_hat);
				const Real d2 = DivKL(mu, mu_hat, prec_2, prec_hat);
				printf("LogPol var Gradient %d: finite differences %g analytic %g \n",
				i, (p2-p1)/0.0002, varGrad[i]);
				printf("Control var Gradient %d: finite differences %g analytic %g \n",
				i, (A_2-A_1)/0.0002, -trlGrad[i]); //the function returns -grad
				printf("DivKL var Gradient %d: finite differences %g analytic %g \n",
				i, (d2-d1)/0.0002, dklGrad[i]);
			}
			#endif

			vector<Real> cgrad = criticGradient(P, mu, var, out, mean, act, 1);
			for(int i = 1; i<1+nL; i++)
			{
					vector<Real> out_1 = out;
					vector<Real> out_2 = out;
					out_1[i] -= 0.0001;
					out_2[i] += 0.0001;
					vector<Real> P_1 = preparePmatrix(out_1);
					vector<Real> P_2 = preparePmatrix(out_2);
					const Real A_1 = computeAdvantage(act, mu, var, P_1, mean);
					const Real A_2 = computeAdvantage(act, mu, var, P_2, mean);
					printf("Value Gradient %d: finite differences %g analytic %g \n",
					i, (A_2-A_1)/0.0002, cgrad[i]);
			}

			#ifndef __ACER_RELAX
			for(int i = 0; i<nA; i++)
			{
					vector<Real> out_1 = out;
					vector<Real> out_2 = out;
					#ifndef __ACER_SAFE
					out_1[1+nL+2*nA+i] -= 0.0001;
					out_2[1+nL+2*nA+i] += 0.0001;
					#else
					out_1[1+nL+1*nA+i] -= 0.0001;
					out_2[1+nL+1*nA+i] += 0.0001;
					#endif
					vector<Real> mean_1 = extractQmean(out_1);
					vector<Real> mean_2 = extractQmean(out_2);
					const Real A_1 = computeAdvantage(act, mu, var, P, mean_1);
					const Real A_2 = computeAdvantage(act, mu, var, P, mean_2);
					printf("MeanAct Gradient %d: finite differences %g analytic %g \n",
					i, (A_2-A_1)/0.0002, cgrad[1+nL+i]);
			}
			#endif
	}

	inline vector<Real> samplePolicy(const vector<Real>& pi,
		const vector<Real>& var, const int thrID) const
	{
		assert(pi.size()==nA);
		assert(var.size()==nA);
		std::vector<Real> ret(nA);
		for(int i=0; i<nA; i++) { //sample current policy
			std::normal_distribution<Real> dist(pi[i], std::sqrt(var[i]));
			ret[i] = dist(generators[thrID]);
		}
		return ret;
	}

	inline Real safeExp(const Real val) const
	{
			return std::exp( std::min(9., std::max(-32.,val) ) );
	}

	inline vector<Real> preparePmatrix(const vector<Real>& out) const
	{
		//assume out[0] is V(state), followed by P coeffs
		vector<Real> _L(nA*nA, 0), _P(nA*nA, 0);
		assert(out.size() >= 1+nL);
		{ //fill lower diag matrix L
			int kL = 1;
			for (int j=0; j<nA; j++)
				for (int i=0; i<nA; i++) {
					if (i<j)
						_L[nA*j + i] = out[kL++];
					else if (i==j)
						_L[nA*j + i] = softPlus(out[kL++]);
				}

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

	inline Real advantageVariance(const vector<Real>& pol, const vector<Real>& var,
		const vector<Real>& P, const vector<Real>& mean) const
	{
		vector<Real> PvarP(nA*nA, 0);

		for (int j=0; j<nA; j++)
		for (int i=0; i<nA; i++) {
			const int ind = nA*j + i;
			for (int k=0; k<nA; k++) {
				const int k1 = nA*j + k;
				const int k2 = nA*k + i;
				PvarP[ind] += P[k1]*var[k]*P[k2];
			}
		}

		Real ret = 0;
		for (int i=0; i<nA; i++)
			ret += 0.5*PvarP[nA*i+i]*var[i];

		for (int j=0; j<nA; j++)
			for (int i=0; i<nA; i++) {
				ret += (pol[j]-mean[j])*(pol[i]-mean[i])*PvarP[nA*j+i];
				#ifdef __ACER_RELAX
					assert(std::fabs(pol[i]-mean[i]) < 2.2e-16);
				#endif
			}

		return ret;
	}

	inline Real diagTerm(const vector<Real>& S, const vector<Real>& mu,
		const vector<Real>& a) const
	{
		assert(S.size() == nA);
		assert(a.size() == nA);
		assert(mu.size() == nA);
		Real Q = 0;
		for (int j=0; j<nA; j++) Q += S[j]*std::pow(mu[j]-a[j],2);
		return Q;
	}

	inline Real quadraticNoise(const vector<Real>& P, const vector<Real>& var, const int thrID) const
	{
		vector<Real> q(nA,0);
		for (int j=0; j<nA; j++)
		{
			const Real scale = 0.1*std::sqrt(3)*std::sqrt(var[j]);
			std::uniform_real_distribution<Real> distn(-scale, scale);
			q[j] = distn(generators[thrID]);
		}

		Real Q = 0;
		for (int j=0; j<nA; j++) for (int i=0; i<nA; i++)
			Q += P[nA*j+i]*q[i]*q[j];

		return Q;
	}

	inline Real quadraticTerm(const vector<Real>& P, const vector<Real>& mu,
		const vector<Real>& a) const
	{
		/*
    this function returns the advantage for a given action = -.5 * (a-pi)' P (a-pi)
      Assumptions:
        - if action space is bounded, we are already receiving a that is scaled to unbounded space
		 */
		assert(P.size() == nA*nA);
		assert(mu.size() == nA);
		assert(a.size() == nA);
		Real Q = 0;
		for (int j=0; j<nA; j++)
			for (int i=0; i<nA; i++)  //A = L * L'
				Q += P[nA*j+i]*(a[i]-mu[i])*(a[j]-mu[j]);

		return Q;
	}

	inline Real DivKL(const vector<Real>& mu_cur, const vector<Real>& mu_hat,
        const vector<Real>& prec_cur, const vector<Real>& prec_hat) const
	{
		Real ret = 0;
		for (int i=0; i<nA; i++) {
			ret += prec_cur[i]/prec_hat[i] - 1;
			ret += std::pow(mu_cur[i]-mu_hat[i],2)*prec_cur[i];
			ret += std::log(prec_hat[i]) - std::log(prec_cur[i]);
		}
		return 0.5*ret;
 	}

	inline vector<Real> gradDKL(const vector<Real>& mu_cur, const vector<Real>& mu_hat,
	const vector<Real>& prec_cur, const vector<Real>& prec_hat) const
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
    0.5*(\sum_i( Sigma_1_i*(Sigma_2_i)^-1 + (m_2_i - m_1_i)^2*(Sigma_2_i)^-1 -1 +ln(Sigma_2_i) -ln(Sigma_1_i))
		 */

		vector<Real> ret(2*nA);

		for (int i=0; i<nA; i++)
			ret[i]    = (mu_cur[i]-mu_hat[i])*prec_cur[i];

		for (int i=0; i<nA; i++)
			//               v from trace    v from quadratic term   v from normalization
			ret[i+nA] = .5*(1/prec_hat[i]-1/prec_cur[i] +std::pow(mu_cur[i]-mu_hat[i],2));

		return ret;
	}

	inline vector<Real> gradAcerTrpo(const vector<Real>& DA, const vector<Real>& DKL) const
	{
		assert(DA.size() == nA*2);
		assert(DKL.size() == nA*2);

		vector<Real> gradAcer(nA*2);
		Real dot=0, norm=0;
		for (int j=0; j<nA*2; j++) {
			norm += DKL[j] * DKL[j];
			dot +=  DKL[j] * DA[j];
		}
		const Real proj = std::max((Real)0., (dot - delta)/norm);

		//#ifndef NDEBUG
		//if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
		//#endif

		for (int j=0; j<nA*2; j++)
			gradAcer[j] = DA[j] - proj*DKL[j];

		return gradAcer;
	}

	inline vector<Real> policyGradient(const vector<Real>& mu,
		const vector<Real>& prec, const vector<Real>& act, const Real factor) const
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
		vector<Real> ret(2*nA);

		for (int i=0; i<nA; i++)
		{
			ret[i]    = factor*(act[i]-mu[i])*prec[i];

			#ifdef __ACER_MAX_ACT //clip derivative
				 const Real m = mu[i];
				 const Real s = __ACER_MAX_ACT;
				 ret[i] = clip(ret[i], s-m, -s-m);
			#endif
		}

		for (int i=0; i<nA; i++)
		{
			ret[i+nA] = factor*(.5/prec[i] -0.5*(act[i]-mu[i])*(act[i]-mu[i]));
			#ifdef __ACER_MAX_PREC
			ret[i+nA] = clip(ret[i+nA], __ACER_MAX_PREC-prec[i], __ACER_MIN_PREC-prec[i]);
			#endif
		}

		return ret;
	}

	inline vector<Real> controlGradient(const vector<Real>& pol, const vector<Real>& var,
		const vector<Real>& P, const vector<Real>& mean, const Real eta) const
	{
		vector<Real> gradCC(nA*2, 0);

		for (int j=0; j<nA; j++)
		{
			for (int i=0; i<nA; i++)
			{
				#ifdef __ACER_RELAX //then Qmean and pol must be the same
					assert(std::fabs(mean[i]-pol[i]) < 2.2e-16);
				#endif
				gradCC[j] += eta * P[nA*j +i] * (mean[i] - pol[i]);
			}

			#ifdef __ACER_MAX_ACT //clip derivative
				 const Real m = pol[j];
				 const Real s = __ACER_MAX_ACT;
				 gradCC[j] = clip(gradCC[j], s-m, -s-m);
			#endif
			//if(eta<0) gradCC[j] = 0;
		}

		for (int j=0; j<nA; j++)
		{
			gradCC[j+nA] = eta * 0.5 * P[nA*j +j] * var[j] * var[j];
			#ifdef __ACER_MAX_PREC
			gradCC[j+nA] = clip(gradCC[j+nA], __ACER_MAX_PREC-1/var[j], __ACER_MIN_PREC-1/var[j]);
			#endif
		}
		//for (int i=0; i<nA; i++) gradCC[i] = eta * 2 * (mean[i]-pol[i]) / var[i];
		//for (int j=0; j<nA; j++) gradCC[j+nA] = eta * var[j];
		return gradCC;
	}

	inline Real computeAdvantage(const vector<Real>& act, const vector<Real>& pol,
		const vector<Real>&var, const vector<Real>&P, const vector<Real>&mean) const
	{
		assert(pol.size() == nA);
		assert(mean.size() == nA);
		assert(var.size() == nA);
		assert(P.size() == nA*nA);
		assert(act.size() == nA);
		/*
		computing expectation under policy of (a-pi)'*P*(a-pi) (P non diagonal)
		where policy is defined by mean pi and diagonal covariance matrix
		which is equal to trace[(-0.5*P) * Sigma] (since E(a-pi)=0)
		 */
		Real expectation = quadraticTerm(P, mean, pol);
			assert(expectation > -2.2e-16); //must be pos def
		#ifdef __ACER_RELAX
			assert(std::fabs(expectation) < 2.2e-16);
		#endif
		for(int i=0; i<nA; i++) expectation += P[nA*i+i]*var[i];

		return -0.5*quadraticTerm(P, mean, act) + 0.5*expectation; //subtract expectation from advantage of action
	}

	inline vector<Real> sum3Grads(const vector<Real>& f, const vector<Real>& g,
		const vector<Real>& h) const
	{
		assert(f.size() == 2*nA);
		assert(g.size() == 2*nA);
		assert(h.size() == 2*nA);
		vector<Real> ret(nA*2,0);
		for(int i=0; i<nA*2; i++) ret[i] = f[i]+g[i]+h[i];
		return ret;
	}

	inline vector<Real> sum2Grads(const vector<Real>& f, const vector<Real>& g) const
	{
		assert(f.size() == 2*nA);
		assert(g.size() == 2*nA);
		vector<Real> ret(nA*2,0);
		for(int i=0; i<nA*2; i++) ret[i] = f[i]+g[i];
		return ret;
	}

	inline vector<Real> criticGradient(const vector<Real>& P,
		const vector<Real>& pol, const vector<Real>& var, const vector<Real>& out,
		const vector<Real>& mean, const vector<Real>& act, const Real Qer) const
	{
		assert(out.size()>=1+nL+nA);
		vector<Real> _L(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA), _m(nA);
		#ifndef __ACER_RELAX
			vector<Real> grad(1+nL+nA, 0);
		#else
			vector<Real> grad(1+nL, 0);
		#endif

		grad[0] = Qer;

		int kL = 1;
		for (int j=0; j<nA; j++) {
			_u[j] = act[j] - mean[j];
			_m[j] = pol[j] - mean[j];
			#ifdef __ACER_RELAX
				assert(std::fabs(_m[j]) < 2.2e-16);
			#endif

			for (int i=0; i<nA; i++) {
				if (i<j)
					_L[nA*j + i] = out[kL++];
				else if (i==j)
					_L[nA*j + i] = softPlus(out[kL++]);
			}
		}
		assert(kL==1+nL);

		for (int il=0; il<nL; il++) {
			int kD = 0;
			for (int j=0; j<nA; j++)
				for (int i=0; i<nA; i++) {
					const int ind = nA*j + i;
					_dLdl[ind] = 0;
					if(i<=j) if(kD++==il) _dLdl[ind]=1;
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

			#ifdef  __ENV_MAX_REW
				const Real maxP = __ENV_MAX_REW/__ACER_MAX_ACT/__ACER_MAX_ACT/(1-gamma);
				const Real minP = 0.01/__ACER_MAX_ACT/__ACER_MAX_ACT/(1-gamma);
			#else
				const Real maxP = 1e6;
				const Real maxP = 1e-6;
			#endif

			//add the term dependent on the estimate: applies only to diagonal terms
			for (int j=0; j<nA; j++)
				for (int i=0; i<nA; i++) {
					//not necessary, but maxP constraint should be very lax
					const Real maxPij = maxP - P[nA*j+i];
					//necessary: if on the Q stops depending on P then learning diverges
					const Real minPij = ((i==j) ? minP : -maxP) - P[nA*j+i];
					const Real dOdPij = .5*(_m[i]*_m[j]-_u[i]*_u[j] +(i==j?var[i]:0));
					const Real dEdPij = clip(Qer*dOdPij, maxPij, minPij);
					grad[1+il] += _dPdl[nA*j+i]*dEdPij;
				}
				//	grad[1+il] = std::min(std::max(grad[1+il],-maxP),maxP);
		}
      {
			int kl = 1;
			for (int j=0; j<nA; j++)
				for (int i=0; i<nA; i++) {
					if (i==j) grad[kl] *= diffSoftPlus(out[kl]);
					if (i<=j) kl++;
				}
			assert(kl==1+nL);
      }
		#ifndef __ACER_RELAX
		for (int ia=0; ia<nA; ia++) {
			for (int i=0; i<nA; i++)
				grad[1+nL+ia] += Qer*P[nA*ia+i]*(_u[i]-_m[i]);

			#ifdef __ACER_MAX_ACT //clip derivative
				const Real m = mean[ia];
			 	const Real s = __ACER_MAX_ACT;
				grad[1+nL+ia] = clip(grad[1+nL+ia], s-m, -s-m);
			#endif
		}
		#endif

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

	inline Real softPlus(const Real val) const
	{
		//return 0.5*(val + std::sqrt(val*val+1));
		return sqrt(val + std::sqrt(val*val+1));
	}

	inline Real diffSoftPlus(const Real val) const
	{
		//return 0.5*(1 + val/std::sqrt(val*val+1));
		const Real den = std::sqrt(val*val+1);
		return 0.5*std::sqrt(den+val)/den;
	}

	inline Real clip(const Real val, const Real ub, const Real lb) const
	{
		assert(ub>lb);
		return std::max(std::min(val, ub), lb);
	}
};
