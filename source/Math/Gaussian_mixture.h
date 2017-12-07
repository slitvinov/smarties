/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Gaussian_policy.h"

template<Uint nExperts>
struct Gaussian_mixture
{
public:
  const ActionInfo* const aInfo;
  const Uint iExperts, iMeans, iPrecs, nA, nP;
  const vector<Real>& netOutputs;
  const array<Real, nExperts> unnorm;
  const Real normalization;
  const array<Real, nExperts> experts;
  const array<vector<Real>, nExperts> means, variances, precisions, stdevs;
  //not kosher stuff, but it should work, relies on ordering of operations:

  vector<Real> sampAct;
  Real sampLogPonPolicy=0, sampLogPBehavior=0, sampImpWeight=0, sampInvWeight=0;
  array<Real, nExperts> PactEachExp, DKL_EachExp, logExpBeta;
  Real Pact_Final = -1;
  bool prepared = false;

  static inline Uint compute_nP(const ActionInfo* const aI)
  {
    return nExperts*(1 + 2*aI->dim);
  }

  Gaussian_mixture(const vector <Uint> starts, const ActionInfo*const aI,
    const vector<Real>&out) : aInfo(aI),
    iExperts(starts[0]), iMeans(starts[1]), iPrecs(starts[2]), nA(aI->dim),
    nP(compute_nP(aI)), netOutputs(out), unnorm(extract_unnorm()),
    normalization(compute_norm()), experts(extract_experts()),
    means(extract_mean()), variances(extract_variance()),
    precisions(extract_precision()), stdevs(extract_stdev()) {}
  /*
  Gaussian_mixture(const Gaussian_mixture<nExperts>& c) :
  aInfo(c.aInfo), iExperts(c.iExperts), iMeans(c.iMeans), iPrecs(c.iPrecs),
  nA(c.nA), nP(c.nP), netOutputs(c.netOutputs), unnorm(c.unnorm),
  normalization(c.normalization), experts(c.experts), means(c.means),
  variances(c.variances), precisions(c.precisions), stdevs(c.stdevs)
  { die("no copyconstructing"); }
  */
  Gaussian_mixture& operator= (const Gaussian_mixture<nExperts>& c) { die("no copying"); }


private:
  inline array<Real,nExperts> extract_unnorm() const
  {
    array<Real, nExperts> ret;
    if(nExperts == 1) {ret[0] = 1; return ret;}
    for(Uint i=0;i<nExperts;i++) ret[i]=precision_func(netOutputs[iExperts+i]);
    return ret;
  }
  inline Real compute_norm() const
  {
    Real ret = 0;
    if(nExperts == 1) {return 1;}
    for (Uint j=0; j<nExperts; j++) {
      ret += unnorm[j];
      assert(unnorm[j]>=0);
    }
    return ret + std::numeric_limits<Real>::epsilon();
  }
  inline array<Real,nExperts> extract_experts() const
  {
    array<Real, nExperts> ret;
    if(nExperts == 1) {ret[0] = 1; return ret;}
    assert(normalization>0);
    for(Uint i=0;i<nExperts;i++) ret[i] = unnorm[i]/normalization;
    return ret;
  }
  inline array<vector<Real>,nExperts> extract_mean() const
  {
    array<vector<Real>,nExperts> ret;
    for(Uint i=0; i<nExperts; i++) {
      const Uint start = iMeans + i*nA;
      assert(netOutputs.size() >= start + nA);
      ret[i] = vector<Real>(&(netOutputs[start]),&(netOutputs[start])+nA);
    }
    return ret;
  }
  inline array<vector<Real>,nExperts> extract_variance() const
  {
    array<vector<Real>,nExperts> ret;
    for(Uint i=0; i<nExperts; i++) {
      const Uint start = iPrecs + i*nA;
      assert(netOutputs.size() >= start + nA);
      ret[i] = vector<Real>(nA);
      for (Uint j=0; j<nA; j++) ret[i][j] = precision_func(netOutputs[start+j]);
    }
    return ret;
  }
  inline array<vector<Real>,nExperts> extract_precision() const
  {
    array<vector<Real>,nExperts> ret = variances; //take inverse of precision
    for(Uint i=0;i<nExperts;i++)for(Uint j=0;j<nA;j++)ret[i][j]=1/ret[i][j];
    return ret;
  }
  inline array<vector<Real>,nExperts> extract_stdev() const
  {
    array<vector<Real>,nExperts> ret = variances; //take sqrt of variance
    for(Uint i=0;i<nExperts;i++)for(Uint j=0;j<nA;j++)ret[i][j]=sqrt(ret[i][j]);
    return ret;
  }
  static inline Real precision_func(const Real val)
  {
    //return std::exp(val) + nnEPS; //nan police
    return 0.5*(val + std::sqrt(val*val+1)) +nnEPS;
  }
  static inline Real precision_func_diff(const Real val)
  {
    //return std::exp(val);
    return 0.5*(1.+val/std::sqrt(val*val+1));
  }
  static inline Real oneDnormal(const Real act,const Real mean,const Real prec)
  {
    return std::sqrt(prec/M_PI/2)*std::exp(-std::pow(act-mean,2)*prec/2);
  }

public:
  template <typename T>
  inline string print(const array<T,nExperts> vals)
  {
    std::ostringstream o;
    for (Uint i=0; i<nExperts-1; i++) o << vals[i] << " ";
    o << vals[nExperts-1];
    return o.str();
  }

  inline void prepare(const vector<Real>& unbact, const vector<Real>& beta, const bool bGeometric, const Gaussian_mixture*const pol_hat = nullptr)
  {
    sampAct = map_action(unbact);
    Pact_Final = numeric_limits<Real>::epsilon();  //nan police
    for(Uint j=0; j<nExperts; j++) {
      PactEachExp[j] = 1;
      for(Uint i=0; i<nA; i++)
        PactEachExp[j] *= oneDnormal(sampAct[i], means[j][i], precisions[j][i]);
      Pact_Final += PactEachExp[j] * experts[j];
    }
    assert(Pact_Final>0);
    sampLogPonPolicy = std::log(Pact_Final);
    sampLogPBehavior = evalBehavior(sampAct, beta);
    const Real logW = sampLogPonPolicy - sampLogPBehavior;
    sampImpWeight = bGeometric ? std::exp(logW/nA) : std::exp(logW);
    sampImpWeight = std::min(MAX_IMPW, sampImpWeight);
    sampInvWeight = 1./(sampImpWeight+nnEPS);
    if(pol_hat == nullptr) return;
    for(Uint j=0; j<nExperts; j++) {
      DKL_EachExp[j] = kl_divergence_exp(j,pol_hat);
      logExpBeta[j]  = std::log(experts[j])-std::log(pol_hat->experts[j]);
    }
  }

  static inline Real evalBehavior(const vector<Real>& act, const vector<Real>& beta)
  {
    Real p = 0;
    const Uint NA = act.size();
    for(Uint j=0; j<nExperts; j++) {
      Real pi = 1;
      for(Uint i=0; i<NA; i++) {
        const Real stdi  = beta[i +j*NA +nExperts*(1+NA)]; //then stdevs
        pi *= oneDnormal(act[i], beta[i +j*NA +nExperts], 1/(stdi*stdi));
      }
      p += pi * beta[j]; //beta[j] contains expert
    }
    return std::log(p);
  }
  inline Real evalLogProbability(const vector<Real>& act) const
  {
    Real P = 0;
    for(Uint j=0; j<nExperts; j++) {
      Real pi  = 1;
      for(Uint i=0; i<nA; i++)
        pi *= oneDnormal(act[i], means[j][i], precisions[j][i]);
      P += pi * experts[j];
    }
    return std::log(P);
  }
  
  inline vector<Real> sample(mt19937*const gen, const vector<Real>& beta) const
  {
    std::vector<Real> ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::discrete_distribution<Uint> dE(&(beta[0]), &(beta[0]) +nExperts);
    const Uint experti = dE(*gen);
    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
      if(samp<-NORMDIST_MAX) samp = -NORMDIST_MAX;
      if(samp> NORMDIST_MAX) samp =  NORMDIST_MAX;
      //while (samp > NORMDIST_MAX || samp < -NORMDIST_MAX) samp = dist(*gen);
      const Uint indM = i +experti*nA +nExperts; //after experts come the means
      const Uint indS = i +experti*nA +nExperts*(nA+1); //after means, stdev
      ret[i] = beta[indM] + beta[indS] * samp;
    }
    return ret;
  }
  inline vector<Real> sample(mt19937*const gen) const
  {
    std::vector<Real> ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::discrete_distribution<Uint> dE(&(experts[0]),&(experts[0])+nExperts);
    const Uint experti = dE(*gen);
    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
      if(samp<-NORMDIST_MAX) samp = -NORMDIST_MAX;
      if(samp> NORMDIST_MAX) samp =  NORMDIST_MAX;
      //while (samp > NORMDIST_MAX || samp < -NORMDIST_MAX) samp = dist(*gen);
      ret[i] = means[experti][i] + stdevs[experti][i] * samp;
    }
    return ret;
  }

  inline vector<Real> policy_grad(const vector<Real>& act, const vector<Real>& beta, const Real factor) const
  {
    const Uint NA = act.size();
    vector<Real> ret(nExperts*(1+2*NA), 0), PsBeta(nExperts, 1);
    Real Pact_Beta = numeric_limits<Real>::epsilon(); //nan police
    for(Uint j=0; j<nExperts; j++) {
      for(Uint i=0; i<NA; i++) {
        const Real stdi  = beta[i +j*NA +nExperts*(1+NA)];
        PsBeta[j] *= oneDnormal(act[i], beta[i +j*NA +nExperts], 1/(stdi*stdi));
      }
      assert(PsBeta[j] > 0);
      Pact_Beta += PsBeta[j] * beta[j]; //beta[j] contains expert
    }

    for(Uint j=0; j<nExperts; j++) {
      const Real normExpert = factor * PsBeta[j] / Pact_Beta;
      for(Uint i=0; i<nExperts; i++)
        ret[i] += normExpert * ((i==j)-beta[j])/normalization;

      const Real fac = normExpert * beta[j];
      for (Uint i=0; i<NA; i++) {
        const Uint indM = i+j*nA +nExperts, indS = i+j*nA +(1+nA)*nExperts;
        const Real u = act[i]-beta[indM], stdi = beta[indS];
        ret[indM] = fac*u/(stdi*stdi);
        ret[indS] = 0.5*fac*(u*u/(stdi*stdi) - 1)/(stdi*stdi);
      }
    }
    return ret;
  }

  inline vector<Real> policy_grad(const vector<Real>& act, const Real factor) const
  {
    vector<Real> ret(nExperts +2*nA*nExperts, 0);
    const Real EPS = numeric_limits<Real>::epsilon();
    assert(Pact_Final > 0);
    for(Uint j=0; j<nExperts; j++) {
      const Real normExpert = factor * (PactEachExp[j]/(Pact_Final+EPS));
      assert(PactEachExp[j] > 0);
      for(Uint i=0; i<nExperts; i++)
        ret[i] += normExpert * ((i==j)-experts[j])/normalization;

      //if(PactEachExp[j]<EPS) continue; // NaN police
      const Real fac = normExpert * experts[j];
      for (Uint i=0; i<nA; i++) {
        const Uint indM = i+j*nA +nExperts, indS = i+j*nA +(1+nA)*nExperts;
        const Real u = act[i]-means[j][i];
        //const Real P=sqrt(.5*preci/M_PI)*safeExp(-pow(act[i]-meani,2)*preci);
        ret[indM] = fac*precisions[j][i]*u;
        ret[indS] = 0.5*fac*(u*u*precisions[j][i]-1)*precisions[j][i];
      }
    }
    return ret;
  }

  inline vector<Real> div_kl_opp_grad(const Gaussian_mixture*const pol_hat, const Real fac = 1) const
  {
    vector<Real> ret(nExperts +2*nA*nExperts, 0);
    for(Uint j=0; j<nExperts; j++) {
      for (Uint i=0; i<nA; i++) {
        const Uint indM = i+j*nA +nExperts, indS = i+j*nA +(nA+1)*nExperts;
        const Real meani = means[j][i], preci = precisions[j][i];
        const Real meanh=pol_hat->means[j][i], prech=pol_hat->precisions[j][i];
        ret[indM]= fac*experts[j]*(meani-meanh)*prech;
        ret[indS]= fac*experts[j]*(prech-preci)/2;
      }
      assert(prepared && DKL_EachExp[j] >= 0);
      assert(DKL_EachExp[j] == kl_divergence_exp(j,pol_hat));
      assert(logExpBeta[j] == log(experts[j])-log(pol_hat->experts[j]) );
      const Real tmp = fac*(DKL_EachExp[j] +1 +logExpBeta[j])/normalization;
      for (Uint i=0; i<nExperts; i++) ret[i] += tmp*((i==j)-experts[j]);
    }
    return ret;
  }
  inline vector<Real> div_kl_opp_grad(const vector<Real>&beta, const Real fac=1) const
  {
    //const Real EPS = std::numeric_limits<Real>::epsilon();
    vector<Real> ret(nExperts +2*nA*nExperts, 0);
    for(Uint j=0; j<nExperts; j++) {
      for (Uint i=0; i<nA; i++) {
        const Uint indM = i+j*nA +nExperts, indS = i+j*nA +(nA+1)*nExperts;
        const Real preci = precisions[j][i], prech = 1/std::pow(beta[indS],2);
        ret[indM]= fac*experts[j]*(means[j][i]-beta[indM])*prech;
        ret[indS]= fac*experts[j]*(prech-preci)/2;
      }
      const Real DKL_ExpBeta = kl_divergence_exp(j, beta);
      const Real logRhoBeta = std::log(experts[j]/beta[j]);
      const Real tmp = fac*(DKL_ExpBeta +1 +logRhoBeta)/normalization;
      for (Uint i=0; i<nExperts; i++) ret[i] += tmp*((i==j)-experts[j]);
    }
    return ret;
  }
  inline Real kl_divergence_exp(const Uint expi, const vector<Real>&beta) const
  {
    Real DKLe = 0;
    for (Uint i=0; i<nA; i++) {
      const Real prech = 1/std::pow(beta[i+expi*nA +nExperts*(1+nA)], 2);
      const Real R =prech*variances[expi][i], meanh = beta[i+expi*nA +nExperts];
      DKLe += R-1-std::log(R) +std::pow(means[expi][i]-meanh,2)*prech;
    }
    assert(DKLe>=0);
    return 0.5*DKLe;
  }
  inline Real kl_divergence_exp(const Uint expi, const Gaussian_mixture*const pol) const
  {
    Real DKLe = 0;
    for (Uint i=0; i<nA; i++) {
      const Real pRatio = pol->precisions[expi][i]*variances[expi][i];
      const Real meanh = pol->means[expi][i], prech = pol->precisions[expi][i];
      DKLe += pRatio-1-std::log(pRatio) +std::pow(means[expi][i]-meanh,2)*prech;
    }
    assert(DKLe>=0);
    return 0.5*DKLe;
  }
  inline Real kl_divergence_opp(const Gaussian_mixture*const pol_hat) const
  {
    Real ret = 0;
    for(Uint j=0; j<nExperts; j++) {
      assert(prepared && DKL_EachExp[j] >= 0 && experts[j] > 0);
      assert(DKL_EachExp[j] == kl_divergence_exp(j,pol_hat));
      assert(logExpBeta[j] == log(experts[j])-log(pol_hat->experts[j]) );
      ret += experts[j]*(logExpBeta[j] + DKL_EachExp[j]);
    }
    return ret;
  }
  inline Real kl_divergence_opp(const vector<Real>&beta) const
  {
    Real r = 0;
    for(Uint j=0; j<nExperts; j++)
    r += experts[j]*(std::log(experts[j]/beta[j])+kl_divergence_exp(j,beta));
    return r;
  }
  inline Real kl_div_opp_new(const Gaussian_mixture*const p) const
  {
    Real r = 0;
    for(Uint j=0; j<nExperts; j++)
      r+=experts[j]*(log(experts[j])-log(p->experts[j])+kl_divergence_exp(j,p));
    return r;
  }

  inline void finalize_grad(const vector<Real>&grad, vector<Real>&netGradient) const
  {
    assert(grad.size() == nP);
    for(Uint j=0; j<nExperts; j++) {
      {
        const Real diff = precision_func_diff(netOutputs[iExperts+j]);
        netGradient[iExperts+j] = grad[j] * diff;
      }
      for (Uint i=0; i<nA; i++) {
        netGradient[iMeans +i+j*nA] = grad[i+j*nA +nExperts];
        //if bounded actions pass through tanh!
        //helps against NaNs in converting from bounded to unbounded action space:
        if(aInfo->bounded[i]) {
          if(means[j][i]> BOUNDACT_MAX && netGradient[iMeans +i+j*nA]>0)
            netGradient[iMeans +i+j*nA] = 0;
          else
          if(means[j][i]<-BOUNDACT_MAX && netGradient[iMeans +i+j*nA]<0)
            netGradient[iMeans +i+j*nA] = 0;
        }

        if(precisions[j][i]>ACER_MAX_PREC && grad[i+j*nA +(nA+1)*nExperts]<0)
          netGradient[iPrecs +i+j*nA] = 0;
        else {
          const Real diff = precision_func_diff(netOutputs[iPrecs +i+j*nA]);
          netGradient[iPrecs +i+j*nA] = grad[i+j*nA +(nA+1)*nExperts] * diff;
        }
      }
    }
  }

  inline vector<Real> getBest() const
  {
    const Uint bestExp = std::distance(experts.begin(), std::max_element(experts.begin(),experts.end()));
    return means[bestExp];
  }
  inline vector<Real> finalize(const bool bSample, mt19937*const gen, const vector<Real>& beta) const
  { //scale back to action space size:
    return aInfo->getScaled(bSample ? sample(gen, beta) : getBest());
  }

  inline vector<Real> getBeta() const
  {
    vector<Real> ret(nExperts +2*nA*nExperts);
    for(Uint j=0; j<nExperts; j++) {
      ret[j] = experts[j];
      for (Uint i=0; i<nA; i++) {
        ret[i+j*nA +nExperts]        =  means[j][i];
        ret[i+j*nA +nExperts*(nA+1)] = stdevs[j][i];
      }
    }
    return ret;
  }
  static inline void anneal_beta(vector<Real>& beta, const Real eps) {}
  inline vector<Real> map_action(const vector<Real>& sent) const
  {
    return aInfo->getInvScaled(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI)
  {
    assert(aI->dim);
    return aI->dim;
  }

  void test(const vector<Real>& act, const Gaussian_mixture*const pol_hat) const
  {
    vector<Real> _grad(netOutputs.size());
    const vector<Real> div_klgrad = pol_hat not_eq nullptr? div_kl_opp_grad(pol_hat) : vector<Real>() ;
    const vector<Real> policygrad = policy_grad(act, 1);
    const Uint NEA = nExperts*(1+nA);
    for(Uint i = 0; i<nP; i++)
    {
      vector<Real> out_1 = netOutputs, out_2 = netOutputs;
      const Uint ind = i<nExperts? iExperts+i :
        (i<NEA? iMeans +i-nExperts : iPrecs +i-NEA);
      const Uint ie = i<nExperts? i : (i<NEA? (i-nExperts)/nA : (i-NEA)/nA);
      assert(ie<nExperts);
      if(PactEachExp[ie]<1e-12 && ind >= nExperts) continue;

      out_1[ind] -= 0.0001; out_2[ind] += 0.0001;
      Gaussian_mixture p1(vector<Uint>{iExperts, iMeans, iPrecs}, aInfo, out_1);
      Gaussian_mixture p2(vector<Uint>{iExperts, iMeans, iPrecs}, aInfo, out_2);
      const Real p_1=p1.evalLogProbability(act);
      const Real p_2=p2.evalLogProbability(act);
      {
        finalize_grad(policygrad, _grad);
        const Real fdiff =(p_2-p_1)/.0002, abserr =std::fabs(_grad[ind]-fdiff);
        const Real scale = std::max(std::fabs(fdiff), std::fabs(_grad[ind]));
        if((abserr>1e-7 && abserr/scale>1e-4) && PactEachExp[ie]>nnEPS)
        printf("LogPol grad %d: fin-diff %g analytic %g error %g/%g (%g %g)\n",
        i, fdiff, _grad[ind], abserr,abserr/scale, Pact_Final,PactEachExp[ie]);
        fflush(0);
      }
      if(pol_hat == nullptr) continue;

      const Real d_1=p1.kl_div_opp_new(pol_hat);
      const Real d_2=p2.kl_div_opp_new(pol_hat);
      {
        finalize_grad(div_klgrad, _grad);
        const Real fdiff =(d_2-d_1)/.0002, abserr =std::fabs(_grad[ind]-fdiff);
        const Real scale = std::max(std::fabs(fdiff), std::fabs(_grad[ind]));
        if((abserr>1e-7 && abserr/scale>1e-4) && d_1>1e-8)
        printf("DivKL grad %d: fin-diff %g analytic %g error %g/%g (%g %g)\n",
        i, fdiff, _grad[ind], abserr,abserr/scale, Pact_Final,PactEachExp[ie]);
        fflush(0);
      }
    }
  }
};
