//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryProcessing.h"
#include <algorithm>

MemoryProcessing::MemoryProcessing(const Settings&S, MemoryBuffer*const _RM) :
settings(S), RM(_RM) { }

// update the second order moment of the rewards in the memory buffer
void MemoryProcessing::updateRewardsStats(const Real WR, const Real WS, const bool bInit)
{
  if(not settings.bTrain) return; //if not training, keep the stored values

  if(WR>0 or WS>0)
  {
    long double count = 0, newstdvr = 0;
    std::vector<long double> newSSum(dimS, 0), newSSqSum(dimS, 0);
    const Uint setSize = RM->readNSeq();
    #pragma omp parallel reduction(+ : count, newstdvr)
    {
      std::vector<long double> thNewSSum(dimS, 0), thNewSSqSum(dimS, 0);
      #pragma omp for schedule(dynamic) nowait
      for(Uint i=0; i<setSize; i++) {
        count += Set[i]->ndata();
        for(Uint j=0; j<Set[i]->ndata(); j++) {
          newstdvr += std::pow(Set[i]->tuples[j+1]->r, 2);
          for(Uint k=0; k<dimS && WS>0; k++) {
            const long double sk = Set[i]->tuples[j]->s[k] - mean[k];
            thNewSSum[k] += sk; thNewSSqSum[k] += sk*sk;
          }
        }
      }
      if(WS>0) {
        #pragma omp critical
        for(Uint k=0; k<dimS; k++) {
          newSSum[k]   += thNewSSum[k];
          newSSqSum[k] += thNewSSqSum[k];
        }
      }
    }

    //add up gradients across nodes (masters)
    Ssum1Rdx.update(newSSum);
    Ssum2Rdx.update(newSSqSum);
    Csum1Rdx.update(std::vector<long double>{count});
    Rsum2Rdx.update(std::vector<long double>{newstdvr});
  }

  static constexpr Real EPS = numeric_limits<float>::epsilon();
  const long double count = Csum1Rdx.get(bInit)[0];

  if(WR>0)
  {
   long double varR = Rsum2Rdx.get(bInit)[0]/count;
   if(varR < numeric_limits<long double>::epsilon()) varR = 1;
   if( settings.ESpopSize > 1 ) {
     const Real gamma = settings.gamma;
     const auto Rscal = (std::sqrt(varR)+EPS) * (1-gamma>EPS ? 1/(1-gamma) : 1);
     invstd_reward = (1-WR)*invstd_reward +WR/Rscal;
   } else invstd_reward = (1-WR)*invstd_reward + WR / ( std::sqrt(varR) + EPS );
  }

  if(WS>0)
  {
    const std::vector<long double> SSum1 = Ssum1Rdx.get(bInit);
    const std::vector<long double> SSum2 = Ssum2Rdx.get(bInit);
    for(Uint k=0; k<dimS; k++)
    {
      // this is the sample mean minus mean[k]:
      const long double MmM = SSum1[k]/count;
      // mean[k] = (1-WS)*mean[k] + WS * sample_mean, which becomes:
      mean[k] = mean[k] + WS * MmM;
      // if WS==1 then varS is exact, otherwise update second moment
      // centered around current mean[k] (ie. E[(Sk-mean[k])^2])
      long double varS = SSum2[k]/count - MmM*MmM*(2*WS-WS*WS);
      if(varS < numeric_limits<long double>::epsilon()) varS = 1;
      std[k] = (1-WS) * std[k] + WS * std::sqrt(varS);
      invstd[k] = 1/(std[k]+EPS);
    }
  }

  #ifndef NDEBUG
    if( settings.learner_rank == 0 ) {
     std::ofstream outf("runningAverages.dat", std::ios::app);
     outf<<count<<" "<<1/invstd_reward<<" "<<print(mean)<<" "<<print(std)<<endl;
     outf.flush(); outf.close();
    }
    Uint cntSamp = 0;
    for(Uint i=0; i<setSize; i++) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions.load());
    if(WS>=1)
    {
      std::vector<long double> dbgStateSum(dimS,0), dbgStateSqSum(dimS,0);
      #pragma omp parallel
      {
        std::vector<long double> thr_dbgStateSum(dimS), thr_dbgStateSqSum(dimS);
        #pragma omp for schedule(dynamic)
        for(Uint i=0; i<setSize; i++)
          for(Uint j=0; j<Set[i]->ndata(); j++) {
            const auto S = standardize(Set[i]->tuples[j]->s);
            for(Uint k=0; k<dimS; k++) {
              thr_dbgStateSum[k] += S[k]; thr_dbgStateSqSum[k] += S[k]*S[k];
            }
          }
        #pragma omp critical
        for(Uint k=0; k<dimS; k++) {
          dbgStateSum[k]   += thr_dbgStateSum[k];
          dbgStateSqSum[k] += thr_dbgStateSqSum[k];
        }
      }
      for(Uint k=0; k<dimS; k++) {
        const Real dbgMean = dbgStateSum[k]/cntSamp;
        const Real dbgVar = dbgStateSqSum[k]/cntSamp - dbgMean*dbgMean;
        if(std::fabs(dbgMean)>.001 || std::fabs(dbgVar-1)>.001)
          std::cout <<k<<" mean:"<<dbgMean<<" std:"<<dbgVar<<"\n";
      }
    }
  #endif
}

void MemoryProcessing::prune(const FORGET ALGO, const Fval CmaxRho)
{
  //checkNData();
  assert(CmaxRho>=1);
  // vector indicating location of sequence to delete
  int  oldP = -1, farP = -1, dklP = -1;
  Real dklV = -1, farV = -1, oldV = 9e9;
  const Fval invC = 1/CmaxRho;
  Real _nOffPol = 0, _totDKL = 0;
  const Uint setSize = RM->readNSeq();

  #pragma omp parallel reduction(+ : _nOffPol, _totDKL)
  {
    std::pair<int, Real> farpol{-1, -1}, maxdkl{-1, -1}, oldest{-1, 9e9};
    #pragma omp for schedule(dynamic) nowait
    for(Uint i = 0; i < setSize; i++)
    {
      if(Set[i]->just_sampled >= 0) {
        Set[i]->nOffPol = 0; Set[i]->sumKLDiv = 0;
        for(Uint j=0; j<Set[i]->ndata(); j++) {
          const Fval W = Set[i]->offPolicImpW[j];
          Set[i]->sumKLDiv += Set[i]->KullbLeibDiv[j];
          assert( W>=0  &&  Set[i]->KullbLeibDiv[j]>=0 );
          // sequence is off policy if offPol W is out of 1/C : C
          if(W>CmaxRho || W<invC) Set[i]->nOffPol += 1;
        }
      }

      const Real W_FAR = Set[i]->nOffPol /Set[i]->ndata();
      const Real W_DKL = Set[i]->sumKLDiv/Set[i]->ndata();
      _nOffPol += Set[i]->nOffPol; _totDKL += Set[i]->sumKLDiv;

      if(Set[i]->ID<oldest.second) { oldest.second=Set[i]->ID; oldest.first=i; }
      if(    W_FAR >farpol.second) { farpol.second= W_FAR;     farpol.first=i; }
      if(    W_DKL >maxdkl.second) { maxdkl.second= W_DKL;     maxdkl.first=i; }
    }
    #pragma omp critical
    {
      if(oldest.second<oldV) { oldP=oldest.first; oldV=oldest.second; }
      if(farpol.second>farV) { farP=farpol.first; farV=farpol.second; }
      if(maxdkl.second>dklV) { dklP=maxdkl.first; dklV=maxdkl.second; }
    }
  }

  if(CmaxRho<=1) _nOffPol = 0; //then this counter and its effects are skipped
  avgDKL = _totDKL / RM->readNData();
  nOffPol = _nOffPol;
  minInd = oldV;
  assert(oldP<(int)Set.size() && farP<(int)Set.size() && dklP<(int)Set.size());
  assert( oldP >=  0 && farP >=  0 && dklP >=  0 && fitP >=  0 );
  switch(ALGO) {
    case OLDEST:     delPtr = oldP; break;
    case FARPOLFRAC: delPtr = farP; break;
    case MAXKLDIV:   delPtr = dklP; break;
  }
  // prevent any weird race condition from causing deletion of newest data:
  if(Set[oldP]->ID + (int)setSize < Set[delPtr]->ID) delPtr = oldP;
}

void MemoryProcessing::finalize()
{
  if(delPtr<0) die("undefined behavior");
  const int nB4 = RM->readNSeq();

  // safety measure: do not delete trajectory if Nobs > Ntarget
  // but if N > Ntarget even if we remove the trajectory
  // done to avoid bugs if a sequence is longer than maxTotObsNum
  // negligible effect if hyperparameters are chosen wisely
  const Uint maxTotObsNum_loc = settings.maxTotObsNum_loc;
  if(nTransitions.load()-Set[delPtr]->ndata() > maxTotObsNum_loc)
    RM->removeSequence(delPtr);

  delPtr = -1;
  const long nSeq = RM->readNSeq();
  nPruned += nB4 - nSeq;

  #ifdef PRIORITIZED_ER
   if( stepSinceISWeep++ >= 10 || needs_pass ) {
     updateImportanceWeights();
     RM->needs_pass = false;
     stepSinceISWeep = 0;
   }
  #endif

  // reset flags that signal request to update estimators:
  for(long i=0;i<nSeq;i++) if(Set[i]->just_sampled>=0) Set[i]->just_sampled=-1;
}

void MemoryProcessing::getMetrics(ostringstream& buff)
{
  Real avgR = 0;
  const long nSeq = nSequences.load();
  for(long i=0; i<nSeq; i++) avgR += Set[i]->totR;

  real2SS(buff, invstd_reward*avgR/(nSeq+1e-7), 7, 0);
  real2SS(buff, 1/invstd_reward, 6, 1);
  real2SS(buff, avgDKL, 6, 1);

  buff<<" "<<std::setw(5)<<nSeq;
  buff<<" "<<std::setw(7)<<nTransitions.load();
  buff<<" "<<std::setw(7)<<nSeenSequences_loc.load();
  buff<<" "<<std::setw(8)<<nSeenTransitions_loc.load();
  buff<<" "<<std::setw(7)<<minInd;
  buff<<" "<<std::setw(6)<<(int)nOffPol;

  nPruned=0;
}

void MemoryProcessing::getHeaders(ostringstream& buff)
{
  buff <<
  "| avgR | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nFarP ";
}

FORGET MemoryProcessing::readERfilterAlgo(const string setting,
  const bool bReFER)
{
  if(setting == "oldest")     return OLDEST;
  if(setting == "farpolfrac") return FARPOLFRAC;
  if(setting == "maxkldiv")   return MAXKLDIV;
  //if(setting == "minerror")   return MINERROR; miriad ways this can go wrong
  if(setting == "default") {
    if(bReFER) return FARPOLFRAC;
    else       return OLDEST;
  }
  die("ERoldSeqFilter not recognized");
  return OLDEST; // to silence warning
}
