//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MemoryProcessing_h
#define smarties_MemoryProcessing_h

#include "MemoryBuffer.h"

namespace smarties
{

enum FORGET {OLDEST, FARPOLFRAC, MAXKLDIV};

class MemoryProcessing
{
private:
  MemoryBuffer* const RM;
  const MDPdescriptor & MDP = RM->MDP;
  const Settings & settings = RM->settings;
  const DistributionInfo & distrib = RM->distrib;

  std::vector<nnReal>& invstd = RM->invstd;
  std::vector<nnReal>& mean = RM->mean;
  std::vector<nnReal>& std = RM->std;
  Real& stddev_reward = RM->stddev_reward;
  Real& invstd_reward = RM->invstd_reward;

  const Uint dimS = RM->dimS;
  Uint nPruned = 0, minInd = 0, nOffPol = 0;
  Real avgDKL = 0;
  int delPtr = -1;

  DelayedReductor<long double> Ssum1Rdx = DelayedReductor<long double>(settings, LDvec(dimS, 0) );
  DelayedReductor<long double> Ssum2Rdx = DelayedReductor<long double>(settings, LDvec(dimS, 1) );
  DelayedReductor<long double> Rsum2Rdx = DelayedReductor<long double>(settings, LDvec(   1, 1) );
  DelayedReductor<long double> Csum1Rdx = DelayedReductor<long double>(settings, LDvec(   1, 1) );

  const std::vector<Sequence*>& Set = RM->Set;
  std::atomic<long>& nSequences = RM->nSequences;
  std::atomic<long>& nTransitions = RM->nTransitions;
  std::atomic<long>& nSeenSequences = RM->nSeenSequences;
  std::atomic<long>& nSeenTransitions = RM->nSeenTransitions;
  std::atomic<long>& nSeenSequences_loc = RM->nSeenSequences_loc;
  std::atomic<long>& nSeenTransitions_loc = RM->nSeenTransitions_loc;

public:

  MemoryProcessing(MemoryBuffer*const _RM);

  void updateRewardsStats(const Real WR, const Real WS, const bool bInit=false);

  static FORGET readERfilterAlgo(const std::string setting, const bool bReFER);

  // Algorithm for maintaining and filtering dataset, and optional imp weight range parameter
  void prune(const FORGET ALGO, const Fval CmaxRho = 1);
  void finalize();

  void getMetrics(std::ostringstream& buff);
  void getHeaders(std::ostringstream& buff);

  Uint nFarPol() {
    return nOffPol;
  }
};

}
#endif
