//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "MemoryBuffer.h"

enum FORGET {OLDEST, FARPOLFRAC, MAXKLDIV};
class MemoryProcessing
{
private:
  const Settings & settings;
  MemoryBuffer * const RM;

  vector<memReal>& invstd = RM->invstd;
  vector<memReal>& mean = RM->mean;
  vector<memReal>& std = RM->std;
  Real& invstd_reward = RM->invstd_reward;

  const Uint dimS = RM->dimS;
  Uint nPruned = 0, minInd = 0, nOffPol = 0, avgDKL = 0;
  int delPtr = -1;

  DelayedReductor<long double> Ssum1Rdx = DelayedReductor<long double>(
                      settings.mastersComm, std::vector<long double>(dimS, 0) );

  DelayedReductor<long double> Ssum2Rdx = DelayedReductor<long double>(
                      settings.mastersComm, std::vector<long double>(dimS, 1) );

  DelayedReductor<long double> Rsum2Rdx = DelayedReductor<long double>(
                      settings.mastersComm, std::vector<long double>(   1, 1) );

  DelayedReductor<long double> Csum1Rdx = DelayedReductor<long double>(
                      settings.mastersComm, std::vector<long double>(   1, 1) );

  const std::vector<Sequence*>& Set = RM->Set;
  std::atomic<Uint>& nSequences = RM->nSequences;
  std::atomic<Uint>& nTransitions = RM->nTransitions;
  std::atomic<Uint>& nSeenSequences_loc = RM->nSeenSequences_loc;
  std::atomic<Uint>& nSeenTransitions_loc = RM->nSeenTransitions_loc;

public:

  MemoryProcessing(const Settings&S, MemoryBuffer*const _RM);

  ~MemoryProcessing() { }

  void updateRewardsStats(const Real WR, const Real WS, const bool bInit=false);

  static FORGET readERfilterAlgo(const string setting, const bool bReFER);

  // Algorithm for maintaining and filtering dataset, and optional imp weight range parameter
  void prune(const FORGET ALGO, const Fval CmaxRho = 1);
  void finalize();

  void getMetrics(ostringstream& buff);
  void getHeaders(ostringstream& buff);

  Uint nFarPol() {
    return nOffPol;
  }
};
