/*
 *  NFQ.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner.h"

using namespace std;

class NFQ : public Learner
{
    void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
    void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;
    void dumpNetworkInfo(const int agentId);
    inline Uint maxInd(const vector<Real>& Qs) const
    {
      assert(Qs.size() == nOutputs);
      Real Val = -1e6;
      //let's just assume that if the best is less than -1e6, something is wrong
      Uint Nbest = -1;
      for (Uint i=0; i<nOutputs; ++i) {
          if (Qs[i]>Val) {
            Val = Qs[i];
            Nbest = i;
          }
      }
      return Nbest;
    }
public:
	NFQ(MPI_Comm comm, Environment*const env, Settings & settings);
    void select(const int agentId, State& s, Action& a, State& sOld,
                Action& aOld, const int info, Real r) override;
};
