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
    void Train_BPTT(const int seq, const int thrID=0) const override;
    void Train(const int seq, const int samp, const int thrID=0) const override;
    inline int maxInd(const vector<Real>& Qs) const
    {
      assert(Qs.size() == nOutputs);
      Real Val = -1e6;
      //let's just assume that if the best is less than -1e6, something is wrong
      int Nbest = -1;
      for (int i=0; i<nOutputs; ++i) {
          if (Qs[i]>Val) {
            Val = Qs[i];
            Nbest = i;
          }
      }
      assert(Nbest>=0);
      return Nbest;
    }
public:
	NFQ(MPI_Comm comm, Environment*const env, Settings & settings);
    void select(const int agentId, State& s, Action& a, State& sOld,
                Action& aOld, const int info, Real r) override;
};
