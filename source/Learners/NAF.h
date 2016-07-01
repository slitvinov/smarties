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

class NAF : public Learner
{
    const int nA, nL;
    
    void Train(const int seq, const int first) override;
    void Train(const int seq, const int samp, const int first) override;
    void Train(const vector<int>& seq) override;
    void Train(const vector<int>& seq, const vector<int>& samp) override;

    vector<Real> getPolicy(const vector<Real>& out) const;
    Real computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out) const;
    
public:
    NAF(Environment* env, Settings & settings);
    void select(const int agentId, State& s,Action& a, State& sOld,
                Action& aOld, const int info, Real r) override;
};

