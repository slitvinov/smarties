/*
 *  QLearning.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "../StateAction.h"
#include "QLearning.h"

QLearning::QLearning(Environment* env, Settings & settings) :
Learner(env,settings)
{
}

void QLearning::updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r)
{
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'
    //
    // Q(sOld, aOld) += lRate * [r + gamma*V(s) - Q(sOld, aOld)]
    //
    
    int Nbest, NoldBest;
    Real Vnew(-1e10), Vold(-1e10);
    vector<Real> Qolds(a.actInfo.bounds[0]), Qs(a.actInfo.bounds[0]);
    Q->get(sOld, Qolds, s, Qs, agentId);
    
    for (int i=0; i<Qs.size(); i++) {
        if (Qs[i]>Vnew) {
            Nbest = i;
            Vnew = Qs[i];
        }
        if (Qolds[i]>Vold) {
            NoldBest = i;
            Vold = Qolds[i];
        }
    }
    
    a.vals[0] = Nbest;
    Real Aold = Qolds[aOld.vals[0]];
    
    //Real err = (Vold + (r + gamma*Vnew - Vold)/.2 - Aold);
    Real err = (r + gamma*Vnew - Aold);
    uniform_real_distribution<Real> dis(0.,1.);
    if (dis(*gen) < greedyEps)  a.getRand();
    
    Q->correct(sOld, aOld, err, agentId);
}