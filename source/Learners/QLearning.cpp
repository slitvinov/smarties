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

QLearning::QLearning(QApproximator* newQ, ActionInfo& actInfo, Real newGamma, Real newGreedyEps, Real newLRate) :
Q(newQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate)
{
    rng = new RNG(rand());
    suffix = 0;
}

void QLearning::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, Real r, int Nagent)
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
    Q->get(sOld, Qolds, s, Qs, Nagent);
    
    for (int i=0; i<Qs.size(); i++)
    {
        if (Qs[i]>Vnew)
        {
            Nbest = i;
            Vnew = Qs[i];
        }
        if (Qolds[i]>Vold)
        {
            NoldBest = i;
            Vold = Qolds[i];
        }
    }
    
    a.vals[0] = Nbest;
    Real Aold = Qolds[aOld.vals[0]];
    
    //Real err = (Vold + (r + gamma*Vnew - Vold)/.2 - Aold);
    Real err = (r + gamma*Vnew - Aold);
    Real p = rng->uniform();
    if (p < greedyEps)  a.getRand(rng);
    
    Q->correct(sOld, aOld, err, Nagent);
}

void QLearning::try2restart(string fname)
{
    _info("Restarting from saved policy...\n");

    if ( Q->restart(fname) )
    {
        _info("Restart successful, moving on...\n");
    }
    else
    {
        _info("Not all policies restarted, therefore assumed zero. Moving on...\n");
    }
}

void QLearning::savePolicy(string fname)
{
    _info("\nSaving all policies...\n");
    Q->save(fname);
    _info("Done\n");
}

