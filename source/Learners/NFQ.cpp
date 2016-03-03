/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "../StateAction.h"
#include "NFQ.h"


NFQ::NFQ(QApproximator* newQ, ActionInfo& actInfo, Real newGamma, Real newGreedyEps, Real newLRate) :
Q(newQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate)
{
    rng = new RNG(rand());
    suffix = 0;
}

void NFQ::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, Real r, int Nagent)
{   // No learning here!
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'

    Real Vnew = Q->getMax(s, a, Nagent);
    Real p = rng->uniform();
    if  (p < greedyEps)  a.getRand(rng);
}


void NFQ::improve()
{
    Q->Train();
    /*
    Real err = 1.0;
    Real errold = 10.0;
    Real minerr = 100.0;
    while (fabs((err-errold)/errold)>0.0001 || (err-minerr)/minerr>0.01)
    {
        _info("Iterating batch update\n");
        errold = err;
        err = Q->Train();
        minerr = min(minerr,err);
        Q->save("tmp");
        
        ifstream in("stap.txt");
        if(!in.good())
        {
            _info("Got the message!\n");
            break;
        }
    }
     */
}

void NFQ::try2restart(string fname)
{
    State(sInfo);
    _info("Restarting from history file\n");
    Q->restartSamples();
    
    if ( Q->restart(fname) )
    {
        _info("Restart successful, moving on...\n");
    }
    else
    {
        _info("Not all policies restarted, therefore assumed zero. Moving on...\n");
    }
}

void NFQ::savePolicy(string fname)
{
    _info("\nSaving all policies...\n");
    Q->save(fname);
    _info("Done\n");
}

