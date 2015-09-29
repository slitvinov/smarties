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
#include "../Settings.h"


NFQ::NFQ(QApproximator* newQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate) :
Q(newQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate)
{
    rng = new RNG(rand());
    suffix = 0;
}

void NFQ::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent)
{   // No learning here!
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'

    double best = -1e10;
    actionsIt.reset();
    while (!actionsIt.done())
    {
        const double val = Q->test(s, actionsIt.next(), Nagent);
        if (val >= best + 1e-12)
        {
            best = val;
            actionsIt.memorize();
        }
        // If two actions yield the same Q-value, choose random
        // TODO: Improve for more than 2 same actions
        else if (fabs(best - val) < 1e-12 && rng->uniform() > 0.5)
        {
            actionsIt.memorize();
        }
    }
    a = actionsIt.recall();
    //LSTM: here you perform that action and update memory
    double Qnew = Q->get(s, a, Nagent);
}


void NFQ::NFQimprove(int Niter)
{
    double err = 1;
    
    while (err > 0.015) {
        _info("Iterating batch update\n");
        err = Q->Train();
        
        Q->save("tmp");
    }
}

void NFQ::try2restart(string fname)
{
    State(sInfo);
    _info("Restarting from history file\n");
    
    if ( Q->restart(fname) )
    {
        _info("Restart successful, moving on...\n");
    }
    else
    {
        _info("Not all policies restarted, therefore assumed zero. Moving on...\n");
    }
    
    this->NFQimprove(100);
    Q->save(fname);
    _info("I have a Q and I saved it.\n");
}

void NFQ::savePolicy(string fname)
{
    this->NFQimprove(10);
    _info("\nSaving all policies...\n");
    Q->save(fname);
    _info("Done\n");
}

