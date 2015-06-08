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
#include "../Settings.h"
 
QLearning::QLearning(QApproximator* newQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate) :
Q(newQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate)
{
	rng = new RNG(rand());
    suffix = 0;
}

void QLearning::selectAction(State &s, Action &a)
{
    double best = -1e10;
    actionsIt.reset();
    while (!actionsIt.done())
    {
        const double val = Q->get(s, actionsIt.next());
        if (val >= best + 1e-12)
        {
            best = val;
            actionsIt.memorize();
        }
        else if (fabs(best - val) < 1e-12 && rng->uniform() > 0.5)
        {
            actionsIt.memorize();
        }
    }
    double p = rng->uniform();
    if (p > greedyEps)  a = actionsIt.recall();
    else                a = actionsIt.getRand(rng);
}

void QLearning::update(State &sOld, Action &a, double r, State &s)
{
    //       a, r
    // sOld ------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'
    
    double best = -1e10;
    actionsIt.reset();
    while (!actionsIt.done())
    {
        double val;
        if ((val = Q->get(s, actionsIt.next())) > best)
            best = val;
    }
    
    double err = lRate * (r + gamma*best - Q->get(sOld, a));
    
    if (fabs(err) > 0.001) debug("Q learning: %f --> %f for %s,  act %s\n",
            Q->get(sOld, a), Q->get(sOld, a) + err, sOld.print().c_str(), a.print().c_str());
    
    Q->correct(sOld, a, err);
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

