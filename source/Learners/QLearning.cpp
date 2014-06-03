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
}

void QLearning::selectAction(State &s, Action &a)
{
    double best = -1e10;
    actionsIt.reset();
    while (!actionsIt.done())
    {
        double val;
        if ((val = Q->get(s, actionsIt.next())) > best)
        {
            best = val;
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
    
    double err = lRate * (r + best - Q->get(sOld, a));
    
    debug1("Q learning: %f --> %f\n", Q->get(sOld, a), Q->get(sOld, a) + err);
    
    Q->correct(sOld, a, err);
}
