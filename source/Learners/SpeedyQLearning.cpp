/*
 *  SpeedyQLearning.cpp
 *  rl
 *
 *  Created by Gabriele Abbati on 14/07/2015.
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "../StateAction.h"
#include "SpeedyQLearning.h"

SpeedyQLearning::SpeedyQLearning(QApproximator* newQ, QApproximator* oldQ, ActionInfo& actInfo, Real newGamma, Real newGreedyEps, Real newLRate, Real LAMBDA) :
Q(newQ), Qold(oldQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate), lambda(LAMBDA)
{
    rng = new RNG(rand());
    suffix = 0;
}

void SpeedyQLearning::updateSelect(Trace& t, State& s, Action& a, Real r, Real alphaK)
{
    //       aOld, r
    // sOld ---------> s
    //
    // Find V_k(s) = max Q_k(s, a')
    //                a'
    // 
    // Find V_{k-1}(s) = max Q_{k-1}(s, a')
    //                    a'
    // 
    // Q(sOld, aOld) += alphaK * (r + gamma*V_{k-1}(s) - Q(sOld, aOld)) + 
    //                              (1 - alphaK) * (gamma*best - gamma*bestOld);

    Real best = -1e10;
    Real bestOld = -1e10;

    actionsIt.reset();
    while (!actionsIt.done())
    {   
        Action& tmpAction = actionsIt.next();
        const Real val = Q->test(s, tmpAction, 0);
        const Real valOld = Qold->test(s, tmpAction, 0);
        if (valOld >= bestOld + 1e-12)
        {
            bestOld = val;
        }
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

    Real p = rng->uniform();
    if (p > greedyEps)  a = actionsIt.recall();
    else                a = actionsIt.getRand(rng);

    State&  sOld = *t.hist[t.start].s;
    Action& aOld = *t.hist[t.start].a;

    // Speedy Q Learning
    Real err = alphaK * (r + gamma*bestOld - Q->get(sOld, aOld, 0)) +
                    (1.0 - alphaK) * (gamma*best - gamma*bestOld);
    Qold->set(sOld, aOld, Q->get(sOld, aOld, 0), 0);
    Q->correct(sOld, aOld, err, 0);

}


void SpeedyQLearning::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, Real r, int Nagent)
{
    //       aOld, r
    // sOld ---------> s
    //
    // Find V_k(s) = max Q_k(s, a')
    //                a'
    // 
    // Find V_{k-1}(s) = max Q_{k-1}(s, a')
    //                    a'
    // 
    // Q(sOld, aOld) += alphaK * (r + gamma*V_{k-1}(s) - Q(sOld, aOld)) + 
    //                              (1 - alphaK) * (gamma*best - gamma*bestOld);

    Real best = -1e10;
    Real bestOld = -1e10;

    actionsIt.reset();
    while (!actionsIt.done())
    {   
        Action& tmpAction = actionsIt.next();
        const Real val = Q->test(s, tmpAction, Nagent);
        const Real valOld = Qold->test(s, tmpAction, Nagent);
        if (valOld >= bestOld + 1e-12)
        {
            bestOld = val;
        }
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

    Real p = rng->uniform();
    if (p > greedyEps)  a = actionsIt.recall();
    else                a = actionsIt.getRand(rng);

    // Speedy Q Learning
    Real err = lRate * (r + gamma*bestOld - Q->get(sOld, aOld, Nagent)) +
                    (1.0 - lRate) * (gamma*best - gamma*bestOld);
    Qold->set(sOld, aOld, Q->get(sOld, aOld, Nagent), Nagent);
    Q->correct(sOld, aOld, err, Nagent);
}


void SpeedyQLearning::try2restart(string fname)
{
    _info("Restarting from saved policy...\n");

    string fnameOld = fname + "Old";

    if ( Q->restart(fname) && Qold->restart(fnameOld))
    {
        _info("Restart successful, moving on...\n");
    }
    else
    {
        _info("Not all policies restarted, therefore assumed zero. Moving on...\n");
    }
}

void SpeedyQLearning::savePolicy(string fname)
{
    _info("\nSaving all policies...\n");
    string fnameOld = fname + "Old";
    Q->save(fname);
    Qold->save(fnameOld);
    _info("Done\n");
}

