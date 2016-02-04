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
#include "Sarsa.h"

Sarsa::Sarsa(QApproximator* q, ActionInfo& actInfo, Real gamma, Real greedyEps, Real lRate, Real lambda) :
Q(q), actionsIt(actInfo), gamma(gamma), greedyEps(greedyEps), lRate(lRate), lambda(lambda)
{
    rng = new RNG(rand());
    suffix = 0;
}

void Sarsa::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, Real r, int Nagent)
{
    //       aOld, r
    // sOld ---------> s
    //
    // Next action:  a
    //
    // Q(sOld, aOld) += lRate * [r + gamma*Q(s, a) - Q(sOld, aOld)]
    //
    
    Real Qold = Q->get(sOld, aOld, Nagent);
    
    // Select new action
    Real best = -1e10;
    actionsIt.reset();
    while (!actionsIt.done())
    {
        const Real val = Q->test(s, actionsIt.next(), Nagent);
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

    Real err = lRate * (r + gamma*best - Qold);

    if (fabs(err) > 1e-8) debug1("Sarsa: Updating trace leading to %s,  act %s\n", s.print().c_str(), a.print().c_str());

    int i = t.start;
    while (true)
    {
        History& e = t.hist[i];

        if (e.value > 0) Q->correct(*e.s, *e.a, e.value * err, Nagent);
        e.value *= gamma * lambda;
        Real Qe = Q->get(*e.s, *e.a, Nagent);
        if (fabs(err) > 1e-8 && e.value >= -0.1) debug1("Sarsa: %f --> %f for %s,  act %s\n",
                   Qe , Qe + err, e.s->print().c_str(), e.a->print().c_str());

        i = (i == t.len-1) ? 0 : i + 1;
        if (i == t.start) break;
    }
}

void Sarsa::try2restart(string fname)
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

void Sarsa::savePolicy(string fname)
{
    _info("\nSaving all policies...\n");
    Q->save(fname);
    _info("Done\n");
}

