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

Sarsa::Sarsa(Environment* env, Settings & settings) :
Learner(env,settings), lambda(settings.lambda)
{
    int len;
    if (settings.gamma*settings.lambda < 1e-10)
        len = 2;
    else
        len = round(-10/log10(settings.gamma*settings.lambda)) + 1;
    
    traces.resize(nAgents);
    for (auto& t : traces)
    {
        t.len = len;
        t.start = 0;
        t.hist.resize(len);
        for (auto& e : t.hist)
        {
            e.value = -1;
            e.s = new State(sInfo);
            e.a = new Action(aInfo);
        }
    }
}

void Sarsa::updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r)
{
    //       aOld, r
    // sOld ---------> s
    //
    // Next action:  a
    //
    // Q(sOld, aOld) += lRate * [r + gamma*Q(s, a) - Q(sOld, aOld)]
    //
    int Nbest, NoldBest;
    Real Vnew(-1e10), Vold(-1e10);
    vector<Real> Qolds(a.actInfo.bounds[0]), Qs(a.actInfo.bounds[0]);
    
    traces[agentId].add(sOld, aOld);
    Q->get(sOld, Qolds, s, Qs, agentId);
    for (int i=0; i<a.actInfo.bounds[0]; i++)
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

    Real err = (r + gamma*Vnew - Aold);
    Real p = rng->uniform();
    if (p < greedyEps)  a.getRand(rng);
    
    Q->correct(sOld, aOld, err, agentId);
    
    auto t=traces[agentId];
    int i = t.start;
    while (true)
    {
        History& e = t.hist[i];

        if (e.value > 0) Q->correct(*e.s, *e.a, e.value * err, agentId);
        e.value *= gamma * lambda;
        Real Qe = Q->get(*e.s, *e.a, agentId);
        if (fabs(err) > 1e-8 && e.value >= -0.1) debug1("Sarsa: %f --> %f for %s,  act %s\n",
                   Qe , Qe + err, e.s->print().c_str(), e.a->print().c_str());

        i = (i == t.len-1) ? 0 : i + 1;
        if (i == t.start) break;
    }
}



