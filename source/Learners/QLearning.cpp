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

void QLearning::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent)
{
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'
    //
    // Q(sOld, aOld) += lRate * [r + gamma*V(s) - Q(sOld, aOld)]
    //

    double Qold = Q->get(sOld, aOld, Nagent); //LSTM: also memory advances to new state
    
    //LSTM: now test scenarios without updating memory
    double best = -1e10;
    actionsIt.reset();
    while (!actionsIt.done())
    {
        const double val = Q->test(s, actionsIt.next(), Nagent);
        //_info("Q learning: %f for %s,  act %s\n", val, s.print().c_str(), actionsIt.show().print().c_str());
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
    
    double err =  lRate *(r/10. + gamma*best - Qold);
    //printf("Err = %f\n", err);
    double p = rng->uniform();
    if (p > fabs(err))  a = actionsIt.recall();
    else                a = actionsIt.getRand(rng);
    
    if (fabs(Qold)>1e3)
        die("Mkay\n");
    
    //if (fabs(err) > 0.02) cout << "Err before correct = " << err;
    //if (fabs(err) > 0.02)
    //_info("Q learning: %f (r = %f)--> %f for %s,  act %s\n", Qold, r, Qold + err, sOld.print().c_str(), a.print().c_str());
    Qold = Q->advance(sOld, aOld, Nagent);
    //LSTM: with memory after Q(sOld, aOld)
    Q->correct(sOld, aOld, err, Nagent);
    //double Qnew = Q->get(sOld, aOld, Nagent);
    //_info("Q was %f, err %f, now Q=%f\n",Qold,err,Qnew);
    //if (fabs(err) > 0.02) cout << " Err after correct = " << lRate * (r + gamma*best - Q->test(sOld, aOld, Nagent))  << endl;
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

