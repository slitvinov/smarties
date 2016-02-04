/*
 *  ALearning.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "../StateAction.h"
#include "ALearning.h"
#include "../Settings.h"

ALearning::ALearning(QApproximator* newQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate) :
Q(newQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate)
{
    rng = new RNG(rand());
    suffix = 0;
}

void ALearning::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent)
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
    
    double Vold = Q->getMax(sOld, NoldBest, Nagent); //LSTM: also memory advances to new state
    
    double Vnew = Q->testMax(s, Nbest, Nagent);
    //printf("(AL) Chosen action %d\n",Nbest);
    double Aold = Q->advance(sOld, aOld, Nagent);
    
    //double err = (Vold + (r + gamma*Vnew - Vold)/.2 - Aold);
    double err = (r + gamma*Vnew - Aold);
    //printf("Err = %f\n", err);
    a.vals[0] = Nbest;

    
    //if (fabs(err) > 0.02) cout << "Err before correct = " << err;
    //if (fabs(err) > 0.02)
   // _info("Q learning: %f (r = %f)--> %f for %s,  act %s\n", Aold, r, Aold + err, sOld.print().c_str(), aOld.print().c_str());
    
    //LSTM: with memory after Q(sOld, aOld)
    Q->correct(sOld, aOld, err, Nagent);
    //_info("Q was %f, err %f, now Q=%f\n",Qold,err,Qnew);
    //if (fabs(err) > 0.02) cout << " Err after correct = " << lRate * (r + gamma*best - Q->test(sOld, aOld, Nagent))  << endl;
}

void ALearning::try2restart(string fname)
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

void ALearning::savePolicy(string fname)
{
    _info("\nSaving all policies...\n");
    Q->save(fname);
    _info("Done\n");
}

