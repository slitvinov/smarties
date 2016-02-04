/*
 *  Explorer.cpp
 *  rl
 *
 *  Created by Guido Novati on 10.10.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "../StateAction.h"
#include "Explorer.h"
#include "../Settings.h"

Explorer::Explorer(QApproximator* newQ, QApproximator* errQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate, double lambda) :
Q(newQ), errEst(errQ), actionsIt(actInfo), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate)
{
    rng = new RNG(rand());
    suffix = 0;
}

void Explorer::updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent)
{
    double ALfac = .2;
// Predict Q
    int Nbest, NoldBest, Nnext;
    //double Vold = Q->getMax(sOld, NoldBest, Nagent);
    double Vnew = Q->testMax(s, Nbest, Nagent);
    double Aold = Q->advance(sOld, aOld, Nagent);
// Get Error
    double err =(r + gamma*Vnew) - Aold;
    t.hist[t.start].value = fabs(err/(r + gamma*Vnew)); //t.start is associated with aOld and sOld
    //printf("Set error of trace %d to %f\n", t.start, err);
// Correct Q prediction
    Q->correct(sOld, aOld, err, Nagent); //if NN this be right after Q->advance (or Q->get)!!
// Get uncertainty bounds from state s
    double Unc = errEst->getMax(s, Nnext, Nagent);
    //printf("Max uncertainty %f\n", Unc);
    a.vals[0] = Nnext;
// Correct error estimate!! We need sOldOld and aOldOld!!!
    int i = (t.start == 0) ? t.len-1 : t.start - 1; //Here lie sOldOld and aOldOld
    double errOld = errEst->get(*t.hist[i].s, *t.hist[i].a, Nagent);
    //printf("Trace %d\n", i);
    double errErr = t.hist[i].value + gamma*Unc - errOld;
    errEst->correct(*t.hist[i].s, *t.hist[i].a, errErr, Nagent);
}

void Explorer::try2restart(string fname)
{
    _info("Restarting from saved policy...\n");

    if ( Q->restart(fname) )
    {
        _info("Policy restart successful, moving on...\n");
    }
    else
    {
        _info("Not all policies restarted, therefore assumed zero. Moving on...\n");
    }
    
    if ( errEst->restart(fname) )
    {
        _info("Uncertainty bounds restart successful, moving on...\n");
    }
    else
    {
        _info("Moving on...\n");
    }
}

void Explorer::savePolicy(string fname)
{
    _info("\nSaving all policies...\n");
    Q->save(fname);
    errEst->save(fname);
    _info("Done\n");
}

