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

Explorer::Explorer(Environment* env, Settings & settings) :
Learner(env,settings)
{
}

void Explorer::updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r)
{
    die("Hic sunt leones.\n");
    /*
     The idea here is to have 2 networks: one to approx the Q the other to approximate the error made by the Q
     */
    /*
    Real ALfac = .2;
// Predict Q
    int Nbest, NoldBest, Nnext;
    //Real Vold = Q->getMax(sOld, NoldBest, Nagent);
    Real Vnew = Q->testMax(s, Nbest, Nagent);
    Real Aold = Q->advance(sOld, aOld, Nagent);
// Get Error
    Real err =(r + gamma*Vnew) - Aold;
    t.hist[t.start].value = fabs(err/(r + gamma*Vnew)); //t.start is associated with aOld and sOld
    //printf("Set error of trace %d to %f\n", t.start, err);
// Correct Q prediction
    Q->correct(sOld, aOld, err, Nagent); //if NN this be right after Q->advance (or Q->get)!!
// Get uncertainty bounds from state s
    Real Unc = errEst->getMax(s, Nnext, Nagent);
    //printf("Max uncertainty %f\n", Unc);
    a.vals[0] = Nnext;
// Correct error estimate!! We need sOldOld and aOldOld!!!
    int i = (t.start == 0) ? t.len-1 : t.start - 1; //Here lie sOldOld and aOldOld
    Real errOld = errEst->get(*t.hist[i].s, *t.hist[i].a, Nagent);
    //printf("Trace %d\n", i);
    Real errErr = t.hist[i].value + gamma*Unc - errOld;
    errEst->correct(*t.hist[i].s, *t.hist[i].a, errErr, Nagent);
     */
}