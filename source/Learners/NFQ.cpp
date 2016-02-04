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
    
    //double Qold = Q->get(sOld, aOld, Nagent); //LSTM: also memory advances to new state
    int Nbest, NoldBest;
    double Vold = Q->getMax(sOld, NoldBest, Nagent);
    double Vnew = Q->testMax(s, Nbest, Nagent);
    double Aold = Q->advance(sOld, aOld, Nagent);
    a.vals[0] = Nbest;
    //LSTM: here you perform that action and update memory
}


void NFQ::NFQimprove()
{
    double err = 1.0;
    double errold = 10.0;
    double minerr = 100.0;
    while (fabs((err-errold)/errold)>0.0001 || (err-minerr)/minerr>0.01)
    {
        _info("Iterating batch update\n");
        errold = err;
        err = Q->Train();
        minerr = min(minerr,err);
        Q->save("tmp");
        
        ifstream in("stap.txt");
        if(!in.good())
        {
            _info("Got the message!\n");
            break;
        }
    }
}

void NFQ::try2restart(string fname)
{
    State(sInfo);
    _info("Restarting from history file\n");
    Q->restartSamples();
    
    ofstream fout;
    fout.open("stap.txt",ios::app);
    fout << 1 << endl;
    fout.close();
    
    if ( Q->restart(fname) )
    {
        _info("Restart successful, moving on...\n");
    }
    else
    {
        _info("Not all policies restarted, therefore assumed zero. Moving on...\n");
        this->NFQimprove();
    }
    
    
    Q->save(fname);
    _info("I have a Q and I saved it.\n");
}

void NFQ::savePolicy(string fname)
{
    this->NFQimprove();
    _info("\nSaving all policies...\n");
    Q->save(fname);
    _info("Done\n");
}

