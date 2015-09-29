/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../StateAction.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

class QApproximator
{
protected:
	StateInfo  sInfo;
	ActionInfo actInfo;
	
public:
	QApproximator(StateInfo newSInfo, ActionInfo newActInfo) : sInfo(newSInfo), actInfo(newActInfo) { };
    QApproximator() { };
	
	virtual double get(const State& s, const Action& a, int nAgent)	= 0;
    virtual double test(const State& s, const Action& a, int nAgent) = 0;
    virtual double advance(const State& s, const Action& a, int nAgent) = 0;
	virtual void set(const State& s, const Action& a, double value, int nAgent) = 0;
	virtual void correct(const State& s, const Action& a, double error, int nAgent) = 0;
    virtual double getMax (const State& s, int nAgent) {return 0.0;}
    virtual double testMax (const State& s, int & nAct,  int nAgent) {return getMax(s,nAgent);}
    virtual double advanceMax (const State& s, int nAgent) {return getMax(s,nAgent);}
	virtual void save(string name) = 0;
	virtual bool restart(string name) = 0;
    virtual double Train() = 0;
    void passData(int agentId, State& sOld, Action& a, State& sNew, double reward, double altrew)
    {
        /*
        //FILE * ppFile = fopen("history.txt", "a");
        ofstream fout;
        fout.open("history.txt",ios::app);
        //fprintf(ppFile,"%d %s %s %s %f\n",slave, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r);
        fout << agentId << " " << sOld.printClean().c_str() << sNew.printClean().c_str() << a.printClean().c_str() << reward <<endl; //<< " " << altrew << endl;
        //fclose(ppFile);
        fout.close();
      */
    }
};
