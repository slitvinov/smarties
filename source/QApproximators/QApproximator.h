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

struct Tuple
{
    int agentId;
    State* sOld;
    Action* a;
    State* sNew;
    double reward;
};

struct NFQdata
{
    vector<double> insi;
    vector<double> outi;
    vector<double> pred;
};

struct Transitions
{
    vector<Tuple> Set;
    StateInfo sInfo;
    ActionInfo actInfo;
    
    Transitions(ActionInfo actInfo, StateInfo sInfo): actInfo(actInfo), sInfo(sInfo) {}
    Transitions(){}
    void add(int agentId, State& sOld, Action& a, State& sNew, double reward)
    {
        Tuple tmp;
        
        tmp.sOld   = new State(sInfo);
        tmp.a      = new Action(actInfo);
        tmp.sNew   = new State(sInfo);
        
        tmp.agentId= agentId;
        *tmp.sOld  = sOld;
        *tmp.a     = a;
        *tmp.sNew  = sNew;
        tmp.reward = reward;
        
        //printf("Prova prova roger %d %s %s -> %s %f\n", tmp.agentId, tmp.sOld->print().c_str(), tmp.a->print().c_str(), tmp.sNew->print().c_str(), tmp.reward);
        Set.push_back(tmp); //Growing batch
    }
};

class QApproximator
{
protected:
	StateInfo  sInfo;
	ActionInfo actInfo;
	
public:
	QApproximator(StateInfo newSInfo, ActionInfo newActInfo) : sInfo(newSInfo), actInfo(newActInfo), samples(newActInfo, newSInfo) { };
    QApproximator() { };
    Transitions samples;
	
	virtual double get(const State& s, const Action& a, int nAgent)	= 0;
    virtual double test(const State& s, const Action& a, int nAgent) = 0;
    virtual double advance(const State& s, const Action& a, int nAgent) = 0;
	virtual void set(const State& s, const Action& a, double value, int nAgent) = 0;
	virtual void correct(const State& s, const Action& a, double error, int nAgent) = 0;
    virtual double getMax(const State& s, int nAgent) {return 0.0;}
    virtual double getsmooth(const State& s, const Action& a, int nAgent = 0) {return get(s,a,nAgent);}
    virtual double testMax(const State& s, int & nAct,  int nAgent) {return getMax(s,nAgent);}
    virtual double advanceMax(const State& s, int nAgent) {return getMax(s,nAgent);}
	virtual void save(string name) = 0;
	virtual bool restart(string name) = 0;
    virtual double Train() = 0;
    virtual void passData(int agentId, State& sOld, Action& a, State& sNew, double reward, double altrew)
    {
        ofstream fout;
        fout.open("history.txt",ios::app);
        fout << agentId << " " << sOld.printClean().c_str() << sNew.printClean().c_str() << a.printClean().c_str() << reward << " " << altrew << endl;
        fout.close();
    }
    
    void restartSamples()
    {
        State t_sO(sInfo), t_sN(sInfo);
        vector<double> d_sO(sInfo.dim), d_sN(sInfo.dim);
        Action t_a(actInfo);
        vector<int> d_a(actInfo.dim);
        double reward;
        int agentId;
        
        ifstream in("history.txt");
        std::string line;
        double alt_reward;
        if(in.good())
        {
            unsigned counter = 0;
            while (getline(in, line))
            {
                istringstream line_in(line);
                line_in >> agentId;
                for (int i=0; i<sInfo.dim; i++)
                {
                    line_in >> d_sO[i];
                }
                for (int i=0; i<sInfo.dim; i++)
                {
                    line_in >> d_sN[i];
                }
                for (int i=0; i<actInfo.dim; i++)
                {
                    line_in >> d_a[i];
                }
                
                line_in >> reward;
                line_in >> alt_reward;
                
                //line_in >> reward;
                t_sO.set(d_sO);
                t_sN.set(d_sN);
                t_a.set(d_a);
                samples.add(agentId, t_sO, t_a, t_sN, reward);
            }
        }
        else
        {
            die("WTF couldnt open file history.txt!\n");
        }
        
        in.close();
    }
};
