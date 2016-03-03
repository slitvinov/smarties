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
#include <random>
#include <algorithm>
#include <fstream>
#include "../Settings.h"

struct Tuple
{
    State* sOld;
    Action* a;
    State* sNew;
    Real reward;
};

struct Tuples
{
    vector<vector<Real>> sOld;
    vector<vector<Real>> s;
    vector<Real> r;
    vector<int>  a;
};

struct NFQdata
{
    vector<Real> insi;
    vector<Real> outi;
    vector<Real> pred;
    int aInd;
};

struct Transitions
{
protected:
    int anneal;
    vector<Tuples> Tmp;
    vector<Real> Inp;
    StateInfo sInfo;
    ActionInfo actInfo;
    mt19937 * gen;
    discrete_distribution<int> * dist;
public:
    vector<Real> Errs;
    vector<Tuples> Set;
    vector<Real> Ps, Ws;
    
    Transitions(const int nAgents, ActionInfo actInfo, StateInfo sInfo, int seed): actInfo(actInfo), sInfo(sInfo), anneal(0)
    {
        gen = new mt19937(seed);
        Inp.resize(sInfo.dim);
        Tmp.resize(nAgents);
        dist = new discrete_distribution<int> (1,2);
    }
    
    void add(const int & agentId, State& sOld, Action& a, State& sNew, const Real & reward)
    {
        //printf("Adding tuple %d to agent %d\n",Tmp[agentId].s.size(),agentId);
        sOld.scale(Inp);
        
        if(Tmp[agentId].s.size()>0)
        {
            bool same(true);
            for (int i=0; i<sInfo.dim; i++)
                same = same && fabs(Tmp[agentId].s.back()[i] - Inp[i])<1e-3;
                
            if (!same) {printf("Unexpected change of time series\n"); push_back(agentId);}
        }
        
        Tmp[agentId].sOld.push_back(Inp);
        sNew.scale(Inp);
        Tmp[agentId].s.push_back(Inp);
        
        Tmp[agentId].r.push_back(reward);
        
        Tmp[agentId].a.push_back(a.vals[0]);
        
        //debug2("To stack %d %d %d: %s --> %s with %d was rewarded with %f \n", agentId, Tmp[agentId].r.size(), Set.size(),  sOld.printScaled().c_str(), sNew.printScaled().c_str(), a.vals[0], reward);
    }
    
    void push_back(const int & agentId)
    {
        if(Tmp[agentId].s.size()>3)
        {
            Set.push_back(Tmp[agentId]);
            Errs.push_back(10.0);
        }
        printf("Pushing series %d\n",Set.size());
        clear(agentId);
    }
    
    void clear(const int & agentId)
    {
        //printf("Clearning series %d\n",agentId);
        Tmp[agentId].s.clear();
        Tmp[agentId].sOld.clear();
        Tmp[agentId].a.clear();
        Tmp[agentId].r.clear();
    }
    
    int sample()
    {
        return dist->operator()(*gen);
    }
    void updateP()
    {
        if(Errs.size() != Set.size()) die("That's a problem\n");
        const int N = Errs.size();
        Real beta = .5;//*(1. + (Real)anneal/(anneal+1000));
        anneal++;
        Ps.resize(N);
        Ws.resize(N);
        std::vector<int> inds(N);
        std::iota(inds.begin(), inds.end(), 0);
        //sort in decreasing order of the error
        auto comparator = [this](int a, int b){ return Errs[a] > Errs[b]; };
        std::sort(inds.begin(), inds.end(), comparator);
        
        #pragma omp parallel for
        for(int i=0;i<N;i++)
        {
            Ps[inds[i]]=pow((Real)Set[inds[i]].s.size()/(i+1),0.5);
            //printf("P %f %f %d\n",Ps[inds[i]],Errs[inds[i]],inds[i]);
        }
        Real err = accumulate(Errs.begin(), Errs.end(), 0.);
        Real sum = accumulate(Ps.begin(), Ps.end(), 0.);
        printf("Avg MSE %f\n",err/N);
        #pragma omp parallel for
        for(int i=0;i<N;i++)
        {
            Ps[i]/=sum;
            Ws[i] = pow(N*Ps[i],-beta);
            //printf("%f %d\n",Ps[i],i);
        }
        
        Real scale = *max_element(Ws.begin(), Ws.end());
        //printf("sclae = %f\n",scale);
        
        #pragma omp parallel for
        for(int i=0;i<N;i++)
            Ws[i]/=scale;
        
        delete dist;
        dist = new discrete_distribution<int>(Ps.begin(), Ps.end());
        //die("Job's done\n");
    }
};

class QApproximator
{
protected:
    int nAgents;
	StateInfo  sInfo;
    ActionInfo actInfo;
    Transitions * samples;
	
public:
	QApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings & settings, int nAgents) : nAgents(nAgents), sInfo(newSInfo), actInfo(newActInfo)
    {
        samples = new Transitions(nAgents, actInfo, sInfo, settings.randSeed);
    };
    
    QApproximator() { };
	
    virtual void  get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent) = 0;
	virtual Real get(const State& s, const Action& a, int nAgent)	= 0;
    virtual void set(const State& s, const Action& a, Real value, int nAgent) = 0;
    virtual Real getMax(const State& s, Action& a, int nAgent) = 0;
	virtual void correct(const State& s, const Action& a, Real error, int nAgent) = 0;
    
	virtual void save(string name) = 0;
	virtual bool restart(string name) = 0;
    virtual void Train() = 0;
    
    virtual void passData(int & agentId, int & first, State & sOld, Action & a, State & sNew, Real & reward, vector<Real>& info)
    {
        //if (first)
        //    samples->clear(agentId);
        
        ofstream fout;
        fout.open("obs.dat",ios::app);
        fout << first << " "<< agentId << " " << sOld.printClean().c_str() << sNew.printClean().c_str() << a.printClean().c_str() << reward;
        //cout << first << " "<< agentId << " " << sOld.printClean().c_str() << sNew.printClean().c_str() << a.printClean().c_str() << reward<< endl;
        for (int i = 0; i<info.size(); i++)
            fout << " " << info[i];
        fout << endl;
        fout.close();
        
        samples->add(agentId, sOld, a, sNew, reward);
        
        if (reward<-.99)
            samples->push_back(agentId);
    }
    
    void restartSamples()
    {
        State t_sO(sInfo), t_sN(sInfo);
        vector<Real> d_sO(sInfo.dim), d_sN(sInfo.dim);
        Action t_a(actInfo);
        vector<int> d_a(actInfo.dim);
        Real reward;
        vector<Real> _info;
        int thisId, agentId=0;
        int Ndata, nInfo(0);
        while(true)
        {
            Ndata=0;
            printf("Loading from agent %d\n",agentId);
            ifstream in("history.txt");
            std::string line;
            if(in.good())
            {
                //getline(in, line);
                //istringstream line_0(line);
                //line_0 >> nInfo;
                //_info.resize(nInfo);
                unsigned counter = 0;
                while (getline(in, line))
                {
                    istringstream line_in(line);
                    line_in >> thisId;
                    if (thisId==agentId)
                    {
                        Ndata++;
                        for (int i=0; i<sInfo.dim; i++)
                            line_in >> d_sO[i];
                        for (int i=0; i<sInfo.dim; i++)
                            line_in >> d_sN[i];
                        for (int i=0; i<actInfo.dim; i++)
                            line_in >> d_a[i];
                        
                        line_in >> reward;
                        for(int i=0; i<nInfo; i++)
                        {
                            if (line_in.good()) line_in >> _info[i];
                            else die("Wrong nInfo\n");
                        }

                        t_sO.set(d_sO);
                        t_sN.set(d_sN);
                        t_a.set(d_a);
                        
                        samples->add(0, t_sO, t_a, t_sN, reward);
                        
                        if (reward<-10.99)
                            samples->push_back(0);
                    }
                }
                
                if (Ndata==0 && agentId>0)
                    break;
                agentId++;
            }
            else
            {
                printf("WTF couldnt open file history.txt!\n");
                break;
            }
            
            in.close();
        }
    }
};
