/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Transitions.h"
#include <fstream>
//#define CLEAN //dont

Transitions::Transitions(Environment* env, Settings & settings):
aI(env->aI), sI(env->sI), anneal(0), nBroken(0), nTransitions(0),
nSequences(0), env(env), nAppended(settings.dqnAppendS),
path(settings.samplesFile), bSampleSeq(settings.nnType == 1)
{
    Inp.resize(sI.dimUsed);
    Tmp.resize(settings.nAgents);
    for (int i(0); i<settings.nAgents; i++) {
        Tmp[i] = new Sequence();
    }
    dist = new discrete_distribution<int> (1,2); //dummy
    gen = new Gen(settings.gen);
}

#ifdef _Priority_
void Transitions::updateP()
{
    anneal++;
    const int N = Errs.size();
    Ps.resize(N); Ws.resize(N); inds.resize(N);
    std::iota(inds.begin(), inds.end(), 0);
    //sort in decreasing order of the error
    const auto comparator=[this](int a,int b){return Errs[a]>Errs[b];};
    std::sort(inds.begin(), inds.end(), comparator);

    for(int i=0;i<N;i++) Ps[inds[i]]=pow(1./Real(i+1),0.5);
        
    const Real mean_err = accumulate(Errs.begin(), Errs.end(), 0.)/N;
    const Real sum = accumulate(Ps.begin(), Ps.end(), 0.);
    
    printf("Avg MSE %f %d\n",mean_err,N);
    
    const Real beta = .5*(1.+(Real)anneal/(anneal+500)); //TODO
    for(int i=0;i<N;i++) {
        Ps[i]/= sum;
        Ws[i] = pow(N*Ps[i],-beta);
    }
    
    Real scale = *max_element(Ws.begin(), Ws.end());
    for(int i=0;i<N;i++) Ws[i]/=scale;
    
    delete dist;
    dist = new discrete_distribution<int>(Ps.begin(), Ps.end());
    //die("Job's done\n");
}
#endif

void Transitions::restartSamples()
{
    /*
     This guy only reads using env state and action info
     If states need to be packed, this is performed by add
     */
    State t_sO(sI), t_sN(sI);
    vector<Real> d_sO(sI.dim), d_sN(sI.dim);
    Action t_a(aI, gen->g);
    vector<Real> d_a(aI.dim);
    Real reward(0);
    int thisId(-1), agentId(0), Ndata(0), info(0), nSeq(0), tmp(1);
    
    while(true) {
        Ndata=0;
        ifstream in(path.c_str());
        std::string line;
        if(in.good()) {
            while (getline(in, line))  {
                istringstream line_in(line);
                line_in >> thisId;
                if (thisId==agentId) {
                    Ndata++;
                    line_in >> info;
                    //line_in >> info;
                    for(int i=0; i<sI.dim; i++) line_in >> d_sO[i];
                    for(int i=0; i<sI.dim; i++) line_in >> d_sN[i];
                    for(int i=0; i<aI.dim; i++) line_in >> d_a[i];
                    line_in >> reward;
                    
                    t_sO.set(d_sO);
                    t_sN.set(d_sN);
                    t_a.set(d_a);
                    add(0, info, t_sO, t_a, t_sN, reward);
                }
            }
            if (Ndata==0 && agentId>0) break;
            agentId++;
        } else {
            printf("WTF couldnt open file history.txt!\n");
            break;
        }
        in.close();
    }
    
    printf("Found %d broken chains out of %d / %d.\n",nBroken, nSequences, nTransitions);
    
    for (int i(0); i<20; i++) cout << env->max_scale[i] << " ";
    cout << endl;
    
    for (int i(0); i<20; i++) cout << env->min_scale[i] << " ";
    cout << endl;
}

/*
void Transitions::saveSamples()
{
    ofstream fout;
    fout.open("obs_clean.dat",ios::app);
    for(int i=0; i<Set.size(); i++)
    for(int j=1; j<Set[i].s.size(); j++)
    {
        if (Set[i].s[j-1].size() != Set[i].s[j].size()) { die("How did you manage THAT?\n");}
        
        fout << to_string(0) <<" "<< to_string(j==1) <<" ";
        for(int k=0; k<Set[i].s[j].size(); k++)
            fout<<Set[i].s[j-1][k]<<" ";
        for(int k=0; k<Set[i].s[j].size(); k++)
            fout<<Set[i].s[j][k]<<" ";
        fout<<Set[i].a[j]<<" ";
        fout<<Set[i].r[j];
        fout<<endl;
    }
    fout.close();
}*/

void Transitions::passData(const int agentId, const int info, const State & sOld,
                       const Action & a, const State & sNew, const Real reward)
{
    ofstream fout;
    /*
    fout.open("obs_master.dat",ios::app); //safety
    fout << agentId << " "<< info << " " << sOld.printClean().c_str() <<
            sNew.printClean().c_str() << a.printClean().c_str() << reward;
    fout << endl;
    fout.close();
     */
    
    fout.open("history.txt",ios::app);
    fout << agentId << " "<< info << " " << sOld.printClean().c_str() <<
            sNew.printClean().c_str() << a.printClean().c_str() << reward;
    fout << endl;
    fout.close();
    
    add(agentId, info, sOld, a, sNew, reward);
}

void Transitions::add(const int agentId, const int info, const State& sOld,
      const Action& a, const State& sNew, Real reward)
{
    const int sApp = nAppended*sI.dimUsed;
    
    sOld.scaleUsed(Inp);
    if(Tmp[agentId]->tuples.size()!=0) {
        bool same(true);
        const Tuple * const last =Tmp[agentId]->tuples.back();
        //scaled vec only has used dims:
        for (int i=0; i<sI.dimUsed; i++)
            same = same && fabs(last->s[i] - Inp[i])<1e-4;
        
        if (!same) {
            ++nBroken;
            push_back(agentId); //create new sequence
        }
    }
    
    //if first of a new sequence, create slot for sOld = s_0
    if(Tmp[agentId]->tuples.size()==0) {
        Tuple * t = new Tuple();
        t->s = Inp;
        //appended states are zeros: suck on that, FFNN!
        if (sApp>0) t->s.insert(t->s.end(),sApp,0.);
        Tmp[agentId]->tuples.push_back(t);
    }
    
    //at this point we made sure that sOld == s.back() (right?)
    //we can add sNew:
    sNew.scaleUsed(Inp);
    Tuple * t = new Tuple();
    t->s = Inp;
    if (sApp>0) {
        const Tuple * const last = Tmp[agentId]->tuples.back();
        t->s.insert(t->s.end(),last->s[0],last->s[sApp-1]);
    }
    
    const bool new_sample = env->pickReward(sOld,a,sNew,reward); // || info==2 TODO
    t->r = reward;
    t->a = a.pack();
    t->aC = a.scale();
    Tmp[agentId]->tuples.push_back(t);
    if (new_sample) {
        Tmp[agentId]->ended = true;
        push_back(agentId);
    }
}

void Transitions::push_back(const int & agentId)
{
    if(Tmp[agentId]->tuples.size()>3) {
        /*if (nSequences>5000) {
            printf("Too many sequences, trashing the oldest\n");
            nTransitions-=Set[0]->tuples.size()-1;
            for (int i(0); i<Set[0]->tuples.size(); i++)
                delete Set[0]->tuples[i];
            Set[0]->tuples.clear();
            delete Set[0];
            
            nTransitions+=Tmp[agentId]->tuples.size()-1;
            Set[0] = Tmp[agentId];
        } else */{
            nSequences++;
            Set.push_back(Tmp[agentId]);
            nTransitions+=Tmp[agentId]->tuples.size()-1;
        }
        
        Tmp[agentId] = new Sequence();
        
    } else {
        for (int i(0); i<Tmp[agentId]->tuples.size(); i++)
            delete Tmp[agentId]->tuples[i];
        printf("Trashing %d obs.\n",Tmp[agentId]->tuples.size());
        Tmp[agentId]->tuples.clear();
        Tmp[agentId]->ended = false;
    }
    
#ifdef _Priority_
    if (bSampleSeq) Errs.resize(nSequences,1.);
    else  Errs.resize(nTransitions,1.);
#endif
}

int Transitions::sample()
{
    return dist->operator()(*(gen->g));
}

/*
if (d_sO[0]==0 && d_sO[1]==0) {
    while (reward<-9)
    {
        getline(in, line);
        istringstream line_in(line);
        line_in >> thisId;
        line_in >> info;
        line_in >> info;
        for(int i=0; i<sI.dim; i++) line_in >> d_sO[i];
        //if(d_sO[0]==0 && d_sO[1]==0) continue;
        for(int i=0; i<sI.dim; i++) line_in >> d_sN[i];
        for(int i=0; i<aI.dim; i++) line_in >> d_a[i];
        line_in >> reward;
        
        nSeq++; tmp=1;
    }
    continue;
}
if (reward<-9) tmp = 2;
fout <<agentId<<" "<<nSeq<<" "<<tmp<<" "<<t_sO.printClean().c_str() <<
t_sN.printClean().c_str() << t_a.printClean().c_str() << reward << endl;
tmp=0;
while (reward<-9) {
    getline(in, line);
    istringstream line_in(line);
    line_in >> thisId;
    line_in >> info;
    line_in >> info;
    for(int i=0; i<sI.dim; i++) line_in >> d_sO[i];
    
    if(d_sO[0]==0 && d_sO[1]==0) continue;
    for(int i=0; i<sI.dim; i++) line_in >> d_sN[i];
    for(int i=0; i<aI.dim; i++) line_in >> d_a[i];
    line_in >> reward;
    
    nSeq++; tmp=1;
}*/
//debug9("To stack %d %d %d: %s --> %s with %d was rewarded with %f \n", agentId, Tmp[agentId].r.size(), Set.size(),  sOld.printScaled().c_str(), sNew.printScaled().c_str(), a.vals[0], reward);