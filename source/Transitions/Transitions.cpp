/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Transitions.h"
//#define CLEAN //dont

Transitions::Transitions(Environment* env, Settings & settings):
aI(env->aI), sI(env->sI), anneal(0), mean_err(100.), nbroken(0), env(env)
{
    gen = new mt19937(settings.randSeed);
    Inp.resize(sI.dim);
    Tmp.resize(settings.nAgents);
    dist = new discrete_distribution<int> (1,2); //trash
}

void Transitions::add(const int & agentId, State& sOld, Action& a, State& sNew, const Real & reward)
{
    sOld.scale(Inp);
    if(Tmp[agentId].s.size()>0)
    {
        bool same(true);
        for (int i=0; i<sI.dim; i++)
        if (sI.inUse[i])
        same = same && fabs(Tmp[agentId].s.back()[i] - Inp[i])<1e-3;
            
        if (!same)
        {
            //printf("Unexpected change of time series\n");
            //for (int i=0; i<sI.dim; i++) cout << "--[" << Tmp[agentId].s.back()[i] << " " << Inp[i] << "]-- ";
            //cout << a.vals[0];
            //cout << endl;
            ++nbroken;
            push_back(agentId);
        }
    }
    Tmp[agentId].sOld.push_back(Inp);
    sNew.scale(Inp);
    
    Tmp[agentId].s.push_back(Inp);
    Tmp[agentId].r.push_back(reward);
    Tmp[agentId].a.push_back(a.vals[0]);
    //debug9("To stack %d %d %d: %s --> %s with %d was rewarded with %f \n", agentId, Tmp[agentId].r.size(), Set.size(),  sOld.printScaled().c_str(), sNew.printScaled().c_str(), a.vals[0], reward);
}

void Transitions::push_back(const int & agentId)
{
    if(Tmp[agentId].s.size()>2)
    {
        //printf("Pushing back series of length %d\n",Tmp[agentId].s.size());
        Set.push_back(Tmp[agentId]);
        Errs.push_back(mean_err);
    }
    else  printf("Discarding series of length %d\n",Tmp[agentId].s.size());
    
    clear(agentId);
}

void Transitions::clear(const int & agentId)
{
    Tmp[agentId].s.clear();
    Tmp[agentId].sOld.clear();
    Tmp[agentId].a.clear();
    Tmp[agentId].r.clear();
}

int Transitions::sample()
{
    return dist->operator()(*gen);
}

void Transitions::updateP()
{
    if(Errs.size() != Set.size()) die("That's a problem\n");
    const int N = Errs.size();
    Real beta = .5*(1.+(Real)anneal/(anneal+10000));
    anneal++;
    Ps.resize(N); Ws.resize(N); inds.resize(N);
    std::iota(inds.begin(), inds.end(), 0);
    //sort in decreasing order of the error
    auto comparator = [this](int a, int b){ return Errs[a] > Errs[b]; };
    std::sort(inds.begin(), inds.end(), comparator);

    #pragma omp parallel for
    for(int i=0;i<N;i++) Ps[inds[i]]=pow(1./(i+2),0.5);
        
    mean_err = accumulate(Errs.begin(), Errs.end(), 0.)/N;
    Real sum = accumulate(Ps.begin(), Ps.end(), 0.);
    
    printf("Avg MSE %f %d\n",mean_err,N);
    
    #pragma omp parallel for
    for(int i=0;i<N;i++)
    {
        Ps[i]/= sum;
        Ws[i] = pow(N*Ps[i],-beta);
    }
    
    Real scale = *max_element(Ws.begin(), Ws.end());
    
    #pragma omp parallel for
    for(int i=0;i<N;i++) Ws[i]/=scale;
    
    delete dist;
    dist = new discrete_distribution<int>(Ps.begin(), Ps.end());
    //die("Job's done\n");
}

void Transitions::restartSamples()
{
    State t_sO(sI), t_sN(sI);
    vector<Real> d_sO(sI.dim), d_sN(sI.dim);
    Action t_a(aI);
    vector<int> d_a(aI.dim);
    Real reward;
    vector<Real> _info;
    int thisId, agentId(0), Ndata, nInfo(0), first;

    while(true)
    {
        Ndata=0;
        //printf("Loading from agent %d\n",agentId);
        ifstream in("history.txt");
        std::string line;
        if(in.good())
        {
            while (getline(in, line))
            {
                istringstream line_in(line);
                line_in >> thisId;
                if (thisId==agentId)
                {
                    Ndata++;
                    line_in >> first;
                    for (int i=0; i<sI.dim; i++)
                        line_in >> d_sO[i];
                    for (int i=0; i<sI.dim; i++)
                        line_in >> d_sN[i];
                    for (int i=0; i<aI.dim; i++)
                        line_in >> d_a[i];
                    line_in >> reward;
                    
                    t_sO.set(d_sO);
                    t_sN.set(d_sN);
                    t_a.set(d_a);
                    
                    bool new_sample = env->pickReward(t_sO,t_a,t_sN,reward);
                    add(0, t_sO, t_a, t_sN, reward);
                    if (new_sample) push_back(0);
                }
            }
            
            if (Ndata==0 && agentId>0) break;
            agentId++;
        }
        else
        {
            printf("WTF couldnt open file history.txt!\n");
            break;
        }
        
        in.close();
    }
    
    printf("Found %d broken chains out of %d.\n",nbroken, Set.size());
    
    for (int i(0); i<20; i++) cout << env->max_scale[i] << " ";
    cout << endl;
    for (int i(0); i<20; i++) cout << env->min_scale[i] << " ";
    cout << endl;
}

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
            fout<<Set[i].s[j-1][k] <<" ";
        for(int k=0; k<Set[i].s[j].size(); k++)
            fout<<Set[i].s[j][k] <<" ";
        fout<<Set[i].a[j] <<" ";
        fout<<Set[i].r[j];
        fout<<endl;
    }
    
    fout.close();
}

void Transitions::passData(int & agentId, int & first, State & sOld, Action & a, State & sNew, Real & reward, vector<Real>& info)
{
    ofstream fout;
    fout.open("obs.dat",ios::app); //safety
    fout << agentId << " "<< first << " " << sOld.printClean().c_str() <<
            sNew.printClean().c_str() << a.printClean().c_str() << reward;
    fout << endl;
    fout.close();
    
    fout.open("history.txt",ios::app);
    fout << agentId << " "<< first << " " << sOld.printClean().c_str() <<
            sNew.printClean().c_str() << a.printClean().c_str() << reward;
    fout << endl;
    fout.close();
    
    
    bool new_sample = env->pickReward(sOld,a,sNew,reward);
    add(agentId, sOld, a, sNew, reward);
    if (new_sample) push_back(agentId);
}