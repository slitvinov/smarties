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
#define NmaxDATA 10000

Transitions::Transitions(Environment* _env, Settings & settings):
aI(_env->aI),sI(_env->sI),anneal(0),nBroken(0),nTransitions(0),nSequences(0),
env(_env), nAppended(settings.dqnAppendS), batchSize(settings.dqnBatch),
path(settings.samplesFile), bSampleSeq(settings.nnType == 1),
bRecurrent(settings.nnType==1), bWriteToFile(!(settings.samplesFile=="none")),
iOldestSaved(0)
{
    mean.resize(sI.dimUsed, 0);
    std.resize(sI.dimUsed, 1);
    Inp.resize(sI.dimUsed);
    Tmp.resize(settings.nAgents);
    for (int i(0); i<settings.nAgents; i++) Tmp[i] = new Sequence();

    dist = new discrete_distribution<int> (1,2); //dummy
    gen = new Gen(settings.gen);
    Set.reserve(NmaxDATA);
}

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
    int thisId(-1), agentId(0), Ndata(0), info(0);

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

    printf("Found %d broken chains out of %d / %d.\n",
            nBroken, nSequences, nTransitions);

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
        if (Set[i].s[j-1].size() != Set[i].s[j].size()) {
        die("How did you manage THAT?\n");}

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

void Transitions::passData(const int agentId, const int info, const State& sOld,
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

    if (bWriteToFile) {
        fout.open("history.txt",ios::app);
        fout << agentId << " "<< info << " " << sOld.printClean().c_str() <<
                sNew.printClean().c_str() << a.printClean().c_str() << reward;
        fout << endl;
        fout.close();
    }

    add(agentId, info, sOld, a, sNew, reward);
}

void Transitions::add(const int agentId, const int info, const State& sOld,
      const Action& a, const State& sNew, Real reward)
{
    const int sApp = nAppended*sI.dimUsed;

    sOld.copy_observed(Inp);
    if(Tmp[agentId]->tuples.size()!=0) {
        bool same(true);
        const Tuple * const last = Tmp[agentId]->tuples.back();
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
    sNew.copy_observed(Inp);
    Tuple * t = new Tuple();
    t->s = Inp;
    if (sApp>0) {
        const Tuple * const last = Tmp[agentId]->tuples.back();
        t->s.insert(t->s.end(),last->s[0],last->s[sApp-1]);
    }

    const bool new_sample = env->pickReward(sOld,a,sNew,reward,info);
    t->r = reward;
    t->a = a.vals;

    Tmp[agentId]->tuples.push_back(t);
    if (new_sample) {
        Tmp[agentId]->ended = true;
        push_back(agentId);
    }
}


void Transitions::clearFailedSim(const int agentOne, const int agentEnd)
{
  for (int i = agentOne; i<agentEnd; i++) {
    _dispose_object(Tmp[i]);
    Tmp[i] = new Sequence();
  }
}

void Transitions::push_back(const int & agentId)
{
    if(Tmp[agentId]->tuples.size()>3 || Tmp[agentId]->tuples.size()>1e4) {
        if (nSequences>=NmaxDATA) Buffered.push_back(Tmp[agentId]);
        else {
            nSequences++;
            Set.push_back(Tmp[agentId]);
            nTransitions+=Tmp[agentId]->tuples.size()-1;
        }
    } else {
        //for (int i(0); i<Tmp[agentId]->tuples.size(); i++) {
        //    _dispose_object(Tmp[agentId]->tuples[i]);
        //}
        //printf("Trashing %d obs.\n",Tmp[agentId]->tuples.size());
        _dispose_object(Tmp[agentId]);
        //Tmp[agentId]->tuples.clear();
        //Tmp[agentId]->ended = false;
    }

    Tmp[agentId] = new Sequence();
}

void Transitions::update_samples_mean()
{
	int count = 0;
  vector<Real> oldStd = std;
  vector<Real> oldMean = mean;
	std::fill(std.begin(), std.end(), 0.);
	std::fill(mean.begin(), mean.end(), 0.);

	#pragma omp parallel
	{
		//local sum and counter
		vector<Real> sum(sI.dimUsed,0), sum2(sI.dimUsed,0);
		int cnt = 0;

		#pragma omp for schedule(dynamic)
		for(int i=0; i<Set.size(); i++)
		for(const auto & t : Set[i]->tuples) {
			assert(t->s.size() == sI.dimUsed);
			cnt++;
			for (int j=0; j<sI.dimUsed; j++) {
				sum2[j] += t->s[j]*t->s[j];
				sum[j] += t->s[j];
			}
		}

		#pragma omp critical
		{
			count += cnt;
			for (int i=0; i<sI.dimUsed; i++) {
				mean[i] += sum[i];
				std[i] += sum2[i];
			}
		}
	}
  bool bSimilar = true;
	std::cout << "States stds: [";
	for (int i=0; i<sI.dimUsed; i++) {
    bSimilar&= (fabs(std[i]-oldStd[i])/max(fabs(std[i]),fabs(oldStd[i]))<.01);
		std[i] = std::sqrt((std[i] - mean[i]*mean[i]/Real(count))/Real(count));
		std::cout << std[i] << " ";
  }
	std::cout << "]. States means: [";
	for (int i=0; i<sI.dimUsed; i++) {
    bSimilar&= (fabs(mean[i]-oldMean[i])/std[i]<.01);
    mean[i] /= Real(count);
    std::cout << mean[i] << " ";
  }
	std::cout << "]" << std::endl;
  if (!bSimilar)
  warn("Means and/or std changed too much, try increasing the buffer size.\n");
}

vector<Real> Transitions::standardize(const vector<Real>&  state) const
{
	vector<Real> tmp(sI.dimUsed);
   assert(state.size() == sI.dimUsed);
   std::normal_distribution<Real> noise(0.,0.01);
	for (int i=0; i<sI.dimUsed; i++) {
      tmp[i] = (state[i] - mean[i])/std[i];
      tmp[i] += noise(*(gen->g));
   }
    return tmp;
}

void Transitions::synchronize()
{
#if 1==1
	assert(nSequences==Set.size() && NmaxDATA == nSequences);
	#pragma omp parallel for schedule(dynamic)
	for(int i=0; i<Set.size(); i++) {
		int count = 0;
		Set[i]->MSE = 0.;

		for(const auto & t : Set[i]->tuples) {
			Set[i]->MSE += t->SquaredError;
			count++;
		}
		Set[i]->MSE /= (Real)(count-1);
		/*
		for(const auto & t : Set[i]->tuples)
		for (int i=0; i<sI.dimUsed; i++)
		  Set[i]->MSE += std::pow((t->s[i] - mean[i])/std[i], 2);
	   */
	}
    const auto compare=[this](Sequence* a, Sequence* b){return a->MSE<b->MSE;};
    std::sort(Set.begin(), Set.end(), compare);
    if(Set.front()->MSE > Set.back()->MSE) die("WRONG\n");
    iOldestSaved = 0;
#endif
    int nTransitionsInBuf(0),nTransitionsDeleted(0),bufferSize(Buffered.size());
    for(auto & bufTransition : Buffered) {
        const int ind = iOldestSaved++;
        iOldestSaved = (iOldestSaved == NmaxDATA) ? 0 : iOldestSaved;

        nTransitionsDeleted += Set[ind]->tuples.size()-1;
        nTransitionsInBuf += bufTransition->tuples.size()-1;

        nTransitions -= Set[ind]->tuples.size()-1;
        _dispose_object(Set[ind]);

        nTransitions += bufTransition->tuples.size()-1;
        Set[ind] = bufTransition;
    } //number of sequences remains constant
    printf("Removing %lu sequences (avg length %f) associated with small MSE"
      "error in favor of new ones (avg lendth %f)\n", Buffered.size(),
      nTransitionsDeleted/(Real)bufferSize, nTransitionsInBuf/(Real)bufferSize);
    Buffered.resize(0); //no clear?
}

void Transitions::updateSamples()
{
	if(Buffered.size()>0) {
    printf("nSequences %d > NmaxDATA %d (nTransitions=%d, avg seq len = %f).\n",
             nSequences, NmaxDATA, nTransitions, nTransitions/(Real)nSequences);
    synchronize();
   } else
    printf("nSequences %d < NmaxDATA %d (nTransitions=%d, avg seq len = %f).\n",
             nSequences, NmaxDATA, nTransitions, nTransitions/(Real)nSequences);


    const int ndata = (bRecurrent) ? nSequences : nTransitions;
    inds.resize(ndata);
    std::iota(inds.begin(), inds.end(), 0);
    random_shuffle(inds.begin(), inds.end(), *(gen));
    update_samples_mean();
}

int Transitions::sample()
{
    return dist->operator()(*(gen->g));
}

#ifdef _Priority_
void Transitions::updateP()
{
    die("Not correctly implemented. Go away!\n")
    anneal++;
    const int ndata = (bRecurrent) ? nSequences : nTransitions;
    Ps.resize(ndataN); Ws.resize(ndata); inds.resize(ndata);
    std::iota(inds.begin(), inds.end(), 0);

    //sort in decreasing order of the error
    if (bRecurrent) {
    	for(auto & samp : Set) {
			int count(0);
			samp->MSE = 0.;
			for(const auto & t : samp->tuples) {
				samp->MSE += t->SquaredError;
				count++;
			}
			samp->MSE /= count;
		}
        const auto compare=[this](int a,int b){return Set[a]->MSE>Set[b]->MSE;};
        std::sort(inds.begin(), inds.end(), compare);
    } else {
        const auto comparator=[this](int a,int b){
			int k(0), back(0), indT(Set[0]->tuples.size()-1);
			while (a >= indT) {
				back = indT;
				indT += Set[++k]->tuples.size()-1;
			}

			int k(0), back(0), indT(Set[0]->tuples.size()-1);
			while (b >= indT) {
				back = indT;
				indT += Set[++k]->tuples.size()-1;
			}

			seq[i]  = k;
			samp[i] = ind-back;
			index[i] = ind;
        	return Set[ka]->tuples[]->SquaredError > MSE>Set[b]->MSE;
        };
        std::sort(inds.begin(), inds.end(), comparator);
    }

    for(int i=0;i<N;i++) Ps[i]=pow(1./Real(inds[i]+1),0.5);

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
}
#endif
