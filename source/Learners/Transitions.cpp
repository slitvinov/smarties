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
#include <dirent.h>
#include <iterator>

Transitions::Transitions(MPI_Comm comm, Environment* const _env, Settings & settings):
mastersComm(comm), env(_env), nAppended(settings.dqnAppendS), batchSize(settings.dqnBatch),
maxSeqLen(settings.maxSeqLen), minSeqLen(settings.minSeqLen),
maxTotSeqNum(settings.maxTotSeqNum), iOldestSaved(0), bSampleSeq(settings.nnType),
bRecurrent(settings.nnType), bWriteToFile(!(settings.samplesFile=="none")),
bNormalize(settings.nnTypeInput), bTrain(settings.bTrain==1),
path(settings.samplesFile), anneal(0), nBroken(0), nTransitions(0),
nSequences(0), aI(_env->aI), sI(_env->sI), generators(settings.generators), old_ndata(0)
{
    mean.resize(sI.dimUsed, 0);
    std.resize(sI.dimUsed, 1);
    Tmp.resize(max(settings.nAgents,1));
    for (int i=0; i<max(settings.nAgents,1); i++)
        Tmp[i] = new Sequence();

    curr_transition_id.resize(max(settings.nAgents,1));
    dist = new discrete_distribution<int> (1,2); //dummy
    gen = new Gen(&generators[0]);
    Set.reserve(maxTotSeqNum);
}

void Transitions::restartSamplesNew()
{
    /*
      This guy only reads using env state and action info
      If states need to be packed, this is performed by add
    */
    printf("About to read from %s...\n",path.c_str());
    int agentId=0, maxAgentID=0;

    while(true)
    {
        int Ndata=0, thisId=0, oldSampID=-1, oldInfo=2, info=2, sampID=-1;
        vector<Real> vecSold(sI.dim), vecSnew(sI.dim);
        vector<Real> vecAct(aI.dim), policy(aI.dim*2), empty_policy(0);
        State oldState(sI), newState(sI);
        Action action(aI, gen->g);
        Real reward = 0;

        ifstream in(path.c_str());
        std::string line;
        if(in.good())
        {
            while (getline(in, line))
            {
                istringstream line_in(line);
                line_in >> thisId;
                maxAgentID = std::max(thisId, maxAgentID);
                if (thisId==agentId)
                {
                    Ndata++;
                    oldInfo = info;
                    oldSampID = sampID;
                    line_in >> info >> sampID;

                    if((sampID==0) != (info==1))
                      die("Mismatch in transition counter\n");
                    if(sampID != oldSampID+1) {
                      if(info!=1) die("Mismatch in transition change\n");
                      if(oldInfo!=2) nBroken++;
                    }
                    if (info == 1) vecSold = vector<Real>(sI.dim);
                    else vecSold = vecSnew;

                    for(int i=0; i<sI.dim; i++) line_in >> vecSnew[i];
                    for(int i=0; i<aI.dim; i++) line_in >> vecAct[i];
                    line_in >> reward;

                    if(info!=2)
                    for(int i=0; i<aI.dim*2; i++) line_in >> policy[i];

                    oldState.set(vecSold);
                    newState.set(vecSnew);
                    action.set(vecAct);

                    if(info==2)
                    add(0, info, oldState, action, empty_policy, newState, reward);
                    else
                    add(0, info, oldState, action, policy, newState, reward);
                }
            }

            if (Tmp[0]->tuples.size()!=0) push_back(0);
            if (Ndata==0 && agentId>maxAgentID) break;
            agentId++;
        } else {
            printf("WTF couldnt open file history.txt!\n");
            break;
        }
        in.close();
    }

    printf("Found %d broken chains out of %d / %d.\n",
            nBroken, nSequences, nTransitions);
    const bool update_meanstd_needed = nTransitions>0;
    if(syncBoolOr(update_meanstd_needed))
      update_samples_mean(1.0);
    old_ndata = nTransitions;
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
    printf("About to read from %s...\n",path.c_str());
    while(true) {
        Ndata=0;
        std::ifstream in(path.c_str());
        std::string line;
        if(in.good()) {
            while (getline(in, line))  {
                std::istringstream line_in(line);
                if(agentId==0 && thisId==-1) {
		  std::istringstream testline(line);
                  int len = std::distance(std::istream_iterator<std::string>(testline),
                                          std::istream_iterator<std::string>());
                  if(len != 2 + sI.dim*2 + aI.dim + 1) {
                    if(len != 3 + sI.dim + aI.dim*3 + 1) die("Wrong history file\n");
                    printf("Reading data from stochastic policy\n");
                    in.close();
                    return restartSamplesNew();
                  }
                }

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
            if (Ndata==0 && agentId>1) break;
            agentId++;
        } else {
            printf("WTF couldnt open file history.txt!\n");
            break;
        }
        in.close();
    }

    printf("Found %d broken chains out of %d / %d.\n",
            nBroken, nSequences, nTransitions);
    const bool update_meanstd_needed = nTransitions>0;
    if(syncBoolOr(update_meanstd_needed))
      update_samples_mean(1.0);
    old_ndata = nTransitions;
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

int Transitions::passData(const int agentId, const int info, const State& sOld,
                       const Action & a, const State & sNew, const Real reward)
{
    ofstream fout;
    {
      const std::string fname = "obs_agent_"+std::to_string(agentId)+".dat";
      fout.open(fname.c_str(),ios::app); //safety
      fout << agentId << " "<< info << " " << sOld.printClean().c_str() <<
              sNew.printClean().c_str() << a.printClean().c_str() << reward;
      fout << endl;
      fout.close();
    }

    if (bWriteToFile) {
        fout.open("history.txt",ios::app);
        fout << agentId << " "<< info << " " << sOld.printClean().c_str() <<
                sNew.printClean().c_str() << a.printClean().c_str() << reward;
        fout << endl;
        fout.flush();
        fout.close();
    }

    return add(agentId, info, sOld, a, sNew, reward);
}

int Transitions::passData(const int agentId, const int info, const State& sOld,
   const Action& a, const vector<Real>& mu, const State& s, const Real reward)
{
    assert(agentId<curr_transition_id.size());
    const int ret = add(agentId, info, sOld, a, mu, s, reward);
    if (ret) curr_transition_id[agentId] = 0;

    ofstream fout;

    if (bWriteToFile)
    {
      fout.open("history.txt",ios::app); //safety
      fout << agentId << " " << info << " " << curr_transition_id[agentId] << " "
       << s.printClean().c_str() << a.printClean().c_str() << reward << printVec(mu);
      fout << endl;
      fout.flush();
      fout.close();
    }
    {
      const std::string fname = "obs_agent_"+std::to_string(agentId)+".dat";
      fout.open(fname.c_str(),ios::app); //safety
      fout << agentId << " " << info << " " << curr_transition_id[agentId] << " "
       << s.printClean().c_str() << a.printClean().c_str() << reward << printVec(mu);
      fout << endl;
      fout.flush();
      fout.close();
    }

	  curr_transition_id[agentId]++;
    return ret;
}

int Transitions::add(const int agentId, const int info, const State & sOld,
   const Action& aNew, const vector<Real>& mu, const State& sNew, Real rNew)
{
    //return value is 1 if the agent states buffer is empty or on initial state
    int ret = 0;
    const int sApp = nAppended*sI.dimUsed;
    if (Tmp[agentId]->tuples.size()!=0 && info == 1) {
      push_back(agentId); //create new sequence
      ret = 1;
    } else if(Tmp[agentId]->tuples.size()==0) {
      assert(info == 1);
      ret = 1; //new Sequence
    }

    if (Tmp[agentId]->tuples.size() >= maxSeqLen) {
      //upper limit to how long a sequence can be
      //printf("Sequence is too long!\n");
      const Tuple * const l = Tmp[agentId]->tuples.back();
      Tuple * t = new Tuple(); //backup last state
      t->s = l->s; t->a = l->a; t->r = l->r, t->mu = l->mu;
      t->SquaredError = l->SquaredError;

      push_back(agentId); //create new sequence
      Tmp[agentId]->tuples.push_back(t);
    }

    //we can add sNew:
    Tuple * t = new Tuple();
    t->s = sNew.copy_observed();
    if (sApp>0) {
        if(Tmp[agentId]->tuples.size()==0)
          t->s.insert(t->s.end(),sApp,0.);
        else {
          const Tuple * const last = Tmp[agentId]->tuples.back();
          t->s.insert(t->s.end(),last->s.begin(),last->s.begin()+sApp);
          assert(last->s.size()==t->s.size());
        }
    }

    const bool end_seq = env->pickReward(sOld,aNew,sNew,rNew,info);
    assert((info==2)==end_seq); //alternative not supported
    t->a = aNew.vals;
    t->r = rNew;
    t->mu = mu;

    Tmp[agentId]->tuples.push_back(t);
    if (end_seq) {
        Tmp[agentId]->ended = true;
        push_back(agentId);
    }

    return ret;
}

int Transitions::add(const int agentId, const int info, const State& sOld,
      const Action& a, const State& sNew, Real reward)
{
    //return value is 1 in two cases:
    //if the agent states buffer is empty
    //is the stored s does not match sold
    int ret = 0;
    const int sApp = nAppended*sI.dimUsed;
    const vector<Real> vecSold = sOld.copy_observed();

    if(Tmp[agentId]->tuples.size()!=0) {
        bool same(true);
        const Tuple * const last = Tmp[agentId]->tuples.back();
        //scaled vec only has used dims:
        for (int i=0; i<sI.dimUsed; i++)
            same = same && fabs(last->s[i] - vecSold[i])<1e-4;

        if (!same) {
            printf("Broken chain %s, %g\n", sNew.print().c_str(), reward);
            push_back(agentId); //create new sequence
            ret = 1;
        }
    } else ret = 1;

    if (Tmp[agentId]->tuples.size() >= maxSeqLen) {
      //upper limit to how long a sequence can be
      //printf("Sequence is too long!\n");
      const Tuple * const l = Tmp[agentId]->tuples.back();
      Tuple * t = new Tuple(); //backup last state
      t->s =l->s; t->a=l->a; t->r =l->r;
      t->SquaredError = l->SquaredError;

      push_back(agentId); //create new sequence
      Tmp[agentId]->tuples.push_back(t);
    }

    //if first of a new sequence, create slot for sOld = s_0
    if(Tmp[agentId]->tuples.size()==0) {
        Tuple * t = new Tuple();
        t->s = vecSold;
        //appended states are zeros: suck on that, FFNN!
        if (sApp>0) t->s.insert(t->s.end(),sApp,0.);
        Tmp[agentId]->tuples.push_back(t);
    }

    //at this point we made sure that sOld == s.back() (right?)
    //we can add sNew:
    const vector<Real> vecSnew = sNew.copy_observed();
    Tuple * t = new Tuple();
    t->s = vecSnew;
    if (sApp>0) {
        const Tuple * const last = Tmp[agentId]->tuples.back();
        vector<Real> prev(sApp);
        for(int i=0; i<sApp; i++) prev[i] = last->s[i];
        t->s.insert(t->s.end(),prev.begin(),prev.end());
    }

    const bool end_seq = env->pickReward(sOld,a,sNew,reward,info);
    t->r = reward;
    t->a = a.vals;
    /*
    ofstream fout;
    fout.open("rewards.dat",ios::app);
    fout<<t->s[0]<<" "<<t->s[1]<<" "<<t->s[2]<<" "<<t->s[3]<<" "<<reward<<endl;
    fout.flush();
    fout.close();
    */
    Tmp[agentId]->tuples.push_back(t);
    if (end_seq) {
        Tmp[agentId]->ended = true;
        push_back(agentId);
    }
    return ret;
}

void Transitions::clearFailedSim(const int agentOne, const int agentEnd)
{
  for (int i = agentOne; i<agentEnd; i++) {
    _dispose_object(Tmp[i]);
    Tmp[i] = new Sequence();
  }
}

void Transitions::pushBackEndedSim(const int agentOne, const int agentEnd)
{
  for (int i = agentOne; i<agentEnd; i++)
    push_back(i);
}

void Transitions::push_back(const int & agentId)
{
    if(Tmp[agentId]->tuples.size() > minSeqLen ) {
        if (nSequences>=maxTotSeqNum) Buffered.push_back(Tmp[agentId]);
        else {
            nSequences++;
            if (not Tmp[agentId]->ended) ++nBroken;

            Set.push_back(Tmp[agentId]);
            nTransitions+=Tmp[agentId]->tuples.size()-1;
        }
    } else {
        //for (int i(0); i<Tmp[agentId]->tuples.size(); i++) {
        //    _dispose_object(Tmp[agentId]->tuples[i]);
        //}
        printf("Trashing %lu obs.\n",Tmp[agentId]->tuples.size());
        fflush(0);
        _dispose_object(Tmp[agentId]);
        //Tmp[agentId]->tuples.clear();
        //Tmp[agentId]->ended = false;
    }

    Tmp[agentId] = new Sequence();
}

int Transitions::syncBoolOr(int needed) const
{
  assert(needed>=0);
  //if for any rank needed !=0 then return needed !=0
  //used in two cases:
  // - check if any rank needs to update means and std of dataset
  // - check if any rank needs to reset the index array for sampling

  int nMasters;
  MPI_Comm_size(mastersComm, &nMasters);
  if (nMasters > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &needed, 1,
                  MPI_INT, MPI_SUM, mastersComm);
  }
  return needed;
}

void Transitions::update_samples_mean(const Real alpha)
{
  if(!bTrain) return; //if not training, keep the stored values
  if(!bNormalize) return;
	int count = 0;
  vector<Real> newStd(sI.dimUsed,0), newMean(sI.dimUsed,0);

	#pragma omp parallel
	{
		//local sum and counter
		vector<Real> sum(sI.dimUsed,0), sum2(sI.dimUsed,0);
		int cnt = 0;

		#pragma omp for schedule(dynamic)
		for(int i=0; i<Set.size(); i++)
		for(const auto & t : Set[i]->tuples) {
			assert(t->s.size() == sI.dimUsed*(1+nAppended));
			cnt++;
			for (int j=0; j<sI.dimUsed; j++) {
				sum2[j] += t->s[j]*t->s[j];
				sum[j]  += t->s[j];
			}
		}

		#pragma omp critical
		{
			count += cnt;
			for (int i=0; i<sI.dimUsed; i++) {
				newMean[i] += sum[i];
				newStd[i] += sum2[i];
			}
		}
	}

  //add up gradients across nodes (masters)
  int nMasters;
  MPI_Comm_size(mastersComm, &nMasters);
  if (nMasters > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &count, 1,
                  MPI_INT, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, newMean.data(), sI.dimUsed,
                  MPI_VALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, newStd.data(), sI.dimUsed,
                  MPI_VALUE_TYPE, MPI_SUM, mastersComm);
  }

   if(count<batchSize) return;
   /*std::cout << "States stds: [";
	for (int i=0; i<sI.dimUsed; i++) {
    newStd[i] = std::sqrt((newStd[i] - newMean[i]*newMean[i]/Real(count))/Real(count));
    newStd[i] = std::max(newStd[i],1e-8);
    std[i] = std[i]*(1.-alpha) + alpha*newStd[i];
    std::cout << std[i] << " ";
  }
  std::cout << "]. States means: [";
	for (int i=0; i<sI.dimUsed; i++) {
    newMean[i] /= Real(count);
    mean[i] = mean[i]*(1.-alpha) + alpha*newMean[i];
    std::cout << mean[i] << " ";
  }
  std::cout << "]" << std::endl;*/
}

vector<Real> Transitions::standardize(const vector<Real>& state, const Real noise, const int thrID) const
{
    if(!bNormalize) return state;
    vector<Real> tmp(sI.dimUsed*(1+nAppended));
    assert(state.size() == sI.dimUsed*(1+nAppended));
    for (int j=0; j<1+nAppended; j++)
    for (int i=0; i<sI.dimUsed; i++) {
      const int k = j*sI.dimUsed + i;
      tmp[k] = (state[k] - mean[i])/(std[i]+1e-8);
      //tmp[k] = state[k]/(std[i]+1e-8);
    }

    if (noise>0) {
      assert(generators.size()>thrID);
      std::normal_distribution<Real> distn(0.,noise);
      for (int i=0; i<sI.dimUsed*(1+nAppended); i++)
        tmp[i] += distn(generators[thrID]);
    }
    return tmp;
}

void Transitions::synchronize()
{
  #if 1==0
	assert(nSequences==Set.size() && maxTotSeqNum == nSequences);
	uniform_real_distribution<Real> dis(0.,1.);
  //if (dis(*(gen->g))>0.01) {
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<Set.size(); i++) {
      Set[i]->MSE = 0.;
      #if 1
        for(const auto & t : Set[i]->tuples)
        Set[i]->MSE = std::max(Set[i]->MSE, t->SquaredError);
      #else
        unsigned count = 0;
        for(const auto & t : Set[i]->tuples) {
        //assert(t->SquaredError>0); //last one has error 0
        Set[i]->MSE += t->SquaredError;
        count++;
        }
        assert(count);
        Set[i]->MSE /= (Real)(count-1);
      #endif
      /*
      for(const auto & t : Set[i]->tuples)
      for (int i=0; i<sI.dimUsed; i++)
      Set[i]->MSE += std::pow((t->s[i] - mean[i])/std[i], 2);
      */
    }
    const auto compare=[this](Sequence* a, Sequence* b){return a->MSE<b->MSE;};
    std::sort(Set.begin(), Set.end(), compare);
    if(Set.front()->MSE > Set.back()->MSE) die("WRONG\n");
  //} else random_shuffle(Set.begin(), Set.end(), *(gen));

  iOldestSaved = 0;
  #endif

  int nTransitionsInBuf=0, nTransitionsDeleted=0, bufferSize=Buffered.size();
  for(auto & bufTransition : Buffered) {
      const int ind = iOldestSaved++;
      iOldestSaved = (iOldestSaved >= maxTotSeqNum) ? 0 : iOldestSaved;

      if (not Set[ind]->ended) --nBroken;
      nTransitionsDeleted += Set[ind]->tuples.size()-1;
      nTransitionsInBuf += bufTransition->tuples.size()-1;

      nTransitions -= Set[ind]->tuples.size()-1;
      _dispose_object(Set[ind]);

      nTransitions += bufTransition->tuples.size()-1;
      if (not bufTransition->ended) ++nBroken;
      Set[ind] = bufTransition;
  } //number of sequences remains constant
  printf("Removing %lu sequences (avg length %f) associated with small MSE"
    "error in favor of new ones (avg lendth %f)\n", Buffered.size(),
    nTransitionsDeleted/(Real)bufferSize, nTransitionsInBuf/(Real)bufferSize);
  Buffered.resize(0); //no clear?
}

void Transitions::updateSamples(const Real alpha)
{
  bool update_meanstd_needed = false;
	if(Buffered.size()>0) {
    printf("nSequences %d > maxTotSeqNum %d (nTransitions=%d, avgSeqLen=%f).\n",
        nSequences, maxTotSeqNum, nTransitions, nTransitions/(Real)nSequences);
    synchronize();
    update_meanstd_needed = true;
    old_ndata = nTransitions;
  } else {
    printf("nSequences %d < maxTotSeqNum %d (nTransitions=%d, avgSeqLen=%f).\n",
        nSequences, maxTotSeqNum, nTransitions, nTransitions/(Real)nSequences);
    const int ndata = nTransitions;
    update_meanstd_needed = ndata!=old_ndata;
    old_ndata = ndata;
  }
	update_meanstd_needed = update_meanstd_needed && alpha;
  if(syncBoolOr(update_meanstd_needed))
    update_samples_mean(alpha);

  const int ndata = (bRecurrent) ? nSequences : nTransitions;
  inds.resize(ndata);
  #if 0
  if (bRecurrent) {
    delete dist;
    assert(nSequences==Set.size());
    #ifndef NDEBUG
    int recount_Transitions = 0;
    #endif
    for(int i=0; i<nSequences; i++) {
      #ifndef NDEBUG
      recount_Transitions += Set[i]->tuples.size()-1;
      #endif
  		inds[i] = Set[i]->tuples.size()-1;
    }
    assert(recount_Transitions==nTransitions);
    dist = new discrete_distribution<int>(inds.begin(), inds.end());
  }
  else
  #endif
  {
    std::iota(inds.begin(), inds.end(), 0);
    random_shuffle(inds.begin(), inds.end(), *(gen));
  }
}

int Transitions::sample()
{
  const int ind = inds.back();
  inds.pop_back();
  #if 0
  if (bRecurrent) {
    const int sampid = dist->operator()(*(gen->g));
    //printf("Choosing %d with length %lu\n", sampid, Set[sampid]->tuples.size());
    return sampid;
  }
  else
  #endif
    return ind;
}

void Transitions::save(std::string fname)
{
    string nameBackup = fname + "_data_stats";
    FILE * f = fopen(nameBackup.c_str(), "w");
    if (f != NULL)
      for (int i=0; i<sI.dimUsed; i++)
        fprintf(f, "%9.9e %9.9e\n", mean[i], std[i]);
    fclose(f);
}

void Transitions::restart(std::string fname)
{
    string nameBackup = fname + "_data_stats";
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
      debug1("File not found %s\n", nameBackup.c_str());
      #ifndef NDEBUG //if debug, you might want to do this
      if(!bTrain) {die("...and I'm not training\n");}
      #endif
      return;
    }

    for (int i=0; i<sI.dimUsed; i++) {
      in >> mean[i] >> std[i];
      printf("Read: %9.9e %9.9e\n", mean[i], std[i]);
    }
    in.close();
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
			int k=0, back=0, indT=Set[0]->tuples.size()-1;
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
