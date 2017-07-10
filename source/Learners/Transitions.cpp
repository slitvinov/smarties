/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 */

#include "Transitions.h"
#include <dirent.h>
#include <iterator>
#include <parallel/algorithm>

Transitions::Transitions(MPI_Comm comm, Environment* const _env, Settings & _s):
	mastersComm(comm), env(_env), bNormalize(_s.bNormalize), bTrain(_s.bTrain),
	bWriteToFile(!(_s.samplesFile=="none")), bSampleSeq(_s.bSampleSequences),
	maxTotSeqNum(_s.maxTotSeqNum),maxSeqLen(_s.maxSeqLen),minSeqLen(_s.minSeqLen),
	nAppended(_s.appendedObs),batchSize(_s.batchSize),learn_rank(_s.learner_rank),
	learn_size(_s.learner_size),path(_s.samplesFile),sI(_env->sI), aI(_env->aI), generators(_s.generators)
{
	mean.resize(sI.dimUsed, 0);
	std.resize(sI.dimUsed, 1);
	invstd.resize(sI.dimUsed, 1);
	Uint k = 0;
	if (sI.mean.size())
		for (Uint i=0; i<sI.dim; i++)
			if (sI.inUse[i]) {
				mean[k] = sI.mean[i];
				std[k] = sI.scale[i];
				invstd[k] = 1./sI.scale[i];
				k++;
			}
	assert(k == sI.dimUsed);
	assert(_s.nAgents>0);
	Tmp.resize(_s.nAgents);
	for (Uint i=0; i<static_cast<Uint>(_s.nAgents); i++)
		Tmp[i] = new Sequence();

	curr_transition_id.resize(max(_s.nAgents,1));
	gen = new Gen(&generators[0]);
	Set.reserve(maxTotSeqNum);
}

Uint Transitions::restartSamples(const Uint polDim)
{
	vector<State> oldState(curr_transition_id.size(), sI);
	State newState(sI); Action action(aI, &generators[0]);
	const Uint sDim=sI.dim, aDim=aI.dim, writesize = 3+sI.dim+aI.dim+polDim;
	int agentID = 0, info = 0, sampID = 0;
	vector<Real> policy(polDim);
	Real reward = 0;
	char asciipath[256];
	sprintf(asciipath, "rank%02d_%s", learn_rank, path.c_str());
	std::ifstream in(asciipath);
	std::string line;

	if(in.good())
	{
		while (getline(in, line))
		{
			std::istringstream line_in(line);
			line_in >> agentID >> info >> sampID;
			if(oldState.size() <= static_cast<Uint>(agentID))
				 oldState.resize(1+agentID, sI);
			if(curr_transition_id.size() <= static_cast<Uint>(agentID))
				 curr_transition_id.resize(1+agentID,0);
			assert(agentID>=0);
			if((sampID==0) != (info==1))
				die("Mismatch in transition counter\n");
			if(static_cast<Uint>(sampID)!=curr_transition_id[agentID]+1 && info!=1)
				die("Mismatch in transition change\n");
			curr_transition_id[agentID] = sampID;

			for(Uint i=0; i<sDim;   i++) line_in >> newState.vals[i];
			for(Uint i=0; i<aDim;   i++) line_in >> action.vals[i];
			line_in >> reward;
			for(Uint i=0; i<polDim; i++) line_in >> policy[i];

			add(agentID, info, oldState[agentID], action, policy, newState, reward);
			if (info == 2) oldState[agentID].vals = vector<Real>(sDim, 0);
			else oldState[agentID].vals = newState.vals;
		}
		in.close();
		//_die("Job done %lu\n",Tmp.size());
	}
	else
	{
		while (true)
		{
			sprintf(asciipath, "obs_rank%02d_agent%03d.raw", learn_rank, agentID);
			FILE*pFile = fopen(asciipath, "rb");
			if(pFile == NULL) {
				printf("Couldnt open file %s.\n", asciipath);
				break;
			}
			if(oldState.size() <= static_cast<Uint>(agentID))
				 oldState.resize(1+agentID, sI);
			if(curr_transition_id.size() <= static_cast<Uint>(agentID))
				 curr_transition_id.resize(1+agentID,0);

			float* buf = (float*) malloc(writesize*sizeof(float));

			while(true)
			{
				size_t ret = fread(buf,sizeof(float),writesize,pFile);
				if (ret == 0) break;
				if (ret != writesize) _die("Error reading datafile %s", asciipath);
				int* ibuf = (int*) buf;
				info = ibuf[0]; sampID = ibuf[1];

				if((sampID==0) != (info==1))
					die("Mismatch in transition counter\n");
				if(static_cast<Uint>(sampID)!=curr_transition_id[agentID]+1 && info!=1)
					die("Mismatch in transition change\n");
				curr_transition_id[agentID] = sampID;

				Uint k = 2;
				for(Uint i=0; i<sDim;   i++) newState.vals[i] = buf[k++];
				for(Uint i=0; i<aDim;   i++) action.vals[i] = buf[k++];
				reward = buf[k++];
				for(Uint i=0; i<polDim; i++) policy[i] = buf[k++];
				assert(k == writesize);
				add(agentID, info, oldState[agentID], action, policy, newState, reward);
				if (info == 2) oldState[agentID].vals = vector<Real>(sDim, 0);
				else oldState[agentID].vals = newState.vals;
			}
			fclose(pFile);
			free(buf);
			agentID++;
		}
		if(agentID==0)  {
			printf("Couldn't restart transition data.\n");
			return 1;
		}
	}
	printf("Found %d broken chains out of %d / %d.\n",
			nBroken, nSequences, nTransitions);

	return 0;
}

int Transitions::passData(const int agentId, const int info, const State& sOld,
	const Action&aNew, const State&sNew, const Real rew, const vector<Real> muNew)
{
	assert(agentId<static_cast<int>(curr_transition_id.size()) && agentId>=0);
	const int ret = add(agentId, info, sOld, aNew, muNew, sNew, rew);
	if (ret) curr_transition_id[agentId] = 0;
	char asciipath[256];
	{
		sprintf(asciipath, "obs_rank%02d_agent%03d.raw", learn_rank, agentId);
		FILE * pFile = fopen (asciipath, "ab");
		const Uint writesize = (3 + sI.dim + aI.dim + muNew.size())*sizeof(float);
		float* buf = (float*) malloc(writesize);
    memset(buf, 0, writesize);
		int* ibuf = (int*) buf;
		ibuf[0]=info; ibuf[1]=(int)curr_transition_id[agentId];
		Uint k=2;
		for (Uint i=0; i<sI.dim;       i++) buf[k++] = (float) sNew.vals[i];
		for (Uint i=0; i<aI.dim;       i++) buf[k++] = (float) aNew.vals[i];
		buf[k++] = rew;
		for (Uint i=0; i<muNew.size(); i++) buf[k++] = (float) muNew[i];
		assert(k*sizeof(float) == writesize);
		fwrite (buf, sizeof(float), writesize/sizeof(float), pFile);
		fflush(pFile);
		fclose(pFile);
		free(buf);
	}

	if (bWriteToFile)
	{
		sprintf(asciipath, "rank%02d_%s", learn_rank, path.c_str());
		ofstream fout(asciipath, ios::app);
		fout<<agentId<<" "<<info<<" "<<curr_transition_id[agentId]
			<<" "<<sNew._print().c_str()<<" "<<aNew._print().c_str()
			<<" "<<rew<<" "<<print(muNew)<<endl;
		fout.flush();
		fout.close();
	}

	curr_transition_id[agentId]++;
	return ret;
}

int Transitions::add(const int agentId, const int info, const State& sOld,
		const Action& aNew, const vector<Real> muNew, const State& sNew, Real rNew)
{
	//return value is 1 if the agent states buffer is empty or on initial state
	int ret = 0;
	while(static_cast<Uint>(agentId)>=Tmp.size()) Tmp.push_back(new Sequence());

	const Uint sApp = nAppended*sI.dimUsed;
	if (Tmp[agentId]->tuples.size()!=0 && info == 1) {
		printf("Detected partial sequence\n");
		push_back(agentId); //create new sequence
		ret = 1;
	} else if(Tmp[agentId]->tuples.size()==0) {
		if(info!=1) die("Missing initial state\n");
		ret = 1; //new Sequence
	}
	/*
		if(Tmp[agentId]->tuples.size()!=0) {
			const Tuple*const last = Tmp[agentId]->tuples.back();
			printf("Continue %d[%s]=[%s][%s][%s],%g\n",agentId,sOld._print().c_str(),
			print(last->s).c_str(),aNew._print().c_str(),sNew._print().c_str(),rNew);
		}
		else
			printf("Start chain %d[%s][%s][%s],%g\n",agentId, sOld._print().c_str(),
			aNew._print().c_str(),sNew._print().c_str(),rNew);
	*/
	if(Tmp[agentId]->tuples.size()!=0) {
		bool same = true;
		const vector<Real> vecSold = sOld.copy_observed();
		const Tuple*const last = Tmp[agentId]->tuples.back();
		for (Uint i=0; i<sI.dimUsed; i++) //scaled vec only has used dims:
			same = same && std::fabs(last->s[i]-vecSold[i])<1e-8;
		if (!same) {
			printf("Detected partial sequence\n");
			push_back(agentId); //create new sequence
			ret = 1;
		}
	}

	if (Tmp[agentId]->tuples.size() >= maxSeqLen) {
		//upper limit to how long a sequence can be
		//printf("Sequence is too long!\n");
		const Tuple* const l = Tmp[agentId]->tuples.back();
		Tuple * t = new Tuple(); //backup last state
		t->s = l->s; t->a = l->a; t->r = l->r, t->mu = l->mu;
		t->SquaredError = l->SquaredError;
		#ifdef importanceSampling
			t->weight = l->weight;
		#endif
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
	t->mu = muNew;

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
		nSeenSequences++;
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

void Transitions::update_samples_mean(const Real alpha)
{
	if(!bTrain || !bNormalize) return; //if not training, keep the stored values

	long double count = 0;
	vector<long double> newStd(sI.dimUsed,0), newMean(sI.dimUsed,0);

	#pragma omp parallel
	{
		//local sum and counter
		vector<long double> sum(sI.dimUsed,0), sum2(sI.dimUsed,0);
		Uint cnt = 0;

		#pragma omp for schedule(dynamic)
		for(Uint i=0; i<Set.size(); i++)
			for(const auto & t : Set[i]->tuples) {
				assert(t->s.size() == sI.dimUsed*(1+nAppended));
				cnt++;
				for (Uint j=0; j<sI.dimUsed; j++) {
					sum2[j] += t->s[j]*t->s[j];
					sum[j]  += t->s[j];
				}
			}

		#pragma omp critical
		{
			count += cnt;
			for (Uint i=0; i<sI.dimUsed; i++) {
				newMean[i] += sum[i];
				newStd[i] += sum2[i];
			}
		}
	}

	//add up gradients across nodes (masters)
	if (learn_size > 1) {
		MPI_Allreduce(MPI_IN_PLACE, &count, 1,
				MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
		MPI_Allreduce(MPI_IN_PLACE, newMean.data(), sI.dimUsed,
				MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
		MPI_Allreduce(MPI_IN_PLACE, newStd.data(), sI.dimUsed,
				MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
	}

	if(count<batchSize) return;
	for (Uint i=0; i<sI.dimUsed; i++) {
		newStd[i] = std::sqrt((newStd[i]-newMean[i]*newMean[i]/count)/count);
		newStd[i] = std::max(newStd[i],(long double)1e-8);
		newMean[i] /= count;
	}

	if (sI.mean.size()) {
		Uint k=0;
		for (Uint i=0; i<sI.dim; i++)
			if (sI.inUse[i]) {
				mean[k] = sI.mean[i]*(1-alpha) + alpha*newMean[k];
				std[k] = sI.scale[i]*(1-alpha) + alpha*newStd[k];
				invstd[k] = 1./(std[i]+1e-8);
				k++;
			}
		assert(k==sI.dimUsed);
	} else {
		for (Uint i=0; i<sI.dimUsed; i++) {
			mean[i] = mean[i]*(1.-alpha) + alpha*newMean[i];
			std[i] = std[i]*(1.-alpha) + alpha*newStd[i];
			invstd[i] = 1./(std[i]+1e-8);
		}
	}
}

void Transitions::sortSequences()
{
	assert(nSequences==Set.size() && maxTotSeqNum == nSequences);
	//uniform_real_distribution<Real> dis(0.,1.);
	//if (dis(*(gen->g))>0.01) { //small chance to shuffle
	#pragma omp parallel for schedule(dynamic)
	for(Uint i=0; i<Set.size(); i++) {
		Set[i]->MSE = 0.;
		#if 1 //sort by max error
				for(const auto & t : Set[i]->tuples)
					Set[i]->MSE = std::max(Set[i]->MSE, t->SquaredError);
		#else //sort by mean error: penalizes long sequences
				unsigned count = 0;
				for(const auto & t : Set[i]->tuples) {
					//assert(t->SquaredError>0); //last one has error 0
					Set[i]->MSE += t->SquaredError;
					count++;
				}
				assert(count);
				Set[i]->MSE /= (Real)(count-1);
		#endif
		/* //sort by distance from mean of observations statistics?
      for(const auto & t : Set[i]->tuples)
      for (int i=0; i<sI.dimUsed; i++)
      Set[i]->MSE += std::pow((t->s[i] - mean[i])/std[i], 2);
		 */
	}
	const auto compare=[this](Sequence* a, Sequence* b) {
		return a->MSE==0 ? false : (b->MSE==0 ? true : (a->MSE<b->MSE) );
	};
	__gnu_parallel::sort(Set.begin(), Set.end(), compare);
	assert(Set.front()->MSE < Set.back()->MSE || Set.back()->MSE == 0);
	//for(Uint i=0; i<Set.size(); i++) printf("%u %f\n",i,Set[i]->MSE);
	//} else random_shuffle(Set.begin(), Set.end(), *(gen));
	iOldestSaved = 0;
}

void Transitions::synchronize()
{
	#ifdef RESORT_SEQS
	sortSequences();
	#endif

	Uint cnt =0;
	Uint nTransitionsInBuf=0, nTransitionsDeleted=0, bufferSize=Buffered.size();
	//  for(auto & bufTransition : Buffered) {
	if(!bufferSize) return;
	for(Uint j=bufferSize; j>0; j--) {
		cnt++;
		//auto bufTransition = Buffered[i];
		assert(Buffered.size() == j);
		auto bufTransition = Buffered.back();
		const Uint ind = iOldestSaved++;
		iOldestSaved = (iOldestSaved >= maxTotSeqNum) ? 0 : iOldestSaved;

		if (not Set[ind]->ended) {
			if(nBroken==0) die("Error in nBroken counter.\n");
			--nBroken;
		}
		nTransitionsDeleted += Set[ind]->tuples.size()-1;
		nTransitionsInBuf += bufTransition->tuples.size()-1;

		nTransitions -= Set[ind]->tuples.size()-1;
		_dispose_object(Set[ind]);

		nTransitions += bufTransition->tuples.size()-1;
		if (not bufTransition->ended) ++nBroken;
		Set[ind] = bufTransition;
		Buffered.pop_back();
		if(cnt == maxTotSeqNum/20) break; //don't change buffer too much
	} //number of sequences remains constant
	printf("Removing %d sequences (avg length %f) associated with small MSE"
			"error in favor of new ones (avg lendth %f). %lu left in Buffer\n",
			cnt, nTransitionsDeleted/(Real)cnt,
			nTransitionsInBuf/(Real)cnt, Buffered.size());
}

Uint Transitions::updateSamples(const Real annealFac)
{
	bool update_meanstd_needed = false;
	if(Buffered.size()>0) {
		if(!learn_rank)
		printf("nSequences %d > maxTotSeqNum %d (nTransitions=%d, avgSeqLen=%f).\n",
			nSequences, maxTotSeqNum, nTransitions, nTransitions/(Real)nSequences);
		synchronize();
		update_meanstd_needed = true;
		old_ndata = nTransitions;
	} else {
		if(!learn_rank)
		printf("nSequences %d < maxTotSeqNum %d (nTransitions=%d, avgSeqLen=%f).\n",
			nSequences, maxTotSeqNum, nTransitions, nTransitions/(Real)nSequences);
		update_meanstd_needed = nTransitions!=old_ndata;
		old_ndata = nTransitions;
	}
	update_meanstd_needed = update_meanstd_needed && bNormalize && annealFac>0;

	#ifndef importanceSampling
		const Uint ndata = (bSampleSeq) ? nSequences : nTransitions;
		inds.resize(ndata);
		std::iota(inds.begin(), inds.end(), 0);
		__gnu_parallel::random_shuffle(inds.begin(), inds.end(), *(gen));
	#else
		updateP();
	#endif

	return update_meanstd_needed ? 1 : 0;
}

Uint Transitions::sample(const int thrID)
{
	#ifndef importanceSampling
		const Uint ind = inds.back();
		inds.pop_back();
	#else
		const Uint ind = (*dist)(generators[thrID]);
	#endif

	return ind;
}

void Transitions::save(std::string fname)
{
	if(learn_rank) return;
	string nameBackup = fname + "_data_stats";
	FILE * f = fopen(nameBackup.c_str(), "w");
	if (f != NULL)
		for (Uint i=0; i<sI.dimUsed; i++)
			fprintf(f, "%9.9e %9.9e\n", mean[i], std[i]);
	fclose(f);
}

void Transitions::restart(std::string fname)
{
	string nameBackup = fname + "_data_stats";
	ifstream in(nameBackup.c_str());
	debugT("Reading from %s\n", nameBackup.c_str());
	if (!in.good()) {
		debugT("File not found %s\n", nameBackup.c_str());
		#ifndef NDEBUG //if debug, you might want to do this
			if(!bTrain) {die("...and I'm not training\n");}
		#endif
		return;
	}

	for (Uint i=0; i<sI.dimUsed; i++) {
		in >> mean[i] >> std[i];
		printf("Read: %9.9e %9.9e\n", mean[i], std[i]);
	}
	in.close();
}

#ifdef importanceSampling
//Sample sequences: same procedure with importance weights computed from maximun error?
void Transitions::updateP()
{
	const Uint ndata = bSampleSeq ? nSequences : nTransitions;
	inds.resize(ndata);
	std::iota(inds.begin(), inds.end(), 0);
	vector<Real> errors(ndata), Ps(ndata), Ws(ndata);

	{
		Uint k = 0;
		for(Uint i=0; i<Set.size(); i++)
		{
			Real maxerr = 0;
			for(Uint j=0; j<Set[i]->tuples.size()-1; j++)
			{
				if(bSampleSeq) //sample based on max error of the sequence
					maxerr = std::max(maxerr,Set[i]->tuples[j]->SquaredError);
				else //sample based on transition's last error
					errors[k++] = Set[i]->tuples[j]->SquaredError;
			}
			if(bSampleSeq) errors[k++] = maxerr;
		}
		assert(k==ndata);
	}

	const auto comp=[&](const Uint a,const Uint b) {return errors[a]>errors[b];};
	__gnu_parallel::sort(inds.begin(), inds.end(), comp);
	assert(errors[inds.front()] >= errors[inds.back()]);

	//sort in decreasing order of the error. Points with zero error
	//(which means that they are not yet processed)
	//are put at the top:
	#pragma omp parallel for
	for(Uint i=0; i<ndata; i++)
		Ps[inds[i]] = errors[inds[i]]>0 ? std::sqrt(1./(i+1.)) : 1;

	//const Real minP = Ps[inds.back()];
	//const Real sumP = __gnu_parallel::accumulate(Ps.begin(), Ps.end(), 0);
	Real minP = 2, sumP = 0;
	#pragma omp parallel for reduction(min: minP) reduction(+: sumP)
	for(Uint i=0; i<ndata; i++) {
		minP = std::min(minP, Ps[i]);
		sumP += Ps[i];
	}
	assert(minP<=1 && sumP>0);

	#pragma omp parallel for
	for(Uint i=0; i<ndata; i++) {
		Ws[i] = minP/Ps[i];
		Ps[i] = Ps[i]/sumP;
	}

	if(dist not_eq nullptr) delete dist;
	dist = new std::discrete_distribution<Uint>(Ps.begin(), Ps.end());

	{
		Uint k = 0;
		for(Uint i=0; i<Set.size(); i++)
		{
			for(Uint j=0; j<Set[i]->tuples.size()-1; j++)
			{
				if(bSampleSeq) //sample based on max error of the sequence
					Set[i]->tuples[j]->weight = Ws[k];
				else //sample based on transition's last error
					Set[i]->tuples[j]->weight = Ws[k++];
			}
			if(bSampleSeq) k++;
		}
		assert(k==ndata);
	}
}
#endif
