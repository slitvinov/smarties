/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner.h"
#include <chrono>

Learner::Learner(MPI_Comm comm, Environment*const _env, Settings & _s) :
mastersComm(comm), env(_env), tgtUpdateDelay((Uint)_s.targetDelay),
nAgents(_s.nAgents), batchSize(_s.batchSize), nThreads(_s.nThreads),
nAppended(_s.appendedObs), maxTotSeqNum(_s.maxTotSeqNum),
totNumSteps(_s.totNumSteps), learn_rank(_s.learner_rank),
learn_size(_s.learner_size), nInputs(_s.nnInputs), nOutputs(_s.nnOutputs),
bRecurrent(_s.bRecurrent), bSampleSequences(_s.bSampleSequences),
bTrain(_s.bTrain), tgtUpdateAlpha(_s.targetDelay), greedyEps(_s.greedyEps),
gamma(_s.gamma), epsAnneal(_s.epsAnneal), obsPerStep(_s.obsPerStep),
aInfo(env->aI), sInfo(env->sI), gen(&_s.generators[0])
{
	assert(nThreads>0);
	profiler = new Profiler();
	data = new Transitions(mastersComm, env, _s);
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
	data->clearFailedSim(agentOne, agentEnd);
}

void Learner::pushBackEndedSim(const int agentOne, const int agentEnd)
{
	data->pushBackEndedSim(agentOne, agentEnd);
}

void Learner::run(Master* const master)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	vector<Uint> seq(batchSize), samp(batchSize);
	Uint ndata = (bSampleSequences) ? data->nSequences : data->nTransitions;

	int done = 0;
#pragma omp parallel num_threads(nThreads)
#pragma omp master
	if (ndata <= 10*batchSize || !bTrain) {
		if(nAgents<1) die("Learner::run nAgents<1. Nothing to do.\n");
		done = master->run();
	}
	if(done) return;
	
	while (opt->nepoch < totNumSteps) {
		assert(taskCounter == 0);
		//const Real annealFac = annealingFactor();
		ndata = (bSampleSequences) ? data->nSequences : data->nTransitions;

		profiler->push_start("SRT");
		//Uint syncDataStats = 0;
		if(opt->nepoch%100==0 || data->requestUpdateSamples())
			data->updateSamples(0); //update sampling //syncDataStats =

		#ifdef __CHECK_DIFF //check gradients with finite differences
			if (opt->nepoch % 100000 == 0) net->checkGrads();
		#endif

		//CODE TO DO ONLINE UPDATE OF DATA MEAN/STD: unused
		//if (learn_size > 1) {
		//	MPI_Allreduce(MPI_IN_PLACE, &syncDataStats, 1,
		//			MPI_UNSIGNED, MPI_SUM, mastersComm);
		//}
		//if(syncDataStats) data->update_samples_mean(0); //annealFac

		start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(nThreads)
#pragma omp master
		{
			profiler->stop_start("SMP");
			nAddedGradients = bSampleSequences ? sampleSequences(seq) :
				sampleTransitions(seq,samp);
#pragma omp flush
			profiler->stop_start("TSK");

			if(bSampleSequences) {//we are using an LSTM: do BPTT
				for (Uint i=0; i<batchSize; i++) {
					const Uint sequence = seq[i];
#pragma omp task firstprivate(sequence)
					{
						const int thrID = omp_get_thread_num();
						assert(thrID>=0);
						Train_BPTT(sequence, static_cast<Uint>(thrID));
#pragma omp atomic
						taskCounter++;
					}
				}
			} else {
				for (Uint i=0; i<batchSize; i++) {
					const Uint sequence = seq[i];
					const Uint transition = samp[i];
#pragma omp task firstprivate(sequence,transition)
					{
						const int thrID = omp_get_thread_num();
						assert(thrID>=0);
						Train(sequence, transition, static_cast<Uint>(thrID));
#pragma omp atomic
						taskCounter++;
					}
				}
			}
			if(nAgents>0) master->run(); //master goes to communicate with slaves
		}

		assert(nAddedGradients);
		end = std::chrono::high_resolution_clock::now();
		sumElapsed+= std::chrono::duration<Real>(end-start).count()/nAddedGradients;
		dataUsage += nAddedGradients;
		countElapsed++;
		batchUsage++;

		assert(taskCounter == batchSize);
		profiler->stop_start("UPW");
		stackAndUpdateNNWeights();
		updateTargetNetwork();
		if(opt->nepoch%100 ==0) processStats();
		taskCounter = 0;
		profiler->stop_start("DAT");
#pragma omp parallel num_threads(nThreads)
#pragma omp master
		master->run(); //master goes back to comm till enough data is gathered
		profiler->pop_stop();

		if(opt->nepoch%1000==0 && !learn_rank) profiler->printSummary();
	}
}

Uint Learner::sampleTransitions(vector<Uint>& sequences, vector<Uint>& transitions)
{
	assert(sequences.size() == batchSize && transitions.size() == batchSize);
	assert(!bSampleSequences);
	vector<Uint> load(batchSize), sorting(batchSize), s(batchSize), t(batchSize);
	for (Uint i=0; i<batchSize; i++) {
		const Uint ind = data->sample();

		Uint k=0, back=0, indT=data->Set[0]->tuples.size()-1;
		while (ind >= indT) {
			back = indT;
			indT += data->Set[++k]->tuples.size()-1;
		}

		s[i] = k;
		t[i] = ind-back;
		sorting[i] = i;
		//work per transition (applies to algos with off policy corrections):
		load[i] = data->Set[k]->tuples.size()-1 - t[i];
		//load[i] = data->Set[k]->tuples.size()-1; // ~ this would be for RNN
	}

	//sort elements of sorting according to load for each transition:
	const auto compare = [&] (Uint a, Uint b) { return load[a] > load[b]; };
	std::sort(sorting.begin(), sorting.end(), compare);
	assert(load[sorting[0]] >= load[sorting[batchSize-1]]);
	//sort vectors passed to learning algo:
	for (Uint i=0; i<batchSize; i++) {
		transitions[i] = t[sorting[i]];
		sequences[i] = s[sorting[i]];
	}
	return batchSize; //always add one grad per transition
}

Uint Learner::sampleSequences(vector<Uint>& sequences)
{
	assert(sequences.size() == batchSize && bSampleSequences);
	Uint _nAddedGradients = 0;
	for (Uint i=0; i<batchSize; i++)
	{
		const Uint ind = data->sample();
		sequences[i]  = ind;
		//index[i] = ind;
		const Uint seqSize = data->Set[ind]->tuples.size();
		//to normalize mean gradient for update:
		_nAddedGradients += seqSize-1; //last state = terminal, no next reward
	}
	//sort them such that longer ones are started first, reducing overhead!
	const auto compare = [this] (Uint a, Uint b) {
		return data->Set[a]->tuples.size() > data->Set[b]->tuples.size();
	};
	std::sort(sequences.begin(), sequences.end(), compare);

	return _nAddedGradients;
}

bool Learner::checkBatch(unsigned long mastersNiter)
{
	const Uint ndata = bSampleSequences ? data->nSequences : data->nTransitions;
	//if there is noty enough data for training: go back to master
	if (ndata < batchSize*10 || !bTrain) {
		nData_b4PolUpdates = data->nSeenSequences;
		return false;
	}

	//if threads finished processing data, unblock other ranks by adding up grad
	if(taskCounter >= batchSize) return true;
	//otherwise, either train or communicate depending on number of gradient
	// steps to number of observed sequences requested by user
	return data->nSeenSequences - nData_b4PolUpdates > obsPerStep * opt->nepoch;
}

void Learner::save(string name)
{
	if (!learn_rank) {
		opt->save(name);
		data->save(name);

		const string stuff = name+".status";
		FILE * f = fopen(stuff.c_str(), "w");
		if (f == NULL) die("Save fail\n");
		fprintf(f, "policy iter: %d\n", data->anneal);
		fprintf(f, "optimize epoch: %lu\n", opt->nepoch);
		//fprintf(f, "epoch count: %lu\n", stats.epochCount);
		fprintf(f, "nData_b4PolUpdates: %lu\n", nData_b4PolUpdates);
		fclose(f);
	}
}

void Learner::restart(string name)
{
	int masterRank;
	MPI_Comm_rank(mastersComm, &masterRank);
	printf("Restarting from saved policy...\n");

	data->restartSamples(policyVecDim);
	if (name == "none") return;

	if ( opt->restart(name) )
		printf("Restart successful, moving on...\n");
	else
		printf("Not all policies restarted. \n");
	data->restart(name);

	const string stuff = name+".status";
	FILE * f = fopen(stuff.c_str(), "r");
	if(f != NULL) {
		{
			int val=-1;
			fscanf(f, "policy iter: %d\n", &val);
			if(val>=0) data->anneal = val;
			printf("policy iter: %d\n", data->anneal);
		}{
			long unsigned ret = 0;
			fscanf(f, "optimize epoch: %lu\n", &ret);
			if(ret>0) opt->nepoch = ret;
			printf("optimize epoch: %lu\n", opt->nepoch);
		//}{
		//	long unsigned ret = 0;
		//	fscanf(f, "epoch count: %lu\n", &ret);
		//	stats.epochCount = ret;
		//	printf("epoch count: %lu\n", stats.epochCount);
		}{
			long unsigned ret = 0;
			fscanf(f, "nData_b4PolUpdates: %lu\n", &ret);
			nData_b4PolUpdates = ret;
			printf("nData_b4PolUpdates: %lu\n", nData_b4PolUpdates);
		}
		fclose(f);
	}
	else printf("No status\n");
	save("restarted_policy");
}
