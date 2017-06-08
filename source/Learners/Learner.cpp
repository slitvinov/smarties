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

Learner::Learner(MPI_Comm comm, Environment*const _env, Settings & settings) :
mastersComm(comm), env(_env), nAgents(settings.nAgents),
batchSize(settings.batchSize), tgtUpdateDelay((Uint)settings.targetDelay),
nThreads(settings.nThreads), nInputs(settings.nnInputs),
nOutputs(settings.nnOutputs), nAppended(settings.appendedObs),
bRecurrent(settings.bRecurrent), bTrain(settings.bTrain),
tgtUpdateAlpha(settings.targetDelay), gamma(settings.gamma),
greedyEps(settings.greedyEps), epsAnneal(settings.epsAnneal),
taskCounter(batchSize), aInfo(env->aI), sInfo(env->sI),
gen(&settings.generators[0])
{
	assert(nThreads>0);
	profiler = new Profiler();
	data = new Transitions(mastersComm, env, settings);
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
	data->clearFailedSim(agentOne, agentEnd);
}

void Learner::pushBackEndedSim(const int agentOne, const int agentEnd)
{
	data->pushBackEndedSim(agentOne, agentEnd);
}

void Learner::TrainBatch()
{
	const Uint ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
	vector<Uint> seq(batchSize), samp(batchSize);
	if (ndata<batchSize) return; //do we have enough data?
	if (!bTrain) return; //are we training?
	Uint nAddedGradients=0;

	if(data->syncBoolOr(data->inds.size()<batchSize))
	{ //uniform sampling
		data->updateSamples();
		processStats( 0 ); //dump info about convergence
#ifdef __CHECK_DIFF //check gradients with finite differences, just for debug
		if (opt->nepoch % 100000 == 0) net->checkGrads();
#endif
	}

	if(bRecurrent) {
		nAddedGradients = sampleSequences(seq);
		for (Uint i=0; i<batchSize; i++)
			Train_BPTT(seq[i]);
	} else {
		nAddedGradients = sampleTransitions(seq, samp);
		for (Uint i=0; i<batchSize; i++)
			Train(seq[i], samp[i]);
	}

	batchUsage += 1;
	dataUsage += nAddedGradients;

	updateNNWeights(nAddedGradients);
	updateTargetNetwork();
}

void Learner::TrainTasking(Master* const master)
{
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	vector<Uint> seq(batchSize), samp(batchSize);//, index(batchSize);
	Uint nAddedGradients = 0, countElapsed = 0;
	Real sumElapsed = 0;
	Uint ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
	if (ndata <= 10*batchSize || !bTrain) {
		if(nAgents<1) die("Nothing to do, nowhere to go.\n");
		master->run();
	}

	while (true) {
		ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
		taskCounter = 0;
		nAddedGradients = 0;

		if(data->syncBoolOr(data->inds.size()<batchSize))
		{ //reset sampling
			data->updateSamples();
			processStats(sumElapsed/countElapsed); //dump info about convergence
			sumElapsed = 0; countElapsed=0;
			//print_memory_usage();
#ifdef __CHECK_DIFF //check gradients with finite differences
			if (opt->nepoch % 100000 == 0) net->checkGrads();
#endif
		}
		start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(nThreads)
#pragma omp master
		{
			if(bRecurrent) {//we are using an LSTM: do BPTT
				nAddedGradients = sampleSequences(seq);
#pragma omp flush

				for (Uint i=0; i<batchSize; i++) {
					const Uint sequence = seq[i];
#pragma omp task firstprivate(sequence)
					{
						const int thrID = omp_get_thread_num();
						assert(thrID>=0);
						//#ifndef NDEBUG
						//printf("Thread %d to %d\n",thrID,sequence);
						//fflush(0);
						//#endif
						Train_BPTT(sequence, static_cast<Uint>(thrID));

#pragma omp atomic
						taskCounter++;
					}
				}
			} else {
				nAddedGradients = sampleTransitions(seq, samp);
#pragma omp flush

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

			//TODO: can add task to update sampling probabilities for prioritized exp replay
			if(nAgents>0) master->run(); //master goes to communicate with slaves
		}

		end = std::chrono::high_resolution_clock::now();
		const Real len = std::chrono::duration<Real>(end-start).count();
		sumElapsed += len/nAddedGradients;
		countElapsed++;

		batchUsage += 1;
		dataUsage += nAddedGradients;
		//this needs to be compatible with multiple servers
		stackAndUpdateNNWeights(nAddedGradients);
		// this can be handled node wise
		updateTargetNetwork();
	}
}

Uint Learner::sampleTransitions(vector<Uint>& sequences, vector<Uint>& transitions)
{
	assert(sequences.size() == batchSize && transitions.size() == batchSize);
	assert(!bRecurrent);
	vector<Uint> load(batchSize), sorting(batchSize), s(batchSize), t(batchSize);
	for (Uint i=0; i<batchSize; i++)
	{
		const Uint ind = data->sample();

		Uint k=0, back=0, indT=data->Set[0]->tuples.size()-1;
		while (ind >= indT) {
			back = indT;
			indT += data->Set[++k]->tuples.size()-1;
		}

		s[i] = k;
		t[i] = ind-back;
		sorting[i] = i;
		load[i] = data->Set[k]->tuples.size()-1 - t[i];
		//load[i] = data->Set[k]->tuples.size()-1;
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
	assert(sequences.size() == batchSize && bRecurrent);
	Uint nAddedGradients = 0;
	for (Uint i=0; i<batchSize; i++)
	{
		const Uint ind = data->sample();
		sequences[i]  = ind;
		//index[i] = ind;
		const Uint seqSize = data->Set[ind]->tuples.size();
		//to normalize mean gradient for update:
		nAddedGradients += seqSize-1; //last state = terminal, no next reward
	}
	//sort them such that longer ones are started first, reducing overhead!
	const auto compare = [this] (Uint a, Uint b) {
		return data->Set[a]->tuples.size() > data->Set[b]->tuples.size();
	};
	std::sort(sequences.begin(), sequences.end(), compare);

	return nAddedGradients;
}

bool Learner::checkBatch(unsigned long mastersNiter)
{
	const unsigned long dataNiter = bRecurrent ? data->nSeenSequences : mastersNiter;
	const Uint ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
	if (ndata<batchSize*10 || !bTrain) {
		mastersNiter_b4PolUpdates = dataNiter;
		return false;
	}  //do we have enough data? TODO k*ndata?

	//Constraint on over-using stale data too much
	if(epochCounter>data->nSeenSequences) return false;//dataUsage>mastersNiter||

	//if we are using a cheap to simulate env, we want to prioritize networks
	//if optimizer has done less updates than master has done communications
	// ratio is 1 : 1 in DQN paper
	//then let master thread go to help other threads finish the batch
	//const long unsigned learnerNiter = opt->nepoch + mastersNiter_b4PolUpdates;
	//if (env->cheaperThanNetwork && dataNiter > learnerNiter) return true;

	//If the transition buffer is already backed up, train and pause communicating
	if(data->Buffered.size() >= data->maxTotSeqNum/20)
		return true;

	return taskCounter >= batchSize;
}

void Learner::save(string name)
{
	int masterRank;
	MPI_Comm_rank(mastersComm, &masterRank);
	//    net->save(name);
	if (!masterRank) {
		opt->save(name);
		data->save(name);

		const string stuff = name+".status";
		FILE * f = fopen(stuff.c_str(), "w");
		if (f == NULL) die("Save fail\n");
		fprintf(f, "policy iter: %d\n", data->anneal);
		fprintf(f, "optimize epoch: %lu\n", opt->nepoch);
		//fprintf(f, "epoch count: %d\n", stats.epochCount);
		fclose(f);
	}
}

void Learner::restart(string name)
{
	int masterRank;
	MPI_Comm_rank(mastersComm, &masterRank);
	printf("Restarting from saved policy...\n");

	data->restartSamples();
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
		}{
			//int val=-1;
			//fscanf(f, "epoch count: %d\n", &val);
			//if(val>=0) stats.epochCount = val;
			//printf("epoch count: %d\n", stats.epochCount);
		}
		fclose(f);
	}
	else printf("No status\n");
	save("restarted_policy");
}

void Learner::dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
		const vector<Uint>& nbins) {}
