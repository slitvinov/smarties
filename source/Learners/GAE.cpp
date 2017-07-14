/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "GAE.h"

GAE::GAE(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm, _env, settings, settings.nnOutputs), nA(_env->aI.dim), generators(settings.generators)
{
	vector<Real> out_weight_inits = {-1, settings.outWeightsPrefac, -1};
	buildNetwork(net, opt, net_outputs, settings, out_weight_inits);
	printf("GAE: Built network with outputs: %s %s\n",
		print(net_indices).c_str(),print(net_outputs).c_str());
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
}

void GAE::buildNetwork(Network*& _net , Optimizer*& _opt, const vector<Uint> nouts, Settings& settings, vector<Real> weightInitFac, const vector<Uint> addedInputs)
{
	const string netType = settings.nnType, funcType = settings.nnFunc;
	const vector<int> lsize = settings.readNetSettingsSize();
	assert(nouts.size()>0);

	if(!weightInitFac.size()) weightInitFac.resize(nouts.size(),-1);
	if(weightInitFac.size()!=nouts.size()) die("Err in output weights factors size\n");

	Builder build(settings);
	//check if environment wants a particular network structure
	if(not env->predefinedNetwork(&build)) build.addInput(nInputs);

	Uint nsplit = min(static_cast<size_t>(settings.splitLayers),lsize.size());
	for(Uint i=0; i<lsize.size()-nsplit; i++)
		build.addLayer(lsize[i], netType, funcType);

	const Uint firstSplit = lsize.size()-nsplit;
	const vector<int> lastJointLayer = vector<int>{build.getLastLayerID()};

	if(nsplit) {
#ifdef INTEGRATEANDFIREMODEL
		die("GAE: nsplit with INTEGRATEANDFIREMODEL\n");
#endif
		for (Uint i=0; i<nouts.size(); i++)
		{
			build.addLayer(lsize[firstSplit], netType, funcType, lastJointLayer);
			for (Uint j=firstSplit+1; j<lsize.size(); j++)
				build.addLayer(lsize[j], netType, funcType);
			build.addOutput(static_cast<int>(nouts[i]) , "FFNN", weightInitFac[i]);
		}
	} else {
#ifndef INTEGRATEANDFIREMODEL
		const int sum =static_cast<int>(accumulate(nouts.begin(),nouts.end(),0));
		const Real fac=*max_element(weightInitFac.begin(),weightInitFac.end());
		build.addOutput(sum, "FFNN", lastJointLayer, fac);
		assert(fac<=1.);
#else
		build.addOutput(1, "FFNN", lastJointLayer, -1.);
		build.addOutput(nA,"IntegrateFire","Sigm",lastJointLayer,weightInitFac[1]);
	#ifdef INTEGRATEANDFIRESHARED
		build.addParamLayer(1, "Linear", 1);
	#else
		build.addParamLayer(nA, "Linear", 1);
	#endif
#endif
	}

	_net = build.build();
	if(learn_size>1) {
		MPI_Bcast(_net->weights,_net->getnWeights(),MPI_NNVALUE_TYPE,0,mastersComm);
		MPI_Bcast(_net->biases, _net->getnBiases(), MPI_NNVALUE_TYPE,0,mastersComm);
	}

	_net->updateFrozenWeights();
	_opt = new AdamOptimizer(_net, profiler, settings);
if (!learn_rank)
	_opt->save("initial");
#ifndef NDEBUG
	MPI_Barrier(mastersComm);
	_opt->restart("initial");
	_opt->save("restarted"+to_string(learn_rank));
#endif
}

void GAE::select(const int agentId, const Agent& agent)
{
	if(agent.Status==2) { //no need for action, just pass terminal s & r
		data->passData(agentId,agent,vector<Real>(policyVecDim,0)); return; }

	vector<Real> output = output_stochastic_policy(agentId, agent);
	assert(output.size() == nOutputs);

	const Lognormal_policy pol = prepare_policy(output);
	vector<Real> beta_mean=pol.getMean(), beta_std=pol.getStdev(), beta(2*nA,0);

	for(Uint i=0; i<nA; i++) {
		beta[i] = beta_mean[i];
		beta[nA+i] = beta_std[i];
		std::lognormal_distribution<Real> dist_cur(beta_mean[i], beta_std[i]);
		agent.a->vals[i] = bTrain ? dist_cur(*gen) : beta_mean[i];
	}

	data->passData(agentId, agent, beta);
	dumpNetworkInfo(agentId);
}
/*
void GAE::run(Master* const master)
{
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
		master->run(); //master goes back to comm till enough data is gathered
		profiler->pop_stop();

		if(opt->nepoch%1000==0 && !learn_rank) profiler->printSummary();
	}
}
*/
void GAE::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	die("ERROR GAE::Train on sequences rather than samples.\n");
}

void GAE::Train_BPTT(const Uint seq, const Uint thrID) const
{
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Activation*> series = net->allocateUnrolledActivations(ndata-1);

	for (Uint k=0; k<ndata-1; k++) {
		const Tuple * const _t = data->Set[seq]->tuples[k]; // s, a, mu
		net->seqPredict_inputs(data->standardize(_t->s), series[k]);
	}
	net->seqPredict_execute(series, series);

	Real A_GAE = 0, Vnext = 0, V_MC = 0;
	//if partial sequence then compute value of last state (!= R_end)
	if(not data->Set[seq]->ended) {
		series.push_back(net->allocateActivation());
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		vector<Real> out_T(nOutputs, 0);
		net->predict(data->standardize(_t->s), out_T, series, ndata-1);
		Vnext = out_T[net_indices[0]];
		delete series.back();
		series.pop_back();
	}

	for (int k=static_cast<int>(ndata)-2; k>=0; k--)
	{
		vector<Real> out = net->getOutputs(series[k]);
		vector<Real> grad = compute(seq, k, A_GAE, Vnext, V_MC, out, thrID);

		//write gradient onto output layer:
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
		clip_gradient(grad, stdGrad[0], seq, k);
		net->setOutputDeltas(grad, series[k]);
	}

	if (thrID==0) net->backProp(series, net->grad);
	else net->backProp(series, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series);
}
