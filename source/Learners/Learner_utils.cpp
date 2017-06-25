/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 * TODO:
 	- define a virtual func to remove aux tasks from output
  - fix dump policy to be more general
 */

#include "Learner_utils.h"
#include "../Math/Utils.h"

void Learner_utils::stackAndUpdateNNWeights(const Uint nAddedGradients)
{
	assert(bTrain);
	opt->nepoch++;
	//add up gradients across threads
	opt->stackGrads(net->grad, net->Vgrad);
	//add up gradients across nodes (masters)
	int nMasters;
	MPI_Comm_size(mastersComm, &nMasters);
	if (nMasters > 1) {
		MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
				MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
		MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
				MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
	}
	//update is deterministic: can be handled independently by each node
	//communication overhead is probably greater than a parallelised sum
	assert(nMasters>0);
	opt->update(net->grad, nAddedGradients*nMasters);
}

void Learner_utils::updateNNWeights(const Uint nAddedGradients)
{
	assert(bTrain && nAddedGradients>0);
	//add up gradients across nodes (masters)
	int nMasters;
	MPI_Comm_size(mastersComm, &nMasters);
	if (nMasters > 1) {
		MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
				MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
		MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
				MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
	}
	opt->nepoch++;
	opt->update(net->grad, nAddedGradients);
}

void Learner_utils::updateTargetNetwork()
{
	assert(bTrain);
	if (cntUpdateDelay == 0) { //DQN-style frozen weight
		cntUpdateDelay = tgtUpdateDelay;
		opt->moveFrozenWeights(tgtUpdateAlpha);
	}
	if(cntUpdateDelay>0) cntUpdateDelay--;
}

void Learner_utils::buildNetwork(Network*& _net , Optimizer*& _opt,
		const vector<Uint> nouts, Settings & settings,
		vector<Real> weightInitFac, const vector<Uint> addedInputs)
{
	const string netType = settings.nnType;
	const string funcType = settings.nnFunc;
	const vector<int> lsize = settings.readNetSettingsSize();
	assert(nouts.size()>0);

	//edit to multiply the init factor for weights to output layers (one val per layer)
	// negative value (or 1) means normal initialization
	//why on earth would this be needed? policy outputs are better if initialized to be small
	if(!weightInitFac.size()) weightInitFac.resize(nouts.size(),-1);
	if(weightInitFac.size()!=nouts.size())
		die("Err in output weights factors size\n");

	Builder build(settings);
	//check if environment wants a particular network structure
	if (not env->predefinedNetwork(&build))
		build.addInput(nInputs);

	vector<int> inputToFullyConn = {build.getLastLayerID()};
	for (Uint i=0; i<addedInputs.size(); i++) {
		build.addInput(addedInputs[i]);
		inputToFullyConn.push_back(build.getLastLayerID());
	}

	const Uint nsplit = min(static_cast<size_t>(settings.splitLayers),lsize.size());
	for (Uint i=0; i<lsize.size()-nsplit; i++) {
		build.addLayer(lsize[i], netType, funcType, inputToFullyConn);
		inputToFullyConn.resize(0); //when size is 0 it links to last one
	}

	const Uint firstSplit = lsize.size()-nsplit;
	const vector<int> lastJointLayer =
	(inputToFullyConn.size()==0) ? vector<int>{build.getLastLayerID()} : inputToFullyConn;
	// ^ were the various inputs already fed to a FC layer? then link to previous

	if(nsplit) {
		for (Uint i=0; i<nouts.size(); i++)
		{
			build.addLayer(lsize[firstSplit], netType, funcType, lastJointLayer);

			for (Uint j=firstSplit+1; j<lsize.size(); j++)
				build.addLayer(lsize[j], netType, funcType);

			build.addOutput(static_cast<int>(nouts[i]) , "FFNN", weightInitFac[i]);
		}
	} else {
		const int sum =static_cast<int>(accumulate(nouts.begin(),nouts.end(),0));
		const Real fac=*max_element(weightInitFac.begin(),weightInitFac.end());
		build.addOutput(sum, "FFNN", lastJointLayer, fac);
		assert(fac<=1.);
	}

	_net = build.build();

#ifndef __EntropySGD
	_opt = new AdamOptimizer(_net, profiler, settings);
#else
	_opt = new EntropySGD(_net, profiler, settings);
#endif
	_opt->save("initial");
#ifndef NDEBUG
	_opt->restart("initial");
	_opt->save("restarted");
#endif
	//for (const auto & l : _net->layers) l->profiler = profiler;
}

vector<Real> Learner_utils::output_stochastic_policy(const int agentId,
	State& s, Action& a, State& sOld, Action& aOld, const int info, Real r)
{
	Activation* currActivation = net->allocateActivation();
	vector<Real> output(nOutputs);
	vector<Real> input = s.copy_observed();
	//if required, chain together nAppended obs to compose state
	if (nAppended>0) {
		const Uint sApp = nAppended*sInfo.dimUsed;
		if(info==1)
			input.insert(input.end(),sApp, 0);
		else {
			assert(data->Tmp[agentId]->tuples.size()!=0);
			const Tuple * const last = data->Tmp[agentId]->tuples.back();
			input.insert(input.end(),last->s.begin(),last->s.begin()+sApp);
			assert(last->s.size()==input.size());
		}
	}

	if(info==1) {
		net->predict(data->standardize(input), output, currActivation
			#ifdef __EntropySGD //then we sample from target weights
				, net->tgt_weights, net->tgt_biases
			#endif
		);
	} else { //then if i'm using RNN i need to load recurrent connections (else no effect)
		Activation* prevActivation = net->allocateActivation();
		prevActivation->loadMemory(net->mem[agentId]);
		net->predict(data->standardize(input), output, prevActivation, currActivation
			#ifdef __EntropySGD //then we sample from target weights
				, net->tgt_weights, net->tgt_biases
			#endif
		);
		_dispose_object(prevActivation);
	}
	//save network transition
	currActivation->storeMemory(net->mem[agentId]);
	_dispose_object(currActivation);
	return output;
}

vector<Real> Learner_utils::output_value_iteration(const int agentId, State& s,
	 Action& a, State& sOld, Action& aOld, const int info, Real r)
{
	assert(info==1 || data->Tmp[agentId]->tuples.size());
	Activation* currActivation = net->allocateActivation();
	vector<Real> output(nOutputs);
	vector<Real> inputs(nInputs,0);
	s.copy_observed(inputs);
	if (info==1) {
		vector<Real> scaledSold = data->standardize(inputs);
		net->predict(scaledSold, output, currActivation);
	} else {
		//then if i'm using RNN i need to load recurrent connections
		if (nAppended>0) {
			const Tuple * const last = data->Tmp[agentId]->tuples.back();
			assert(last->s.size()==nInputs);
			for(Uint i=0; i<nAppended*sInfo.dimUsed; i++)
				inputs[sInfo.dimUsed + i] = last->s[i];
		}
		Activation* prev = net->allocateActivation();
		prev->loadMemory(net->mem[agentId]);
		net->predict(data->standardize(inputs), output, prev, currActivation);
		_dispose_object(prev);
	}
	//save network transition
	currActivation->storeMemory(net->mem[agentId]);
	_dispose_object(currActivation);
	return output;
}

void Learner_utils::processStats(const Real avgTime)
{
	stats.minQ= 1e5; stats.maxQ=-1e5; stats.MSE=0;
	stats.avgQ=0; stats.relE=0; stats.dumpCount=0;
	for (Uint i=0; i<Vstats.size(); i++) {
		stats.MSE += Vstats[i]->MSE; //stats.relE += Vstats[i]->relE;
		stats.avgQ += Vstats[i]->avgQ;
		stats.stdQ += Vstats[i]->stdQ;
		stats.dumpCount += Vstats[i]->dumpCount;
		stats.minQ = std::min(stats.minQ,Vstats[i]->minQ);
		stats.maxQ = std::max(stats.maxQ,Vstats[i]->maxQ);
		Vstats[i]->minQ= 1e5; Vstats[i]->maxQ=-1e5; Vstats[i]->MSE=0;
		Vstats[i]->avgQ=0; Vstats[i]->stdQ=0; Vstats[i]->dumpCount=0;
	}
	if (stats.dumpCount<2) return;
	stats.epochCount++;
	epochCounter = stats.epochCount;
	const long double sum=stats.avgQ, sumsq=stats.stdQ, cnt=stats.dumpCount;
	//stats.MSE  /= cnt-1;
	stats.MSE   = std::sqrt(stats.MSE/cnt);
	stats.avgQ /= cnt; //stats.relE/=stats.dumpCount;
	stats.stdQ  = std::sqrt((sumsq-sum*sum/cnt)/cnt);

	long double sumWeights = 0, distTarget = 0, sumWeightsSq = 0;
#pragma omp parallel for reduction(+:sumWeights,distTarget,sumWeightsSq)
	for (Uint w=0; w<net->getnWeights(); w++){
		sumWeights += std::fabs(net->weights[w]);
		sumWeightsSq += net->weights[w]*net->weights[w];
		distTarget += std::pow(net->weights[w]-net->tgt_weights[w],2);
	}
	processGrads();
	ofstream filestats;
	filestats.open("stats.txt", ios::app);
	printf("%d (%lu), rmse:%Lg, avg_Q:%Lg, std_Q:%Lg, min_Q:%Lg, max_Q:%Lg, weight:[%Lg %Lg %Lg], dT %f\n",
		stats.epochCount, opt->nepoch, stats.MSE, stats.avgQ, stats.stdQ,
		stats.minQ, stats.maxQ, sumWeights, sumWeightsSq, distTarget, avgTime);
	filestats<<stats.epochCount<<"\t"<<stats.MSE<<"\t" <<stats.relE<<"\t"
			<<stats.avgQ<<"\t"<<stats.stdQ<<"\t"<<stats.maxQ<<"\t"<<stats.minQ<<"\t"
			<<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<distTarget<<"\t"
			<<stats.dumpCount<<"\t"<<opt->nepoch<<"\t"<<avgTime<<endl;
	filestats.close();
	fflush(0);
	if (stats.epochCount % 100==0) save("policy");
}

void Learner_utils::processGrads()
{
	const vector<Real> oldsum = avgGrad[0], oldstd = stdGrad[0];
	statsVector(avgGrad, stdGrad, cntGrad);
	//std::ostringstream o; o << "Grads avg (std): ";
	//for (Uint i=0; i<avgGrad[0].size(); i++)
	//	o<<avgGrad[0][i]<<" ("<<stdGrad[0][i]<<") ";
	//cout<<o.str()<<endl;
	ofstream filestats;
	filestats.open("grads.txt", ios::app);
	filestats<<print(avgGrad[0]).c_str()<<" "<<print(stdGrad[0]).c_str()<<endl;
	filestats.close();
	for (Uint i=0; i<avgGrad[0].size(); i++) {
		avgGrad[0][i] = .99*oldsum[i] +.01*avgGrad[0][i];
		//stdGrad[0][i] = .99*oldstd[i] +.01*stdGrad[0][i];
		stdGrad[0][i] = max(0.99*oldstd[i], stdGrad[0][i]);
	}
}

void Learner_utils::dumpPolicy()
{
	//a fail in any of these amounts to a big and fat TODO
	if(nAppended) die("TODO missing features\n");
	const Uint nDumpPoints = env->getNdumpPoints();

	FILE * pFile = fopen ("dump.txt", "ab");
	vector<Real> output(nOutputs), dump(nInputs+5);
	Activation* act = net->allocateActivation();
	for (Uint i=0; i<nDumpPoints; i++)
	{
		vector<Real> state = env->getDumpState(i);
		assert(state.size()==nInputs);
		net->predict(data->standardize(state), output, act);
		Uint k=0;
		for (Uint j=0; j<nInputs;  j++) dump[k++] = state[j];
		for (Uint j=0; j<5; j++) dump[k++] = output[j];
		//state.insert(state.end(),output.begin(),output.end()); //unsafe
		fwrite(dump.data(),sizeof(Real),dump.size(),pFile);
	}
	_dispose_object(act);
	fclose (pFile);
}

void Learner_utils::dumpNetworkInfo(const int agentId)
{
	#ifdef _dumpNet_
	if (bTrain) return;
	#else
	return;
	#endif
	net->dump(agentId);
	vector<Real> output(nOutputs);
	const Uint ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
	if (ndata == 0) return;

	vector<Activation*> series_base = net->allocateUnrolledActivations(ndata);
	for (Uint k=0; k<ndata; k++) {
		const Tuple * const _t = data->Tmp[agentId]->tuples[k];
		net->predict(data->standardize(_t->s), output, series_base, k);
	}

	string fname="gradInputs_"+to_string(agentId)+"_"+to_string(ndata)+".dat";
	ofstream out(fname.c_str());
	if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
	//first row of file is net output:
	for(Uint j=0; j<nOutputs; j++) out << output[j] << " ";
	out << "\n";
	//sensitivity of value for this action in this state wrt all previous inputs
	Uint start0 = ndata > MAX_UNROLL_BFORE ? ndata-MAX_UNROLL_BFORE-1 : 0;
	for (Uint ii=start0; ii<ndata; ii++) {
		Uint start1 = ii > MAX_UNROLL_BFORE ? ii-MAX_UNROLL_BFORE-1 : 0;
		for (Uint i=0; i<nInputs; i++) {
			vector<Activation*> series =net->allocateUnrolledActivations(ndata);
			for (Uint k=start1; k<ndata; k++) {
				vector<Real> state = data->Tmp[agentId]->tuples[k]->s;
				if (k==ii) state[i] = 0;
				net->predict(data->standardize(state), output, series, k);
			}
			vector<Real> oDiff = net->getOutputs(series.back());
			vector<Real> oBase = net->getOutputs(series_base.back());
			//other rows of file are d(net output)/d(state_i(t):
			for(Uint j=0; j<nOutputs; j++) {
				const Real dOut = oDiff[j]-oBase[j];
				const Real dState = data->Tmp[agentId]->tuples[ii]->s[i];
				out << dOut/dState << " ";
			}
			out << "\n";
			net->deallocateUnrolledActivations(&series);
		}
	}
	out.close();
	net->deallocateUnrolledActivations(&series_base);
}
