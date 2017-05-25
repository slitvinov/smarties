/*
 *  LSTMNet.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include "LSTMNet.h"
#include <cassert>

using namespace ErrorHandling;


FishNet::FishNet(Settings & settings) : nInputs(settings.nnInputs), nOutputs(settings.nnOutputs), nAgents(settings.nAgents), bRecurrent(settings.nnType==1)
{
	vector<int> lsize;
	lsize.push_back(settings.nnLayer1);
	if (settings.nnLayer2>1) {
		lsize.push_back(settings.nnLayer2);
		if (settings.nnLayer3>1) {
			lsize.push_back(settings.nnLayer3);
			if (settings.nnLayer4>1) {
				lsize.push_back(settings.nnLayer4);
				if (settings.nnLayer5>1) {
					lsize.push_back(settings.nnLayer5);
				}
			}
		}
	}

    profiler = new Profiler();
	net = new Network(settings);
	net->addInput(nInputs);
	string lType = bRecurrent ? "LSTM" : "Normal";
	for (int i=0; i<lsize.size(); i++) net->addLayer(lsize[i], lType);
	net->addOutput(nOutputs, "Normal");
	net->build();
	opt = new AdamOptimizer(net, profiler, settings);
}

void FishNet::save(string fname)
{
    return net->save(fname);
}

bool FishNet::restart(string fname)
{
    return net->restart(fname);
}

void FishNet::train(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, int batchsize, int nepochs)
{
    if (inputs.size() != targets.size()) die("Mismatch between batch size of targets and inputs\n");

    std::chrono::time_point<std::chrono::high_resolution_clock> start,end;
    const int ndata = inputs.size();
    const int nbatches = floor((Real)ndata/batchsize);
    vector<const vector<Real>*> batch_in(batchsize), batch_out(batchsize);

    indexes.reserve(ndata);
    for (int i=0; i<ndata; ++i)  {
        if (static_cast<int>(inputs[i].size()) != nInputs) die("Mismatch between size of input %d and net inputs\n",i);
        if (static_cast<int>(targets[i].size()) != nOutputs) die("Mismatch between size of output %d and net outputs\n",i);
        indexes.push_back(i);
    }

    for (int e=0; e<nepochs; e++) {
        start = std::chrono::high_resolution_clock::now();
        Real batch_err(0.), err;
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int b=0; b<nbatches; ++b) {
            for (int i=0; i<batchsize; ++i) {
                batch_in[i]  =  &inputs[indexes[batchsize*b+i]];
                batch_out[i] = &targets[indexes[batchsize*b+i]];
            }
            trainBatch(batch_in,batch_out,err);
            batch_err+=err;
        }
        end = std::chrono::high_resolution_clock::now();
        printf("Epoch %d/%d took %f seconds and had absolute MSE of %f. \n",e,nepochs,std::chrono::duration<Real>(end-start).count(),batch_err/ndata);
        //cout << profiler->printStat() << endl;
    }
}

void FishNet::train(const vector<vector<vector<Real>>>& inputs, const vector<vector<vector<Real>>>& targets, int nepochs)
{
    if (inputs.size() != targets.size()) die("Mismatch between batch size of targets and inputs\n");
    printf("Data has size %lu %lu\n",inputs.size(), inputs[0].size());

    std::chrono::time_point<std::chrono::high_resolution_clock> start,end;
    const int ndata = inputs.size();

    indexes.reserve(ndata);
    for (int i=0; i<ndata; ++i)  {
        if (inputs[i].size() != targets[i].size()) die("Mismatch between batch size of targets and inputs\n");
        for(size_t j=0; j!=inputs[i].size(); j++) {
            if (static_cast<int>(inputs[i][j].size())!=nInputs) die("Mismatch between size of input %d and net inputs\n", (int)j);
            if (static_cast<int>(targets[i][j].size())!=nOutputs) die("Mismatch between size of output %d and net outputs\n", (int)j);
        }
        indexes.push_back(i);
    }
    net->checkGrads(inputs[101], 2);//inputs[101].size()-1
    for (int e=0; e<nepochs; e++) {
        start = std::chrono::high_resolution_clock::now();
        Real batch_err(0.), err(100.);
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int b=0; b<ndata; ++b)  {
            trainSeries(inputs[indexes[b]],targets[indexes[b]],err);
            batch_err+=err;
        }
        end = std::chrono::high_resolution_clock::now();
        printf("Epoch %d/%d took %f seconds and had absolute MSE of %f. \n",e,nepochs,std::chrono::duration<Real>(end-start).count(),batch_err/(Real)ndata);
        if (batch_err/(Real)ndata > 1.)  printf("Problem %f %d \n", batch_err, ndata);
        //cout << profiler->printStat() << endl;
    }
    net->checkGrads(inputs[101], 2);//inputs[101].size()-1
}


void FishNet::trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE)
{
    trainMSE = 0.;
    int nseries = inputs.size();
    vector<Real> res(nOutputs), errs(nOutputs,0);
    Activation* netActivation = net->allocateActivation();
    netActivation->clearErrors();

    for (int k=0; k<nseries; k++) {
        net->predict(*(inputs[k]), res, netActivation);

        for (int j =0; j<nOutputs; j++) {
            const Real err = (*(targets[k]))[j] - res[j];
            errs[j] = err;
            trainMSE += 0.5*err*err;
        }
        net->backProp(errs, netActivation, net->grad);
    }

    opt->update(net->grad, nseries);
    trainMSE /= (Real)nseries;
    _dispose_object(netActivation);
}

void FishNet::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    vector<Real> res(nOutputs), errs(nOutputs,0);
    const int nseries = inputs.size();
    vector<Activation*> timeSeries = net->allocateUnrolledActivations(nseries);
    net->clearErrors(timeSeries);

    for (int k=0; k<nseries; k++) {
    	net->predict(inputs[k], res, timeSeries, k);

        for (int i=0; i<nOutputs; i++) {
            const Real err = targets[k][i] - res[i];
            errs[i] = err;
            trainMSE += 0.5*err*err;
        }
        net->setOutputDeltas(errs, timeSeries[k]);
    }

    net->backProp(timeSeries, net->grad);
    opt->update(net->grad,nseries);
    trainMSE /= (Real)nseries;
    net->deallocateUnrolledActivations(&timeSeries);
}

void FishNet::predict(const vector<Real>& S1, vector<Real>& Q1, const vector<Real>& S2, vector<Real>& Q2, int iAgent)
{   //used for RL, used not to mess with mem
    if (nInputs != static_cast<int>(S1.size()) || nInputs != static_cast<int>(S2.size())) die("Wrong input dim\n");
    if (iAgent  >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");

    Activation* prevActivation = net->allocateActivation();
    Activation* currActivation = net->allocateActivation();
    Activation* nextActivation = net->allocateActivation();

    Q1.resize(nOutputs);
    Q2.resize(nOutputs);
    net->loadMemory(net->mem[iAgent], prevActivation);
    net->predict(S1, Q1, prevActivation, currActivation);
    net->predict(S2, Q2, currActivation, nextActivation);
    net->loadMemory(net->mem[iAgent], nextActivation);

    _dispose_object(prevActivation);
    _dispose_object(currActivation);
    _dispose_object(nextActivation);
}

void FishNet::predict(const vector<Real>& input, vector<Real>& output, int iAgent)
{
    if (nInputs != static_cast<int>(   input.size())) die("Wrong input dim\n");
    if (iAgent  >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");
    Activation* prevActivation = net->allocateActivation();
    Activation* currActivation = net->allocateActivation();

    output.resize(nOutputs); //might be a problem. Then again, I wouldn't call it MY problem
    net->loadMemory(net->mem[iAgent], prevActivation);
    net->predict(input, output, prevActivation, currActivation);
    net->loadMemory(net->mem[iAgent], currActivation);

    _dispose_object(prevActivation);
    _dispose_object(currActivation);
}

void FishNet::predict(const vector<vector<Real>>& inputs, vector<vector<Real>>& outputs)
{
    int nseries = inputs.size();
    vector<Activation*> timeSeries = net->allocateUnrolledActivations(nseries);
    vector<Real> res(nOutputs);
    outputs.clear();

    for (int k=0; k<nseries; k++) {
        if (nInputs != static_cast<int>(inputs[k].size())) die("Wrong input %d dim\n", k);
        net->predict(inputs[k], res, timeSeries, k);
        outputs.push_back(res);
    }
    net->deallocateUnrolledActivations(&timeSeries);
}

void FishNet::predict(const vector<Real>& input, vector<Real>& output)
{
    if (nInputs != static_cast<int>(   input.size())) die("Wrong input dim\n");
    Activation* currActivation = net->allocateActivation();
    output.resize(nOutputs); //might be a problem. Then again, I wouldn't call it MY problem
    net->predict(input, output, currActivation);
    _dispose_object(currActivation);
}
