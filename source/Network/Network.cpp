/*
 *  LSTMNet.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Builder.h"
#include "Network.h"
using namespace ErrorHandling;

void Network::seqPredict_inputs(const vector<Real>& _input, Activation* const currActivation) const
{
  assert(_input.size()==nInputs);
  for (Uint j=0; j<nInputs; j++)
    currActivation->outvals[iInp[j]] = _input[j];
}

void Network::seqPredict_execute(
		const vector<Activation*>& series_1, vector<Activation*>& series_2,
		const Real* const _weights, const Real* const _biases) const
{
	const Uint T = std::min(series_1.size(), series_2.size());
    for (Uint j=0; j<nLayers; j++)
	for (Uint t=0; t<T; t++)  {
		Activation* const currActivation = series_2[t];
		const Activation* const prevActivation = t ? series_1[t-1] : nullptr;
        layers[j]->propagate(prevActivation,currActivation,_weights,_biases);
	}
}

void Network::seqPredict_output(vector<Real>& _output, Activation* const currActivation) const
{
    assert(_output.size()==nOutputs);
    for (Uint i=0; i<nOutputs; i++)
        _output[i] = *(currActivation->outvals + iOut[i]);
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
						vector<Activation*>& timeSeries, const Uint n_step,
						const Real* const _weights, const Real* const _biases) const
{
	assert(n_step<timeSeries.size());

	Activation* const currActivation = timeSeries[n_step];
	Activation* const prevActivation = n_step==0 ? nullptr : timeSeries[n_step-1];

  assert(_input.size()==nInputs);
  for (Uint j=0; j<nInputs; j++)
    currActivation->outvals[iInp[j]] = _input[j];

  for (Uint j=0; j<nLayers; j++)
      layers[j]->propagate(prevActivation,currActivation,_weights,_biases);

  assert(_output.size()==nOutputs);

  for (Uint i=0; i<nOutputs; i++)
      _output[i] = *(currActivation->outvals + iOut[i]);
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
						Activation* const prevActivation, Activation* const currActivation,
						const Real* const _weights, const Real* const _biases) const
{
    assert(_input.size()==nInputs);
    for (Uint j=0; j<nInputs; j++)
      currActivation->outvals[iInp[j]] = _input[j];

    for (Uint j=0; j<nLayers; j++)
        layers[j]->propagate(prevActivation,currActivation,_weights,_biases);

    assert(_output.size()==nOutputs);

    for (Uint i=0; i<nOutputs; i++)
        _output[i] = *(currActivation->outvals + iOut[i]);
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
		Activation* const net, const Real* const _weights, const Real* const _biases) const
{
    assert(_input.size()==nInputs);
    for (Uint j=0; j<nInputs; j++)
    	net->outvals[iInp[j]] = _input[j];

    for (Uint j=0; j<nLayers; j++)
        layers[j]->propagate(net,_weights,_biases);

    assert(_output.size()==nOutputs);

    for (Uint i=0; i<nOutputs; i++)
        _output[i] = net->outvals[iOut[i]];
}

void Network::backProp(vector<Activation*>& timeSeries, const Real* const _weights,
                        const Real* const _biases, Grads* const _grads) const
{
  assert(timeSeries.size()>0);
	const Uint last = timeSeries.size()-1;

  if (last == 0) { //just one activation
    for (Uint i=1; i<=nLayers; i++)
      layers[nLayers-i]->backPropagate((Activation*)nullptr,timeSeries[last],
                                     (Activation*)nullptr, _grads, _weights, _biases);
  } else if (last == 1) {
    for (Uint i=1; i<=nLayers; i++)
      layers[nLayers-i]->backPropagate(timeSeries[0],timeSeries[1],
                                  (Activation*)nullptr, _grads, _weights, _biases);
    for (Uint i=1; i<=nLayers; i++)
      layers[nLayers-i]->backPropagate((Activation*)nullptr,timeSeries[0],
                                     timeSeries[1], _grads, _weights, _biases);
  } else {
    for (Uint i=1; i<=nLayers; i++)
      layers[nLayers-i]->backPropagate(timeSeries[last-1],timeSeries[last],
                                  (Activation*)nullptr, _grads, _weights, _biases);

    for (Uint k=last-1; k>=1; k--)
    for (Uint i=1; i<=nLayers; i++)
      layers[nLayers-i]->backPropagate(timeSeries[k-1],timeSeries[k],timeSeries[k+1],
                                        _grads, _weights, _biases);

    for (Uint i=1; i<=nLayers; i++) {
      layers[nLayers-i]->backPropagate((Activation*)nullptr, timeSeries[0],
                                       timeSeries[1], _grads, _weights, _biases);
    }
  }
}

void Network::backProp(const vector<Real>& _errors,
		Activation* const net, const Real* const _weights, const Real* const _biases, Grads* const _grads) const
{
	net->clearErrors();
	assert(_errors.size()==nOutputs);
	for (Uint i=0; i<nOutputs; i++)
		net->errvals[iOut[i]] = _errors[i];

    for (Uint i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagate(net, _grads, _weights, _biases);
}

void Network::clearErrors(vector<Activation*>& timeSeries) const
{
	for (Uint k=0; k<timeSeries.size(); k++) timeSeries[k]->clearErrors();
}

void Network::setOutputDeltas(const vector<Real>& _errors, Activation* const net) const
{
    assert(_errors.size()==nOutputs);
    for (Uint i=0; i<nOutputs; i++)
    	net->errvals[iOut[i]] = _errors[i];
}

void Network::updateFrozenWeights()
{
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (Uint j=0; j<nWeights; j++)
            *(tgt_weights + j) = *(weights + j);

        #pragma omp for nowait
        for (Uint j=0; j<nBiases; j++)
            *(tgt_biases + j) = *(biases + j);
    }
}

Activation* Network::allocateActivation() const
{
	return new Activation(nNeurons,nStates);
}

vector<Activation*> Network::allocateUnrolledActivations(Uint length) const
{
	vector<Activation*> ret(length);
	for (Uint j=0; j<length; j++)
		ret[j] = new Activation(nNeurons,nStates);
	return ret;
}

void Network::deallocateUnrolledActivations(vector<Activation*>* const ret) const
{
	for (auto & trash : *ret) _dispose_object(trash);
}

void Network::appendUnrolledActivations(vector<Activation*>* const ret, Uint length) const
{
	for (Uint j=0; j<=length; j++)
		ret->push_back(new Activation(nNeurons,nStates));
}

Network::Network(Builder* const B, Settings & settings) :
nAgents(B->nAgents),   nThreads(B->nThreads), nInputs(B->nInputs),
nOutputs(B->nOutputs), nLayers(B->nLayers),   nNeurons(B->nNeurons),
nWeights(B->nWeights), nBiases(B->nBiases),   nStates(B->nStates),
bDump(not settings.bTrain), iOut(B->iOut), iInp(B->iInp),
layers(B->layers), links(B->links), weights(B->weights), biases(B->biases),
tgt_weights(B->tgt_weights), tgt_biases(B->tgt_biases), grad(B->grad),
Vgrad(B->Vgrad), generators(settings.generators), mem(B->mem)
{
  dump_ID.resize(nAgents);
  updateFrozenWeights();
}

void Network::checkGrads()
{
    printf("Checking gradients\n");
    const Uint seq_len = 3;
    const Real incr = 1./32./32./32./8.;
    const Real tol  = 1e-4;
    Grads * testg = new Grads(nWeights,nBiases);
    Grads * check = new Grads(nWeights,nBiases);
    Grads * error = new Grads(nWeights,nBiases);
    for (Uint j=0; j<nWeights; j++) check->_W[j] = 0;
    for (Uint j=0; j<nBiases; j++)  check->_B[j] = 0;
    for (Uint j=0; j<nWeights; j++) error->_W[j] = 0;
    for (Uint j=0; j<nBiases; j++)  error->_B[j] = 0;
    vector<Activation*> timeSeries = allocateUnrolledActivations(seq_len);

    for (Uint t=0; t<seq_len; t++)
    for (Uint o=0; o<nOutputs; o++)
    {
        vector<Real> res(nOutputs);
        Grads * testG = new Grads(nWeights,nBiases);
        vector<vector<Real>> inputs(seq_len,vector<Real>(nInputs,0));
        for (Uint k=0; k<timeSeries.size(); k++) timeSeries[k]->clearErrors();
        //for (Uint k=0; k<timeSeries.size(); k++) timeSeries[k]->clearOutput();
        //for (Uint k=0; k<timeSeries.size(); k++) timeSeries[k]->clearInputs();

        normal_distribution<Real> dis_inp(0,2);
        for(Uint i=0;i<seq_len;i++)
        for(Uint j=0;j<nInputs;j++)
        inputs[i][j]=dis_inp(generators[0]);

        for (Uint k=0; k<seq_len; k++)
        {
        	predict(inputs[k], res, timeSeries, k);
          vector<Real> errs(nOutputs,0);
          if(k==t) errs[o] = -1.;
          setOutputDeltas(errs, timeSeries[k]);
        }
        backProp(timeSeries, testG);

        Real diff;
        for (Uint w=0; w<nWeights; w++) {
            //1
            weights[w] += incr;
            for (Uint k=0; k<seq_len; k++) {
            	predict(inputs[k], res, timeSeries, k);
              if(k==t) diff = -res[o]/(2*incr);
            }
            //2
            weights[w] -= 2*incr;
            for (Uint k=0; k<seq_len; k++) {
            	predict(inputs[k], res, timeSeries, k);
              if(k==t) diff += res[o]/(2*incr);
            }
            //0
            weights[w] += incr;

            const Real scale = max(fabs(testG->_W[w]), fabs(diff));
            if (scale < 2.2e-16) continue;
            const Real err = fabs(testG->_W[w]-diff);
            const Real relerr = err/scale;
            //if(relerr>check->_W[w])
            if(err>error->_W[w])
            {
              testg->_W[w] = testG->_W[w];
              check->_W[w] = relerr;
              error->_W[w] = err;
            }
        }
        for (Uint w=0; w<nBiases; w++) {
            //1
            biases[w] += incr;
            for (Uint k=0; k<seq_len; k++) {
            	predict(inputs[k], res, timeSeries, k);
              if(k==t) diff = -res[o]/(2*incr);
            }
            //2
            biases[w] -= 2*incr;
            for (Uint k=0; k<seq_len; k++) {
            	predict(inputs[k], res, timeSeries, k);
              if(k==t) diff += res[o]/(2*incr);
            }
            //0
            biases[w] += incr;

            const Real scale = max(fabs(testG->_B[w]), fabs(diff));
            if (scale < 2.2e-16) continue;
            const Real err = fabs(testG->_B[w] -diff);
            const Real relerr = err/scale;
            //if(relerr>check->_B[w])
            if(err>error->_B[w])
            {
              testg->_B[w] = testG->_B[w];
              check->_B[w] = relerr;
              error->_B[w] = err;
            }
        }

        _dispose_object(testG);
    }

    long double sum1 = 0, sumsq1 = 0, sum2 = 0, sumsq2 = 0, sum3 = 0, sumsq3 = 0;
    for (Uint w=0; w<nWeights; w++) {
      if (check->_W[w]>tol && testg->_W[w] > 2.2e-16) {
            cout <<"W"<<w<<" relative error:"<<check->_W[w]
                         <<" analytical:"<<testg->_W[w]<<endl;
      }
      sum1   += fabs(testg->_W[w]);
      sum2   += fabs(check->_W[w]);
      sum3   += fabs(error->_W[w]);
      sumsq1 += testg->_W[w]*testg->_W[w];
      sumsq2 += check->_W[w]*check->_W[w];
      sumsq3 += error->_W[w]*error->_W[w];
    }

    for (Uint w=0; w<nBiases; w++) {
      if (check->_B[w]>tol && testg->_B[w] > 2.2e-16) {
            cout <<"B"<<w<<" relative error:"<<check->_B[w]
                         <<" analytical:"<<testg->_B[w]<<endl;
      }
      sum1   += fabs(testg->_B[w]);
      sum2   += fabs(check->_B[w]);
      sum3   += fabs(error->_B[w]);
      sumsq1 += testg->_B[w]*testg->_B[w];
      sumsq2 += check->_B[w]*check->_B[w];
      sumsq3 += error->_B[w]*error->_B[w];
    }
    const long double NW = nWeights + nBiases;
    const long double mean1 = sum1/NW;
    const long double mean2 = sum2/NW;
    const long double mean3 = sum3/NW;
    const long double std1 = sqrt((sumsq1 - sum1*sum1/NW)/NW);
    const long double std2 = sqrt((sumsq2 - sum2*sum2/NW)/NW);
    const long double std3 = sqrt((sumsq3 - sum3*sum3/NW)/NW);
    printf("Mean relative error:%Le (std:%Le). Mean absolute error:%Le (std:%Le). Mean absolute gradient:%Le (std:%Le).\n",
      mean2, std2, mean3, std3, mean1, std1);
    _dispose_object(testg);
    _dispose_object(check);
    _dispose_object(error);
    deallocateUnrolledActivations(&timeSeries);
    fflush(0);
}

void Network::dump(const int agentID)
{
    if (not bDump) return;
    char buf[500];
    sprintf(buf, "%07u", (Uint)dump_ID[agentID]);
    string nameNeurons  = "neuronOuts_" + to_string(agentID) + "_" + string(buf) + ".dat";
    string nameMemories = "cellStates_" + to_string(agentID) + "_" + string(buf) + ".dat";
    string nameOut_Mems = "out_states_" + to_string(agentID) + "_" + string(buf) + ".dat";
    {
        ofstream out(nameOut_Mems.c_str());
        if (!out.good()) _die("Unable to open save into file %s\n", nameOut_Mems.c_str());
        for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameNeurons.c_str());
        if (!out.good()) _die("Unable to open save into file %s\n", nameNeurons.c_str());
        for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameMemories.c_str());
        if (!out.good()) _die("Unable to open save into file %s\n", nameMemories.c_str());
        for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    dump_ID[agentID]++;
}
