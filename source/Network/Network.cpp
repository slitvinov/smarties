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

void Network::seqPredict_inputs(const vector<Real>& _input, Activation* const currActivation) const
{
  assert(_input.size()==nInputs);
  for (Uint j=0; j<nInputs; j++) currActivation->outvals[iInp[j]] = _input[j];
}

//cache friendly prop for time series: time is fast index, layer is slow index
void Network::seqPredict_execute(
  const vector<Activation*>& series_1, vector<Activation*>& series_2,
  const Uint start, const nnReal* const _weights, const nnReal* const _biases) const
{
  const Uint T = std::min(series_1.size()+1,series_2.size());
  for (Uint j=0; j<nLayers; j++)
  for (Uint t=start; t<T; t++)  {
    Activation* const currActivation = series_2[t];
    const Activation* const prevActivation = t ? series_1[t-1] : nullptr;
    layers[j]->propagate(prevActivation,currActivation,_weights,_biases);
  }
}

void Network::seqPredict_output(vector<Real>& _out, Activation* const a) const
{
  assert(_out.size()==nOutputs);
  for (Uint i=0; i<nOutputs; i++) _out[i] = a->outvals[iOut[i]];
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
    vector<Activation*>& timeSeries, const Uint n_step,
    const nnReal* const _weights, const nnReal* const _biases) const
{
  assert(n_step<timeSeries.size());

  Activation* const currActivation = timeSeries[n_step];
  Activation* const prevActivation = n_step==0 ? nullptr : timeSeries[n_step-1];

  assert(_input.size()==nInputs);
  for(Uint j=0; j<nInputs; j++) currActivation->outvals[iInp[j]] = _input[j];

  for(Uint j=0; j<nLayers; j++)
    layers[j]->propagate(prevActivation,currActivation,_weights,_biases);

  assert(_output.size()==nOutputs);
  for(Uint i=0; i<nOutputs; i++) _output[i] = currActivation->outvals[iOut[i]];
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
  const Activation* const prevActivation, Activation* const currActivation,
  const nnReal* const _weights, const nnReal* const _biases) const
{
  assert(_input.size()==nInputs);
  for(Uint j=0; j<nInputs; j++) currActivation->outvals[iInp[j]] = _input[j];

  for(Uint j=0; j<nLayers; j++)
    layers[j]->propagate(prevActivation,currActivation,_weights,_biases);

  assert(_output.size()==nOutputs);
  for(Uint i=0; i<nOutputs; i++) _output[i] = currActivation->outvals[iOut[i]];
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
  Activation*const net, const nnReal*const _ws, const nnReal*const _bs) const
{
  assert(_input.size()==nInputs);
  for (Uint j=0; j<nInputs; j++) net->outvals[iInp[j]] = _input[j];

  for (Uint j=0; j<nLayers; j++) layers[j]->propagate(net, _ws, _bs);

  assert(_output.size()==nOutputs);
  for (Uint i=0; i<nOutputs; i++) _output[i] = net->outvals[iOut[i]];
}

void Network::backProp(vector<Activation*>&S, const nnReal*const _ws,
  const nnReal*const _bs, Grads*const _gs) const
{
  //cache friendly backprop: backprops from terminal layers to first layers
  //and from last time step to first, with layer being the 'slow index'
  //maximises reuse of weights in cache by getting each layer done in turn
  //but a layer cannot have recur connection to upper layer... would be weird
  assert(S.size()>0);
  const Uint last = S.size()-1;

  if (last == 0)  //just one activation
    for (Uint i=1; i<=nLayers; i++)
      layers[nLayers-i]->backPropagate(nullptr, S[0], nullptr, _gs, _ws, _bs);
  else
  {
    for (Uint i=1; i<=nLayers; i++)
    {
      layers[nLayers-i]->backPropagate(S[last-1],S[last],nullptr,_gs,_ws,_bs);

      for (Uint k=last-1; k>=1; k--)
      layers[nLayers-i]->backPropagate(S[k-1], S[k], S[k+1], _gs, _ws, _bs);

      layers[nLayers-i]->backPropagate(nullptr, S[0], S[1], _gs, _ws, _bs);
    }
  }
}

void Network::backProp(const vector<Real>& _errors, Activation* const net,
  const nnReal*const _weights,const nnReal*const _bias,Grads*const _grad) const
{
  net->clearErrors();
  assert(_errors.size()==nOutputs);
  for (Uint i=0; i<nOutputs; i++) net->errvals[iOut[i]] = _errors[i];

  for (Uint i=1; i<=nLayers; i++)
    layers[nLayers-i]->backPropagate(net, _grad, _weights, _bias);
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

Network::Network(Builder* const B, Settings & settings) :
  nAgents(B->nAgents),   nThreads(B->nThreads), nInputs(B->nInputs),
  nOutputs(B->nOutputs), nLayers(B->nLayers),   nNeurons(B->nNeurons),
  nWeights(B->nWeights), nBiases(B->nBiases),   nStates(B->nStates),
  bDump(not settings.bTrain), layers(B->layers), links(B->links),
  weights(B->weights), biases(B->biases), tgt_weights(B->tgt_weights),
  tgt_biases(B->tgt_biases), grad(B->grad), Vgrad(B->Vgrad), mem(B->mem),
  generators(settings.generators), iOut(B->iOut), iInp(B->iInp)
{
  dump_ID.resize(nAgents);
  updateFrozenWeights();
}

void Network::checkGrads()
{
  printf("Checking gradients\n");
  const Uint seq_len = 3;
  const nnReal incr = 1./32./32./32./8.;
  const nnReal tol  = 1e-4;
  Grads * testg = new Grads(nWeights,nBiases);
  Grads * check = new Grads(nWeights,nBiases);
  Grads * error = new Grads(nWeights,nBiases);
  for (Uint j=0; j<nWeights; j++) {check->_W[j] = 0; error->_W[j] = 0;}
  for (Uint j=0; j<nBiases; j++)  {check->_B[j] = 0; error->_B[j] = 0;}
  vector<Activation*> timeSeries = allocateUnrolledActivations(seq_len);

  //TODO: check with #pragma omp parallel for collapse(2): add a critical/atomic region and give each thread a copy of weights for finite diff
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

      nnReal diff = 0;
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

        const nnReal scale = max(fabs(testG->_W[w]), fabs(diff));
        if (scale < 2.2e-16) continue;
        const nnReal err = fabs(testG->_W[w]-diff);
        const nnReal relerr = err/scale;
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

        const nnReal scale = max(fabs(testG->_B[w]), fabs(diff));
        if (scale < 2.2e-16) continue;
        const nnReal err = fabs(testG->_B[w] -diff);
        const nnReal relerr = err/scale;
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
    if (check->_W[w]>tol && testg->_W[w] > 2.2e-16)
    cout<<"W"<<w<<" rel err:"<<check->_W[w]<<" analytical:"<<testg->_W[w]<<endl;

    sum1   += fabs(testg->_W[w]);
    sum2   += fabs(check->_W[w]);
    sum3   += fabs(error->_W[w]);
    sumsq1 += testg->_W[w]*testg->_W[w];
    sumsq2 += check->_W[w]*check->_W[w];
    sumsq3 += error->_W[w]*error->_W[w];
  }

  for (Uint w=0; w<nBiases; w++) {
    if (check->_B[w]>tol && testg->_B[w] > 2.2e-16)
    cout<<"B"<<w<<" rel err:"<<check->_B[w]<<" analytical:"<<testg->_B[w]<<endl;

    sum1   += fabs(testg->_B[w]);
    sum2   += fabs(check->_B[w]);
    sum3   += fabs(error->_B[w]);
    sumsq1 += testg->_B[w]*testg->_B[w];
    sumsq2 += check->_B[w]*check->_B[w];
    sumsq3 += error->_B[w]*error->_B[w];
  }
  const long double NW = nWeights + nBiases;
  const long double std1 = sqrt((sumsq1 - sum1*sum1/NW)/NW);
  const long double std2 = sqrt((sumsq2 - sum2*sum2/NW)/NW);
  const long double std3 = sqrt((sumsq3 - sum3*sum3/NW)/NW);
  const long double mean1 = sum1/NW, mean2 = sum2/NW, mean3 = sum3/NW;
  printf("Mean rel err:%Le (std:%Le). Mean abs err:%Le (std:%Le). Mean abs grad:%Le (std:%Le).\n", mean2, std2, mean3, std3, mean1, std1);
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
  string nameNeurons  = "neuronOuts_"+to_string(agentID)+"_"+string(buf)+".dat";
  string nameMemories = "cellStates_"+to_string(agentID)+"_"+string(buf)+".dat";
  string nameOut_Mems = "out_states_"+to_string(agentID)+"_"+string(buf)+".dat";
  {
    ofstream out(nameOut_Mems.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameOut_Mems.c_str());
    for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
    for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
    out << "\n";
    out.close();
  }
  {
    ofstream out(nameNeurons.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameNeurons.c_str());
    for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
    out << "\n";
    out.close();
  }
  {
    ofstream out(nameMemories.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameMemories.c_str());
    for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
    out << "\n";
    out.close();
  }
  dump_ID[agentID]++;
}
