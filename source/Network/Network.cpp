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

vector<Real> Network::predict(const vector<Real>& _inp,
  const Activation*const prevStep, const Activation*const currStep,
  const Parameters*const _weights) const
{
  assert(_inp.size()==nInputs);
  currStep->setInput(_inp);

  const Parameters*const W = _weights == nullptr ? weights : _weights;

  for(Uint j=1; j<nLayers; j++) layers[j]->forward(prevStep, currStep, W);

  return currStep->getOutput();
}

void Network::backProp( const Activation*const prevStep,
                        const Activation*const currStep,
                        const Activation*const nextStep,
                        const Parameters*const _gradient,
                        const Parameters*const _weights) const
{
  for (Uint i=layers.size()-1; i>0; i--)
    layers[i]->backward(prevStep, currStep, nextStep, _gradient, _weights);
}

void Network::backProp(const vector<Activation*>& netSeries,
                       const Uint stepLastError,
                       const Parameters*const _grad,
                       const Parameters*const _weights) const
{
  //cache friendly backprop: backprops from upper layers to bottom layers
  //and from last time step to first, with layer being the 'slow index'
  //maximises reuse of weights in cache by getting each layer done in turn
  assert(stepLastError <= netSeries.size());
  const Parameters*const W = _weights == nullptr ? weights : _weights;

  if (stepLastError == 0) return; //no errors placed
  else
  if (stepLastError == 1)  //errors placed at first time step
    for(Uint i=layers.size()-1; i>0; i--)
      layers[i]->backward(nullptr, netSeries[0], nullptr, _grad, W);
  else
  {
    const Uint T = stepLastError - 1;
    for(Uint i=layers.size()-1; i>0; i--)
    {
      layers[i]->backward(netSeries[T-1],netSeries[T],nullptr,        _grad,W);

      for (Uint k=T-1; k>0; k--)
      layers[i]->backward(netSeries[k-1],netSeries[k],netSeries[k+1], _grad,W);

      layers[i]->backward(       nullptr,netSeries[0],netSeries[1],   _grad,W);
    }
  }
}

Network::Network(Builder* const B, Settings & settings) :
  nAgents(B->nAgents), nThreads(B->nThreads), nInputs(B->nInputs),
  nOutputs(B->nOutputs), nLayers(B->nLayers), bDump(not settings.bTrain),
  layers(B->layers), weights(B->weights), tgt_weights(B->tgt_weights),
  Vgrad(B->Vgrad), mem(B->mem), generators(settings.generators) {
  dump_ID.resize(nAgents, 0);
}

#if 0
void Network::checkGrads()
{
  printf("Checking gradients\n");
  const Uint seq_len = 3;
  const nnReal incr = std::sqrt(nnEPS);
  const nnReal tol  = std::sqrt(incr);
  Grads * testg = new Grads(nWeights,nBiases);
  Grads * check = new Grads(nWeights,nBiases);
  Grads * error = new Grads(nWeights,nBiases);
  vector<Activation*> timeSeries = allocateUnrolledActivations(seq_len);

  //TODO: check with #pragma omp parallel for collapse(2): add a critical/atomic region and give each thread a copy of weights for finite diff
  for (Uint t=0; t<seq_len; t++)
    for (Uint o=0; o<nOutputs; o++)
    {
      vector<Real> res(nOutputs);
      Grads * testG_bck = new Grads(nWeights,nBiases);
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
      backProp(timeSeries, testG_bck);
      sort_bck_to_fwd(testG_bck->_W, testG->_W);
      for(Uint j=0; j<nBiases; j++) testG->_B[j] = testG_bck->_B[j];

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
        if (scale < nnEPS) continue;
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
        if (scale < nnEPS) continue;
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
      _dispose_object(testG_bck);
    }

  long double sum1 = 0, sumsq1 = 0, sum2 = 0, sumsq2 = 0, sum3 = 0, sumsq3 = 0;
  for (Uint w=0; w<nWeights; w++) {
    if (check->_W[w]>tol && testg->_W[w] > nnEPS)
    cout<<"W"<<w<<" rel err:"<<check->_W[w]<<" analytical:"<<testg->_W[w]<<endl;

    sum1   += fabs(testg->_W[w]);
    sum2   += fabs(check->_W[w]);
    sum3   += fabs(error->_W[w]);
    sumsq1 += testg->_W[w]*testg->_W[w];
    sumsq2 += check->_W[w]*check->_W[w];
    sumsq3 += error->_W[w]*error->_W[w];
  }

  for (Uint w=0; w<nBiases; w++) {
    if (check->_B[w]>tol && testg->_B[w] > nnEPS)
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
#endif
