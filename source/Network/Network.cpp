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
	assert(n_step<timeSeries.size() && n_step>=0);

	Activation* const currActivation = timeSeries[n_step];
	Activation* const prevActivation = n_step==0 ? nullptr : timeSeries[n_step-1];

  assert(_input.size()==nInputs);
  for (Uint j=0; j<nInputs; j++)
    currActivation->outvals[iInp[j]] = _input[j];

  for (Uint j=0; j<nLayers; j++)
      layers[j]->propagate(prevActivation,currActivation,_weights,_biases);

  assert(soutput.size()==nOutputs);

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

    assert(soutput.size()==nOutputs);

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

    assert(soutput.size()==nOutputs);

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
	for (Uint k=0; k<timeSeries.size(); k--) timeSeries[k]->clearErrors();
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

void Network::loadMemory(Mem * _M, Activation * _N) const
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
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

/*
Real* Network::assignDropoutMask(unsigned int s2, unsigned int s3)
{
    if (Pdrop > 0) {
    	die("You are probably using dropout wrong anyway\n");
      Real * dropW;
      _allocateClean(dropW, nWeights);
      s2 += 2*nWeights; //not sure whether it is required that seeds
      s3 += 4*nWeights; //should be sorted from smallest to biggest... be safe

      for (int w=0; w<nWeights; w++) {

      }
        const Real Pkeep = 1. - Pdrop;
        Real fac = 1./Pkeep; //the others have to compensate

    } else return weights;
}
*/

void Network::checkGrads(const vector<vector<Real>>& inputs, int inp_seq_len)
{
    Uint seq_len = inp_seq_len<0 ? inputs.size() : static_cast<Uint>(inp_seq_len);

    printf("Checking gradients\n");
    vector<Uint> errorPlacements(seq_len);
    vector<Real> partialResults(seq_len);
    vector<Activation*> timeSeries = allocateUnrolledActivations(seq_len);
    assert(timeSeries.size() == seq_len);
    vector<Real> res(nOutputs); //allocate net output

    const Real incr = 1e-7;
    const Real tol  = 1e-3;
    uniform_real_distribution<Real> dis(0.,1.);
    //figure out where to place some errors at random in outputs
    for (Uint i=0; i<seq_len; i++) {
      errorPlacements[i] = nOutputs*dis(generators[0]);
      std::cout << "Placing error in "<< errorPlacements[i]<<std::endl;
    }

    Grads * testg = new Grads(nWeights,nBiases);
    Grads * testG = new Grads(nWeights,nBiases);
    clearErrors(timeSeries);

    for (Uint k=0; k<seq_len; k++) {
    	predict(inputs[k], res, timeSeries, k);
      vector<Real> errs(nOutputs,0);
      //for (Uint i=0; i<nOutputs; i++) errs[i] = -1.;
      errs[errorPlacements[k]] = -1.;
      setOutputDeltas(errs, timeSeries[k]);
    }

    backProp(timeSeries, testG);

    FILE * f;
    f = fopen("weights_finite_diffs.txt", "w");
    if (f == NULL) die("check grads fail\n");

    for (Uint w=0; w<nWeights; w++) {
        //1
        weights[w] += incr;
        for (Uint k=0; k<seq_len; k++) {
        	predict(inputs[k], res, timeSeries, k);
            partialResults[k] = -res[errorPlacements[k]];
        }
        //2
        weights[w] -= 2*incr;
        for (Uint k=0; k<seq_len; k++) {
        	predict(inputs[k], res, timeSeries, k);
            partialResults[k] += res[errorPlacements[k]];
        }
        //0
        weights[w] += incr;

        Real diff(0);
        for (Uint k=0; k<seq_len; k++) diff += partialResults[k];
        testg->_W[w] = diff/(2.*incr);

        //const Real scale = fabs(*(biases+w));
        const Real scale = std::max(std::fabs(testG->_W[w]),
                                    std::fabs(testg->_W[w]));
        const Real err = (testG->_W[w] - testg->_W[w])/scale;
        if (fabs(err)>tol || !nonZero(testG->_W[w])) {
        //if (1) {

              cout <<"W"<<w<<" analytical:"<<testG->_W[w]
                           <<" finite:"<<testg->_W[w]
                           <<" error:"<<err<<endl;

          fprintf(f, "%d %g %g %g\n", w, testG->_W[w], testg->_W[w], err);
        }
    }

    fclose(f);
    f = fopen("biases_finite_diffs.txt", "w");
    if (f == NULL) die("check grads fail 2\n");

    for (Uint w=0; w<nBiases; w++) {
        //1
        *(biases+w) += incr;
        for (Uint k=0; k<seq_len; k++) {
			predict(inputs[k], res, timeSeries, k);
			partialResults[k] = -res[errorPlacements[k]];
		}
        //2
        *(biases+w) -= 2*incr;
        for (Uint k=0; k<seq_len; k++) {
			predict(inputs[k], res, timeSeries, k);
			partialResults[k] += res[errorPlacements[k]];
		}
        //0
        *(biases+w) += incr;

        Real diff(0);
        for (Uint k=0; k<seq_len; k++) diff += partialResults[k];
        testg->_B[w] = diff/(2.*incr);

        //const Real scale = fabs(*(biases+w));
        const Real scale = std::max(std::fabs(testG->_B[w]),
                                    std::fabs(testg->_B[w]));
        const Real err = (testG->_B[w] - testg->_B[w])/scale;
        if (fabs(err)>tol || !nonZero(testG->_B[w])) {
        //if (1) {

              cout <<"B"<<w<<" analytical:"<<testG->_B[w]
                           <<" finite:"<<testg->_B[w]
                           <<" error:"<<err<<endl;

          fprintf(f, "%d %g %g %g\n", w, testG->_B[w], testg->_B[w], err);
        }
    }
    _dispose_object(testg);
    _dispose_object(testG);
    deallocateUnrolledActivations(&timeSeries);
    printf("\n");
    fclose(f);
    fflush(0);
}
#if 0
void Network::save(const string fname)
{
  {
    printf("Saving into %s\n", fname.c_str());
    fflush(0);
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());

    out.precision(20);
    out << nWeights << " "  << nBiases << " " << nLayers  << " " << nNeurons << endl;

    for (int i=0; i<nWeights; i++) {
        if (std::isnan(*(weights + i)) || std::isinf(*(weights + i))) {
            die("Caught a nan\n");
        } else {
            out << *(weights + i) << "\n";
        }
    }

    for (int i=0; i<nBiases; i++) {
       if (std::isnan(*(biases + i)) || std::isinf(*(biases + i))) {
            die("Caught a nan\n");
        } else {
            out << *(biases + i) << "\n";
        }
    }

    out.flush();
    out.close();
    string command = "cp " + nameBackup + " " + fname;
    system(command.c_str());
  }
  {
    string nameBackup = fname + "_mems_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good())
      die("Unable to open save into file %s\n", nameBackup.c_str());

    for(int agentID=0; agentID<nAgents; agentID++) {
      for (int j=0; j<nNeurons; j++) out << mem[agentID]->outvals[j] << "\n";
      for (int j=0; j<nStates;  j++) out << mem[agentID]->ostates[j] << "\n";
    }

    out.flush();
    out.close();
    string command = "cp " + nameBackup + " " + fname + "_mems";
    system(command.c_str());
  }
}
#endif
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
        if (!out.good()) die("Unable to open save into file %s\n", nameOut_Mems.c_str());
        for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameNeurons.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameNeurons.c_str());
        for (Uint j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameMemories.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameMemories.c_str());
        for (Uint j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    dump_ID[agentID]++;
}
#if 0
bool Network::restart(const string fname)
{
    {
      string nameBackup = fname;
      ifstream in(nameBackup.c_str());
      debug1("Reading from %s\n", nameBackup.c_str());
      if (!in.good()) {
          error("Couldnt open file %s \n", nameBackup.c_str());
          return false;
      }

      int readTotWeights, readTotBiases, readNNeurons, readNLayers;
      in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;

      if (readTotWeights != nWeights || readTotBiases != nBiases || readNLayers != nLayers || readNNeurons != nNeurons)
      die("Network parameters differ!");

      Real tmp;
      for (int i=0; i<nWeights; i++) {
          in >> tmp;
          if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
          weights[i] = tmp;
      }

      for (int i=0; i<nBiases; i++) {
          in >> tmp;
          if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
          biases[i] = tmp;
      }
      in.close();
      updateFrozenWeights();
    }
    {
      string nameBackup = fname + "_mems";
      ifstream in(nameBackup.c_str());
      debug1("Reading from %s\n", nameBackup.c_str());
      if (!in.good()) {
          error("Couldnt open file %s \n", nameBackup.c_str());
          return false;
      }

      Real tmp;
      for(int agentID=0; agentID<nAgents; agentID++) {
        for (int j=0; j<nNeurons; j++) {
          in >> tmp;
          if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
          mem[agentID]->outvals[j] = tmp;
        }
        for (int j=0; j<nStates; j++) {
          in >> tmp;
          if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
          mem[agentID]->ostates[j] = tmp;
        }
      }
      in.close();
    }
    return true;
}
#endif
