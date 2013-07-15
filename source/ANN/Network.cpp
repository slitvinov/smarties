/*
 *  Network.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 20.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>
#include <iostream>

#include "Network.h"
#include "../ErrorHandling.h"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/operations.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>


using namespace ErrorHandling;

//namespace ANN
//{
	Network::Network(vector<int>& layerSize, double eta, double alpha) :
	nInputs(layerSize.front()), nOutputs(layerSize.back()), nLayers(layerSize.size()), eta(eta), alpha(alpha), rng(0)
	{
		Layer* first = new Layer(nInputs, new Linear);
		layers.push_back(first);
		
		for (int i=1; i<nLayers; i++)
		{
			Layer* l = (i == nLayers - 1) ? new Layer(layerSize[i],  new Linear) : new Layer(layerSize[i],  new Tanh);
			Layer* prev = layers.back();
			layers.push_back(l);
			prev->connect(l, &rng);
		}

		inputs.resize(nInputs);
		outputs.resize(nOutputs);
		errors.resize(nOutputs);
		
		first->connect2inputs(inputs);
		layers.back()->connect2outputs(outputs);
		layers.back()->connect2errors(errors);
	}
	
	void Network::predict(const vector<double>& inputs, vector<double>& outputs)
	{
		for (int i=0; i<nInputs; i++)
			*(this->inputs[i]) = inputs[i];
		
		for (int i=0; i<nLayers; i++)
			layers[i]->propagate();
		
		for (int i=0; i<nOutputs; i++)
			outputs[i] = *(this->outputs[i]);
	}

	void Network::improve(const vector<double>& errors)
	{
		for (int i=0; i<nOutputs; i++)
			*(this->errors[i]) = errors[i];
		
		for (int i=nLayers-1; i>=0; i--)
			layers[i]->backPropagate();
		
		for (int i=0; i<nLayers; i++)
			layers[i]->adjust(eta, alpha);
	}


	NetworkLM::NetworkLM(vector<int>& layerSize, double muFactor, int batchSize) :
	Network(layerSize, 0, 0), muFactor(muFactor), batchSize(batchSize)
	{
		totWeights = 0;
		for (int i=0; i<nLayers-1; i++)
		{
			totWeights += layers[i]->nNeurons * (layers[i+1]->nNeurons - 1);
		}
		
		J.resize(nOutputs*batchSize, totWeights);
		I = ublas::identity_matrix<double>(totWeights);
		
		mu = 0.01;
		nInBatch = 0;
		
		e.resize(batchSize*nOutputs);
		dw.resize(totWeights);
		
		muMax = 1e10;
		muMin = 1e-1;
	}

	NetworkLM::NetworkLM(vector<int>& layerSize, int batchSize) : NetworkLM(layerSize, 1 + batchSize/100, batchSize) {};

	void NetworkLM::predict(const vector<double>& inputs, vector<double>& outputs)
	{
		for (int i=0; i<nInputs; i++)
			*(this->inputs[i]) = inputs[i];
		
		for (int i=0; i<nLayers; i++)
			layers[i]->propagate();
		
		for (int i=0; i<nOutputs; i++)
			outputs[i] = *(this->outputs[i]);
		
		batch.push_back(inputs);
		batchOut.push_back(outputs);
	}

	void NetworkLM::improve(const vector<double>& errors)
	{
		vector<double> tmpVec(nOutputs);
		for (int i=0; i<nOutputs; i++)
			tmpVec[i] = batchOut.back()[i] - errors[i];
		batchExact.push_back(tmpVec);
		
		for (int i=0; i<nOutputs; i++)
			e(i + nInBatch*nOutputs) = errors[i];
		
		for (int i=0; i<nOutputs; i++)
		{
			for (int j=0; j<nOutputs; j++)
				*(this->errors[j]) = (i==j) ? 1 : 0;  // !!!!!!!!!!!!!!!!!!!!!!
			
			for (int i=nLayers-1; i>=0; i--)
				layers[i]->backPropagate();
			
			int w = 0;
			for (int l=0; l<nLayers; l++)
				for (int n=0; n<layers[l]->nNeurons; n++)
					for (int lnk=0; lnk<layers[l]->neurons[n]->inLinks.size(); lnk++)
						J(i + nInBatch*nOutputs, w++) = -layers[l]->neurons[n]->err * layers[l]->neurons[n]->inLinks[lnk]->neuronFrom->oval;
			
		}
		
		nInBatch++;
		
		if (nInBatch == batchSize)
		{
			nInBatch = 0;
			Q = 0;
			double avgErr = 0;
			double minE = 1e10;
			double maxE = -1e10;
			double newMaxE = 0;
			for (int i=0; i<nOutputs*batchSize; i++)
			{
				Q += e(i) * e(i);
				avgErr += abs(e(i));
				minE = (abs(e(i)) < minE) ? abs(e(i)) : minE;
				maxE = (abs(e(i)) > maxE) ? abs(e(i)) : maxE;
			}
			
			double avgRelErr = 0;
			int w = 0;
			for (int i=0; i<batchSize; i++)
				for (int j=0; j<nOutputs; j++)
					avgRelErr += abs(e(w++)) / abs(batchOut[i][j]);
			
			avgRelErr /= batchSize * nOutputs;
			avgErr    /= batchSize * nOutputs;
			
			double Q0 = Q;
			Q = Q0+1;
			
			//info("Max, minerr, avgrelerr: %f %f %f\n",maxE, minE, avgErr);
			
			w = 0;
			//debug("\n\n\n**************************************************************************************\n");
			//for (int i=0; i<batchSize; i++)
			//{
			//	debug("Inp: [");
			//	for (int j=0; j<nInputs; j++)
			//		debug("%f ", batch[i][j]);
			//	debug("];  Out: %f;  Err: %f;  Exact %f\n", batchOut[i][0], e(w++), batchExact[i][0]);
			//}
				 
			//if (avgRelErr < 0.01 && maxE < 10*minE)
//			{
//				info ("Max, minerr, avgrelerr: %f %f %f\n",maxE, minE, avgRelErr);
//				for (int b=0; b<batchSize; b++)
//				{
//					batch[b].clear();
//					batchOut[b].clear();
//					batchExact[b].clear();
//				}
//				
//				batch.clear();
//				batchOut.clear();
//				batchExact.clear();
//				return;
//			}
			
			JtJ = ublas::prod(ublas::trans(J), J);
			
			while (newMaxE > maxE || Q > Q0)
			{
				dw  = ublas::prod(ublas::trans(J), e);
				tmp = JtJ + mu*I;
				//cout << J << endl << tmp << endl << dw << endl;
				
				ublas::permutation_matrix<double> piv(tmp.size1()); 
				ublas::lu_factorize(tmp, piv);
				ublas::lu_substitute(tmp, piv, dw);
				//cout << dw << endl;
				
				bool nan = false;
				for (int w=0; w<totWeights; w++)
					if (std::isnan((double)(dw(w))) || std::isinf((double)(dw(w))))
						nan = true;
				if (nan)
				{
					mu *= muFactor;
					Q = Q0+1;
					continue;
				}
				
				int w = 0;
				for (int l=0; l<nLayers; l++)
					for (int n=0; n<layers[l]->nNeurons; n++)
						for (int lnk=0; lnk<layers[l]->neurons[n]->inLinks.size(); lnk++)
						{
							debug2("l%d w%d%d = %f\n", l, lnk, n, layers[l]->neurons[n]->inLinks[lnk]->w); 
							layers[l]->neurons[n]->inLinks[lnk]->w += dw(w++);
						}
							
				
				Q = 0;
				newMaxE = -1e10;
				for (int i=0; i<batchSize; i++)
				{
					Network::predict(batch[i], tmpVec);
					
					for (int j=0; j<nOutputs; j++)
					{
						double diff = tmpVec[j] - batchExact[i][j];
						Q += diff * diff;
						newMaxE = (abs(diff) > newMaxE) ? abs(diff) : newMaxE;
					}
				}
				
				newMaxE = 0;
				
				if (newMaxE > maxE || Q > Q0)
				{
					if (mu < muMax)
						mu *= muFactor;
					else
					{
						break;
					}
					rollback();
				}
			}
			//vector<double> invec(6);
//			invec[0] = invec[1] = -2;
//			invec[2] = invec[3] = 2;
//			invec[4] = 3.1;
//			invec[5] = -1.98;
//			Network::predict(invec, tmpVec);
//			
//			if (tmpVec[0] > 1)
//			{
//				debug("  !!! %f\n", tmpVec[0]);
//				for (int i=0; i<batchSize; i++)
//				{
//					Network::predict(batch[i], tmpVec);
//					
//					//if (tmpVec[0] > 0.5)
//					{
//					debug("Inp: [");
//					for (int j=0; j<nInputs; j++)
//						debug("%f ", batch[i][j]);
//					debug("]; %f --> %f;  exact: %f\n",  batchOut[i][0], tmpVec[0], batchExact[i][0]);
//					}
//				}
//				debug("Q0 %f, Q %f,   E0 %f, E %f\n", Q0, Q, maxE, newMaxE);
//				if (debugLvl > 2) cout << dw << endl << endl;
//
//			}
			
			if (mu > muMin) mu /= muFactor;
			
			if (batch.size() != batchSize || batchExact.size() != batchSize || batchOut.size() != batchSize)
				die("Ololo looooooser\n");
			for (int b=0; b<batchSize; b++)
			{
				batch[b].clear();
				batchOut[b].clear();
				batchExact[b].clear();
			}
			
			batch.clear();
			batchOut.clear();
			batchExact.clear();
		}
	}

	inline void NetworkLM::rollback()
	{
		int w = 0;
		for (int l=0; l<nLayers; l++)
			for (int n=0; n<layers[l]->nNeurons; n++)
				for (int lnk=0; lnk<layers[l]->neurons[n]->inLinks.size(); lnk++)
					layers[l]->neurons[n]->inLinks[lnk]->w -= dw(w++);		
	}
	
	Layer::Layer(int nNeurons, ActivationFunction* func) : nNeurons(nNeurons+1)
	{
		neurons.resize(this->nNeurons);
		for (int i=0; i < this->nNeurons; i++)
			neurons[i] = new Neuron(func);
		neurons[this->nNeurons-1]->ival = -1;
	}
	
	void Layer::connect(Layer* next, RNG* rng)
	{
		for (int i=0; i<nNeurons; i++)
			for (int j=0; j<next->nNeurons-1; j++)
			{
				Link* lnk = new Link;
				lnk->neuronFrom = neurons[i];
				lnk->neuronTo   = next->neurons[j];
				lnk->w = rng->uniform(-1/(2.0*nNeurons), 1/(2.0*nNeurons));
				lnk->val    = 0;
				lnk->prevDw = 0;
				
				neurons[i]->outLinks.push_back(lnk);
				neurons[i]->hasOutputs = true;
				next->neurons[j]->inLinks.push_back(lnk);
				next->neurons[j]->hasInputs = true;
			}
		
	}
	
	void Layer::connect2inputs(vector<double*>& vals)
	{
		for (int i=0; i<nNeurons-1; i++)
			vals[i] = &(neurons[i]->ival);
	}

	void Layer::connect2outputs(vector<double*>& vals)
	{
		for (int i=0; i<nNeurons-1; i++)
			vals[i] = &(neurons[i]->oval);
	}
	
	void Layer::connect2errors(vector<double*>& errs)
	{
		for (int i=0; i<nNeurons-1; i++)
			errs[i] = &(neurons[i]->err);
	}
	
	void Layer::propagate()
	{
		for (int i=0; i<nNeurons; i++)
			neurons[i]->exec();
	}
	
	void Layer::backPropagate()
	{
		for (int i=0; i<nNeurons; i++)
			neurons[i]->backExec();
	}
	
	void Layer::adjust(double eta, double alpha)
	{
		for (int i=0; i<nNeurons; i++)
			neurons[i]->adjust(eta, alpha);
	}
	
	Neuron::Neuron(ActivationFunction* func) :
	hasInputs(false), hasOutputs(false), func(func) { };
	
	void Neuron::exec()
	{
		if (hasInputs)
		{
			ival = 0;
			for (int i=0; i<inLinks.size(); i++)
				ival += inLinks[i]->val * inLinks[i]->w;
		}
		
		oval = func->eval(ival);
		
		if (hasOutputs)
		{
			for (int i=0; i<outLinks.size(); i++)
				outLinks[i]->val = oval;
		}
	}
	
	void Neuron::backExec()
	{
		if (hasOutputs)
		{
			err = 0;
			for (int i=0; i<outLinks.size(); i++)
				err += outLinks[i]->val * outLinks[i]->w;
		}
		
		err = func->evalDiff(ival) * err;
		
		if (hasInputs)
		{
			for (int i=0; i<inLinks.size(); i++)
				inLinks[i]->val = err;
		}
	}
	
	void Neuron::adjust(double eta, double alpha)
	{
		if (hasInputs)
		{
			for (int i=0; i<inLinks.size(); i++)
			{
				Neuron* prev = inLinks[i]->neuronFrom;
				inLinks[i]->w += -eta * err * prev->oval + alpha * inLinks[i]->prevDw;
				inLinks[i]->prevDw = -eta * err * prev->oval;
			}
		}
	}	
//}



