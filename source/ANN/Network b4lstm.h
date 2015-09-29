/*
 *  Network.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 20.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <cmath>
#include <armadillo>

#include "Approximator.h"
#include "../rng.h"

using namespace std;

//namespace ANN
//{
	class Layer;
	class Link;
	class Neuron;
	class ActivationFunction;
	
	class Network: public Approximator
	{
	protected:
		
		int nInputs, nOutputs, nLayers;
		
		double eta;
		double alpha;
		RNG rng;
        int batchSize, nInBatch;
        int totWeights;
		
		vector<Layer*>  layers;
		vector<double*> inputs;
		vector<double*> outputs;
		vector<double*> errors;
        
        arma::mat J;
		arma::mat JtJ;
		arma::mat tmp;
		arma::mat I;
		
		arma::vec e;
		arma::vec dw;
		arma::vec Je;
		
	public:
		
		Network(vector<int>& layerSize, double eta, double alpha, int batchSize = -1);
		void predict(const vector<double>& inputs, vector<double>& outputs);
		void improve(const vector<double>& inputs, const vector<double>& errors);
		
		void save(string fname) {;}
		bool restart(string fname) { return false; }
        void setBatchsize(int size);
	};

	class NetworkLM : public Network
	{
	private:
		double mu, muFactor, muMin, muMax;
		
		double Q;
		
		vector< vector<double> > batch;
		vector< vector<double> > batchOut;
		vector< vector<double> > batchExact;
		
	public:
        
		NetworkLM(vector<int>& layerSize, double muFactor, int batchSize = -1);
		void improve(const vector<double>& inputs, const vector<double>& errors);
		inline void   rollback();
		inline double getQ()      { return Q; }
		inline bool   isUpdated() { return nInBatch == 0; }
		
		void save(string fname);
		bool restart(string fname);
        
	};


	class Layer
	{
	public:
		int nNeurons;
		vector<Neuron*>  neurons;
		
		Layer(int nNeurons, ActivationFunction* func);
		void propagate();
		void backPropagate();
		void connect(Layer* next, RNG* rng);
		void connect2inputs(vector<double*>& vals);
		void connect2outputs(vector<double*>& vals);
		void connect2errors(vector<double*>& errs);
		void adjust(double eta, double alpha);
	};
	
	class Link
	{
	public:
		Neuron* neuronTo;
		Neuron* neuronFrom;
		
		double  w;
		double  val;
		double  prevDw;
	};
	
	class Neuron
	{
	public:
		ActivationFunction* func;

		vector<Link*>  inLinks;
		vector<Link*>  outLinks;
		
		double ival;
		double oval;
		double err;
		bool hasInputs;
		bool hasOutputs;
		
		Neuron(ActivationFunction* func);		
		void exec();
		void backExec();
		void adjust(double eta, double alpha);
	};
	
	class ActivationFunction
	{
	public:
		virtual double eval(double& arg) = 0;
		virtual double evalDiff(double& arg) = 0;
	};

	class Tanh : public ActivationFunction
	{
	public:
		inline double eval(double& arg)
		{
			if (arg > 20)  return 1;
			if (arg < -20) return -1;
			
			double ex = exp(arg);
			double e_x = exp(-arg);
			
			return (ex - e_x) / (ex + e_x);
		}
		
		inline double evalDiff(double& arg)
		{
			if (arg > 20 || arg < -20) return 0;
			
			double e2x = exp(2*arg);
			double t = (e2x + 1);
			
			return 4*e2x/(t*t);
		}
	};

	class Linear : public ActivationFunction
	{
	public:
		inline double eval(double& arg)
		{
			return arg;
		}
		
		inline double evalDiff(double& arg)
		{
			return 1;
		}
	};

//}




