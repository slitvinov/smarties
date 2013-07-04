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
#include <boost/numeric/ublas/matrix.hpp>


#include "../rng.h"

using namespace std;
namespace ublas = boost::numeric::ublas;

//namespace ANN
//{
	class Layer;
	class Link;
	class Neuron;
	class ActivationFunction;
	
	class Network
	{
	protected:
		
		int nInputs;
		int nOutputs;
		int nLayers;
		
		double eta;
		double alpha;
		RNG rng;
		
		vector<Layer*>  layers;
		vector<double*> inputs;
		vector<double*> outputs;
		vector<double*> errors;
		
	public:
		
		Network(vector<int>& layerSize, double eta, double alpha);
		virtual void predict(const vector<double>& inputs, vector<double>& outputs);
		virtual void improve(const vector<double>& errors);		
	};

	class NetworkLM : public Network
	{
	private:
		double mu, muFactor, muMin, muMax;
		int totWeights;
		int batchSize;
		int nInBatch;
		
		ublas::matrix<double> J;
		ublas::matrix<double> tmp;
		ublas::matrix<double> I;
		
		ublas::vector<double> e;
		ublas::vector<double> dw;
		
		vector< vector<double> > batch;
		vector< vector<double> > batchOut;
		vector< vector<double> > batchExact;
		
	public:
		NetworkLM(vector<int>& layerSize, double muFactor, int batchSize);
		NetworkLM(vector<int>& layerSize, int batchSize);
		void predict(const vector<double>& inputs, vector<double>& outputs);
		void improve(const vector<double>& errors);
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




