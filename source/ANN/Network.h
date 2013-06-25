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

#include "../rng.h"

using namespace std;

//namespace ANN
//{
	class Layer;
	class Link;
	class Neuron;
	class ActivationFunction;
	
	class Network
	{
	private:
		
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
		
		Network(vector<int> layerSize, double eta, double alpha);
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
			double ex = exp(arg);
			double e_x = exp(-arg);
			
			return (ex - e_x) / (ex + e_x);
		}
		
		inline double evalDiff(double& arg)
		{
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




