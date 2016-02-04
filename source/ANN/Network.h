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
#include "../Settings.h"
using namespace std;

//namespace ANN
//{
	class Layer;
	class Link;
    class Neuron
	class ActivationFunction;



	class Network: public Approximator
	{
	protected:
		
		int nInputs, nOutputs, nLayers;
		
		vt eta;
		vt alpha;
        vt lambda;
		RNG rng;
        int batchSize, nInBatch;
        int totWeights;
		
		vector<Layer*>  layers;
		vector<vt*> inputs;
		vector<vt*> outputs;
		vector<vt*> errors;
        
        arma::mat J;
		arma::mat JtJ;
		arma::mat tmp;
		arma::mat I;
		
		arma::vec e;
		arma::vec dw;
		arma::vec Je;
		
	public:
		Network(vector<int>& layerSize, vt eta, vt alpha,vt lambda = 0, int batchSize = -1);
		void predict(const vector<vt>& inputs, vector<vt>& outputs, int nAgent = 0);
		void improve(const vector<vt>& inputs, const vector<vt>& errors, int nAgent = 0);
		void save(string fname) {;}
		bool restart(string fname) { return false; }
        void setBatchsize(int size);
	};

	class NetworkLM : public Network
	{
	private:
		vt mu, muFactor, muMin, muMax;
		
		vt Q;
		
		vector< vector<vt> > batch;
		vector< vector<vt> > batchOut;
		vector< vector<vt> > batchExact;
		
	public:
        
		NetworkLM(vector<int>& layerSize, vt muFactor, int batchSize = -1);
		void improve(const vector<vt>& inputs, const vector<vt>& errors, int nAgent = 0);
		inline void   rollback();
		inline vt getQ()      { return Q; }
		inline bool   isUpdated() { return nInBatch == 0; }
		
		void save(string fname);
		bool restart(string fname);
        
	};


class Link
	{
	public:
		Neuron* neuronTo;
		Neuron* neuronFrom;
		
		vt  w;
        vt  Dw;
		vt  val;
        vt  err;
		vt  prevDw;
	};
	
class Neuron
	{
	public:
		ActivationFunction* func;

		vector<Link*>  inLinks;
		vector<Link*>  outLinks;
		
        vt ival, err, oval;

		bool hasInputs, hasOutputs;
		
		Neuron(ActivationFunction* func);		
		void exec();
		void backExec();
		void adjust(vt eta, vt alpha, vt lambda=0);
    };

class Layer
    {
        public:
        int nNeurons;
        vector<Neuron*> neurons;
        
        Layer(int nNeurons, ActivationFunction* func);
        void propagate();
        void backPropagate();
        void connect(Layer* next, RNG* rng);
    
        void connect2inputs(vector<vt*>& vals);
        void connect2outputs(vector<vt*>& vals);
        void connect2errors(vector<vt*>& errs);
        void adjust(vt eta, vt alpha, vt lambda);
    };

class ActivationFunction
	{
	public:
		virtual vt eval(vt& arg) = 0;
		virtual vt evalDiff(vt& arg) = 0;
	};

class Tanh : public ActivationFunction
	{
	public:
		inline vt eval(vt& arg)
		{
			if (arg > 20)  return 1;
			if (arg < -20) return -1;
            vt e2x = exp(2.*x);
            return (e2x - 1.) / (e2x + 1.);
		}
		
		inline vt evalDiff(vt& arg)
		{
			if (arg > 20 || arg < -20) return 0;
			
			vt e2x = exp(2.*arg);
			vt t = (e2x + 1.);
			return 4*e2x/(t*t);
		}
	};

class Tanh2 : public ActivationFunction
    {
    public:
        inline vt eval(vt& arg)
        {
            if (arg > 20)  return 2;
            if (arg < -20) return -2;
            vt e2x = exp(2.*x);
            return 2.*(e2x - 1.) / (e2x + 1.);
        }
        
        inline vt evalDiff(vt& arg)
        {
            if (arg > 20 || arg < -20) return 0;
            
            vt e2x = exp(2.*arg);
            vt t = (e2x + 1.);
            return 8.*e2x/(t*t);
        }
    };

class Sigm : public ActivationFunction
    {
    public:
        inline vt eval(vt& arg)
        {
            if (arg > 20)  return 1;
            if (arg < -20) return 0;
            
            vt e_x = exp(-arg);
            return 1. / (1. + e_x);
        }
        
        inline vt evalDiff(vt& arg)
        {
            if (arg > 20 || arg < -20) return 0;
            
            vt ex = exp(arg);
            vt e2x = (1. + ex)*(1. + ex);
            
            return ex/e2x;
        }
    };

class Linear : public ActivationFunction
	{
	public:
		inline vt eval(vt& arg)
		{
			return arg;
		}
		
		inline vt evalDiff(vt& arg)
		{
			return 1;
		}
	};

class Gaussian : public ActivationFunction
    {
    public:
        inline vt eval(vt& x)
        {
            if (std::isnan(x) || std::isinf(x)) return 0;
            if (x > 5 || x < -5) return 0;
            return exp(-10.*x*x);
        }
        inline vt evalDiff(vt& x)
        {
            if (std::isnan(x) || std::isinf(x)) return 0;
            if (x > 5 || x < -5) return 0;
            return -20. * x * exp(-10.*x*x);
        }
    };
//}
