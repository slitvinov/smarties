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
    class HiddenLayer;
	class Link;
	class Neuron;
    class MemoryCell;
    class MemoryBlock;
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
		Network(vector<int>& layerSize, double eta, double alpha,double lambda = 0, int batchSize = -1);
		void predict(const vector<double>& inputs, vector<double>& outputs, int nAgent = 0);
		void improve(const vector<double>& inputs, const vector<double>& errors, int nAgent = 0);
        void predict(const vector<double>& inputs, const vector<double>& memoryin, const vector<double>& ostate, vector<double>& nstate,  vector<double>& outputs) {;}
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
		void improve(const vector<double>& inputs, const vector<double>& errors, int nAgent = 0);
		inline void   rollback();
		inline double getQ()      { return Q; }
		inline bool   isUpdated() { return nInBatch == 0; }
		
		void save(string fname);
		bool restart(string fname);
        
	};

class NetworkLSTM: public Approximator
    {
    protected:
        
        int nInputs, nOutputs, nLayers, nAgents, nMems, nRecurr; //each agents has its memory T_T
        
        double eta;
        double alpha;
        double kappa;
        RNG rng;
        
        vector<HiddenLayer*> layers;
        //vector<Layer*> layers;
        vector<double*> inputs;
        vector<double*> outputs;
        vector<double*> errors;
        vector<double*> memory_in;
        vector<double*> memory_out;
        vector<double*> o_state;
        vector<double*> n_state;
        
    public:
        
        double TotSumWeights();
        NetworkLSTM(vector<int>& layerSize, vector<int>& memorySize, vector<int>& nCellpB, double eta, double alpha, double lambda, double kappa, int nAgents);
        void predict(const vector<double>& input, vector<double>& output, int nAgent = 0);
        void predict(const vector<double>& inputs, const vector<double>& memoryin, const vector<double>& ostate, vector<double>& nstate,  vector<double>& outputs);
        void improve(const vector<double>& inputs, const vector<double>& error, int nAgent = 0);
        void setBatchsize(int size) {cout << TotSumWeights() << endl;}
        void save(string fname);
        bool restart(string fname) { return false; }
    };

class Link
	{
	public:
		Neuron* neuronTo;
		Neuron* neuronFrom;
		
		double  w;
        double  Dw; //only for memory updates
		double  val;
        double  err;
        double  epsilon;
        double  dsdw;
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
		double err, delta, epsilon;
		bool hasInputs;
		bool hasOutputs;
		
		Neuron(ActivationFunction* func);		
		void exec();
		void backExec();
		void adjust(double eta, double alpha, double lambda=0);
        void adjust(double error, double eta, double alpha, double lambda, double kappa);
	};

class MemoryCell : public Neuron
    {
    public:
        
        double Sc, Sc_old, Sc_new; //Need to change per each learner, old and new handle communication
        double sumwd, OGerrfac, FGerrfac, IGerrfac;
        vector<double> dsdw_IN, dsdw_IG, dsdw_FG;
        vector<double> dsdw_INo, dsdw_IGo, dsdw_FGo; //this is getting silly

        Neuron* Input;
        //input, forget, output gates
        Neuron* IG;
        Neuron* FG;
        Neuron* OG;
        
        //quick fix for peephole
        Neuron* ScN;
        Neuron* ScO;
        
        MemoryCell();
        
        void init_dsdw();
        void exec();
        void backExec();
    };

class MemoryBlock
    {
        public:
        int nMemoryCells;
        vector<MemoryCell*> mCells;
        //input, forget, output gates
        Neuron* IG;
        Neuron* FG;
        Neuron* OG;
        
        MemoryBlock(int nCellpB);
        void init_dsdw();
        void exec();
        void backExec();
        void adjust(double error, double eta, double alpha, double lambda, double kappa);
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
    
        void connect2inputs(vector<double*>& vals);
        void connect2outputs(vector<double*>& vals);
        void connect2errors(vector<double*>& errs);
        void adjust(double eta, double alpha, double lambda);
    };

class HiddenLayer
    {
    public:
        int nNeurons;
        int nMemoryBlocks, nCellpB;
        Neuron* basePos;
        Neuron* baseNeg;
        vector<Neuron*> neurons;
        vector<MemoryBlock*> mBlocks;
        
        HiddenLayer(int nNeurons, ActivationFunction* func);
        HiddenLayer(int nBlocks, int nCellpB, int nNeurons, ActivationFunction* func, RNG* rng);
        void init_dsdw();
        void propagate();
        void backPropagate();
        double TotSumWeights();
        void link(Neuron* Nto, Neuron* Nin, RNG* rng, bool ground);
        void normaliseWeights();
        void connect2layers(HiddenLayer* prev, RNG* rng, int dist);
        void connect2memstate(vector<double*>& memory, vector<double*>& Sc_old, vector<double*>& Sc_new, int firstm, int firstr);
        void connect2ground(RNG* rng);
        void connect2inputs(vector<double*>& vals, vector<double*>& mems);
        void connect2outputs(vector<double*>& vals);
        void connect2errors(vector<double*>& errs);
        void adjust(double error, double eta, double alpha, double lambda, double kappa);
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

class Tanh2 : public ActivationFunction
    {
    public:
        inline double eval(double& arg)
        {
            if (arg > 20)  return 2;
            if (arg < -20) return -2;
            
            double ex = exp(arg);
            double e_x = exp(-arg);
            
            return 2.*(ex - e_x) / (ex + e_x);
        }
        
        inline double evalDiff(double& arg)
        {
            if (arg > 20 || arg < -20) return 0;
            
            double e2x = exp(2*arg);
            double t = (e2x + 1);
            
            return 8.*e2x/(t*t);
        }
    };

class Sigm : public ActivationFunction
    {
    public:
        inline double eval(double& arg)
        {
            if (arg > 20)  return 1;
            if (arg < -20) return 0;
            
            double e_x = exp(-arg);
            
            return 1. / (1. + e_x);
        }
        
        inline double evalDiff(double& arg)
        {
            if (arg > 20 || arg < -20) return 0;
            
            double ex = exp(arg);
            double e2x = (1. + ex)*(1. + ex);
            
            return ex/e2x;
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




