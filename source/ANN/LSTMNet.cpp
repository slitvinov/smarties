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
#include <cstdlib>

#include "LSTMNet.h"
#include "../ErrorHandling.h"
using namespace ErrorHandling;
//#define DBG_EXEC
//#define DBG_BACK //panico
//#define DBG_ADJS
//#define DBG_INPUT

NormalLayer::NormalLayer(int layerSize, int normal_1st_link, int normal_n_links, int normal_pos, int n1stWeightHL, int n1stBiasHL, ActivationFunction* f) : nNeurons(layerSize), n1stLink(normal_1st_link), nLinks(normal_n_links), n1stNeuron(normal_pos), n1stWeight(n1stWeightHL), n1stBias(n1stBiasHL), func(f)
{  }

LSTMLayer::LSTMLayer(int recurSize, int recurr_1st_new_link, int recurr_n_new_links, int recurr_1st_old_link, int recurr_n_old_links, int recurr_tot_links, int recurr_pos, int indIG, int indFG, int indOG, int indState, int n1stWeightIG, int n1stWeightFG, int n1stWeightIN, int n1stWeightOG, int n1stWeightPeep, int n1stBiasIG, int n1stBiasFG, int n1stBiasIN, int n1stBiasOG, int n1stdSdWBias, int n1stdSdWIG, int n1stdSdWFG, int n1stdSdWIN, ActivationFunction* f) :
NormalLayer(recurSize, recurr_1st_new_link, recurr_n_new_links, recurr_pos, n1stWeightIN, n1stBiasIN, f),
n1stOld(recurr_1st_old_link), nOldL(recurr_n_old_links), nIG(indIG), nFG(indFG), nOG(indOG), n1stState(indState), n1stIG(n1stWeightIG), n1stFG(n1stWeightFG), n1stOG(n1stWeightOG), n1stPeep(n1stWeightPeep), n1stIGB(n1stBiasIG), n1stFGB(n1stBiasFG), n1stOGB(n1stBiasOG), n1dsdB(n1stdSdWBias), n1dsIG(n1stdSdWIG), n1dsFG(n1stdSdWFG), n1dsIN(n1stdSdWIN), sigm(new SoftSigm)
{ }

FishNet::FishNet(vector<int>& layerSize, vector<int>& recurSize, Settings settings, int nAgents = 1) :
eta(settings.nnEta), alpha(settings.nnAlpha), lambda(settings.nnLambda), kappa(settings.nnKappa), AdFac(settings.nnAdFac), nAgents(nAgents)
{
    beta_t_1 = 0.9; beta_t_2 = 0.999; beta_1 = 0.9; beta_2 = 0.999; epsilon = 1e-8;
    std::mt19937 gen(settings.randSeed);
    std::uniform_real_distribution<Real> dis(-1.,1.);
    
    int nMixedLayers = layerSize.size();
    for (int i=0; i<nMixedLayers; i++) /* TODO (more elegance) */
    {
        layerSize[i] = ceil( (Real)layerSize[i]/SIMD )*SIMD;
        recurSize[i] = ceil( (Real)recurSize[i]/SIMD )*SIMD;
    }
    /* Have fun debugging, me! */
    nInputs = layerSize.front();
    nOutputs = layerSize.back();
    nLayers = 0;
    nNeurons = 0;
    nWeights = 0;
    nBiases = 0; /* more than ovals because also gates have biases */
    ndScelldW = 0;
    nGates = 0;
    nStates = 0;
    /* Count number of links and index of first link for each layer type. new means t, old means t-1 */
    vector<int> recurr_1st_new_link(nLayers), recurr_n_new_links(nLayers), recurr_1st_old_link(nLayers), recurr_n_old_links(nLayers), normal_1st_link(nLayers), normal_n_links(nLayers), recurr_tot_links(nLayers), recurr_pos(nLayers), normal_pos(nLayers);
    /* just where the layer's gates stand along the vector of igates and ogates: ez */
    vector<int> indIG(nLayers), indFG(nLayers), indOG(nLayers), indState(nLayers);
    /* location of first weight (bias) for gates, cell inputs and normal layer, used also for FD */
    vector<int> n1stWeightIG(nLayers), n1stWeightFG(nLayers), n1stWeightIN(nLayers), n1stWeightOG(nLayers), n1stWeightPeep(nLayers), n1stWeightHL(nLayers);
    vector<int> n1stBiasIG(nLayers), n1stBiasFG(nLayers), n1stBiasIN(nLayers), n1stBiasOG(nLayers), n1stBiasHL(nLayers);
    vector<int> n1stdSdWBias(nLayers), n1stdSdWIG(nLayers), n1stdSdWFG(nLayers), n1stdSdWIN(nLayers);
    for (int i=1; i<nMixedLayers; i++)
    { //layer 0 is the input layer
        /* connected to whole prev layer */
        recurr_1st_new_link[i] = nNeurons;
        recurr_n_new_links[i] = recurSize[i-1] + layerSize[i-1];
        /* connected to prev layer and this recur layer */
        normal_1st_link[i] = nNeurons;
        normal_n_links[i] = recurSize[i]+recurSize[i-1]+layerSize[i-1];
        /* update number of outputs */
        nNeurons += recurSize[i-1] + layerSize[i-1];
        /* index of first node: recur are b4 normal */
        recurr_pos[i] = nNeurons;
        normal_pos[i] = nNeurons + recurSize[i];
        /* recur connected to previous realization of i+1 layer */
        recurr_1st_old_link[i] = nNeurons;
        recurr_n_old_links[i] = recurSize[i] + layerSize[i];
        recurr_tot_links[i] = recurr_n_old_links[i] + recurr_n_new_links[i];
        /* count gates, find where they are along igates and ogates, same for state */
        nGates += 3*recurSize[i];
        nStates += recurSize[i];
        indIG[i] = indIG[i-1] + 3*recurSize[i-1];
        indFG[i] = indIG[i-1] + 3*recurSize[i-1] +   recurSize[i];
        indOG[i] = indIG[i-1] + 3*recurSize[i-1] + 2*recurSize[i];
        indState[i] = indState[i-1] + recurSize[i-1];
        /* count all the weights */
        n1stWeightIG[i] = nWeights;
        n1stWeightFG[i] = n1stWeightIG[i] + recurr_tot_links[i]*recurSize[i];
        n1stWeightIN[i] = n1stWeightFG[i] + recurr_tot_links[i]*recurSize[i];
        n1stWeightOG[i] = n1stWeightIN[i] + recurr_tot_links[i]*recurSize[i];
        n1stWeightPeep[i] = n1stWeightIN[i] + recurr_tot_links[i]*recurSize[i];
        nWeights += 4*recurr_tot_links[i]*recurSize[i] + 3*recurSize[i]; //3 peepholes per cell
        n1stWeightHL[i] = nWeights;
        nWeights += normal_n_links[i]*layerSize[i];
        /* so much bias */
        n1stBiasIG[i] = nBiases;
        n1stBiasFG[i] = n1stBiasIG[i] + recurSize[i];
        n1stBiasIN[i] = n1stBiasFG[i] + recurSize[i];
        n1stBiasOG[i] = n1stBiasIN[i] + recurSize[i];
        nBiases += 4*recurSize[i];
        n1stBiasHL[i] = nBiases;
        nBiases += layerSize[i];
        /* count all the dSdW for the cells (biases and peeps included) */
        n1stdSdWBias[i] = ndScelldW;
        n1stdSdWIG[i] = ndScelldW + 5*recurSize[i]; /* 2 peeps and 3 biases */
        n1stdSdWFG[i] = n1stdSdWIG[i] + recurr_tot_links[i]*recurSize[i];
        n1stdSdWIN[i] = n1stdSdWFG[i] + recurr_tot_links[i]*recurSize[i];
        ndScelldW += (3*recurr_tot_links[i]+5)*recurSize[i];
        
        printf("LSTM layer %d: recurr_1st_new_link= %d, recurr_n_new_links= %d, recurr_pos= %d, recurr_1st_old_link= %d, recurr_n_old_links= %d, recurr_tot_links= %d, indIG= %d, indFG= %d, indOG= %d, indState= %d,  n1stWeightIG= %d, n1stWeightFG= %d, n1stWeightIN= %d, n1stWeightOG= %d, n1stWeightPeep= %d, n1stBiasIG= %d, n1stBiasFG= %d, n1stBiasIN= %d, n1stBiasOG= %d, n1stdSdWBias= %d, n1stdSdWIG= %d, n1stdSdWFG= %d, n1stdSdWIN= %d, n1stdSdWOG= %d", i, recurr_1st_new_link[i], recurr_n_new_links[i], recurr_pos[i], recurr_1st_old_link[i], recurr_n_old_links[i], recurr_tot_links[i], indIG[i], indFG[i], indOG[i], indState[i],  n1stWeightIG[i], n1stWeightFG[i], n1stWeightIN[i], n1stWeightOG[i], n1stWeightPeep[i], n1stBiasIG[i], n1stBiasFG[i], n1stBiasIN[i], n1stBiasOG[i], n1stdSdWBias[i], n1stdSdWIG[i], n1stdSdWFG[i], n1stdSdWIN[i]);
        printf("Normal layer %d: normal_1st_link= %d, normal_n_links= %d, normal_pos= %d, n1stWeightHL= %d, n1stBiasHL= %d", normal_1st_link[i], normal_n_links[i], normal_pos[i], n1stWeightHL[i], n1stBiasHL[i]);
    }
    nNeurons += recurSize[nMixedLayers-1] + layerSize[nMixedLayers-1];
    Agents.clear();
    for (int i = 0; i<nAgents; ++i)
    {
        Memory agent(nNeurons, nStates);
        Agents.push_back(agent);
    }
    /* allocate all the shits */
    oldvals = (Real*) _mm_malloc(nNeurons*sizeof(Real), ALLOC);
    outvals = (Real*) _mm_malloc(nNeurons*sizeof(Real), ALLOC);
    in_vals = (Real*) _mm_malloc(nNeurons*sizeof(Real), ALLOC);
    errvals = (Real*) _mm_malloc(nNeurons*sizeof(Real), ALLOC);
    weights = (Real*) _mm_malloc(nWeights*sizeof(Real), ALLOC);
    biases  = (Real*) _mm_malloc(nBiases *sizeof(Real), ALLOC);
    dsdw    = (Real*) _mm_malloc(ndScelldW*sizeof(Real),ALLOC);
    Dw      = (Real*) _mm_malloc(nWeights*sizeof(Real), ALLOC);
    Db      = (Real*) _mm_malloc(nBiases *sizeof(Real), ALLOC);
    igates  = (Real*) _mm_malloc(nGates  *sizeof(Real), ALLOC);
    ogates  = (Real*) _mm_malloc(nGates  *sizeof(Real), ALLOC);
    ostates = (Real*) _mm_malloc(nStates *sizeof(Real), ALLOC);
    nstates = (Real*) _mm_malloc(nStates *sizeof(Real), ALLOC);
    
    /* ADAM optimizer */
    _1stMomW = (Real*) _mm_malloc(nWeights*sizeof(Real), ALLOC);
    _1stMomB = (Real*) _mm_malloc(nBiases *sizeof(Real), ALLOC);
    _2ndMomW = (Real*) _mm_malloc(nWeights*sizeof(Real), ALLOC);
    _2ndMomB = (Real*) _mm_malloc(nBiases *sizeof(Real), ALLOC);
    /* Assumption: if a normal and a LSTM layer have the same index, LSTM happens first */
    for (int i=1; i<nMixedLayers; i++)
    {
        if (recurSize[i]>0)
        {
            /* initialize shits see Glorot 2010 */
            for (int w=n1stWeightIG[i]; w<n1stWeightIG[i]+(4*recurr_tot_links[i]+3)*recurSize[i]; w++)
            weights[w] = dis(gen)*sqrt(6)/(recurr_tot_links[i] + layerSize[i] + recurSize[i]);
            
            for (int w=n1stBiasIG[i]; w<n1stBiasIG[i]+recurSize[i]; w++)
            biases[w] = -10.; /* IG starts decisively closed */
            for (int w=n1stBiasFG[i]; w<n1stBiasFG[i]+recurSize[i]; w++)
            biases[w] =  10.; /* FG starts decisively open */
            for (int w=n1stBiasIN[i]; w<n1stBiasIN[i]+recurSize[i]; w++)
            biases[w] = dis(gen)*sqrt(6)/(recurr_tot_links[i] + layerSize[i] + recurSize[i]);
            for (int w=n1stBiasOG[i]; w<n1stBiasOG[i]+recurSize[i]; w++)
            biases[w] = -10.; /* OG starts decisively closed */
            
            layers.push_back( (i+1!=nMixedLayers || layerSize[i]>0) ?
                             new LSTMLayer(recurSize[i], recurr_1st_new_link[i], recurr_n_new_links[i], recurr_1st_old_link[i], recurr_n_old_links[i], recurr_tot_links[i], recurr_pos[i], indIG[i], indFG[i], indOG[i], indState[i], n1stWeightIG[i], n1stWeightFG[i], n1stWeightIN[i], n1stWeightOG[i], n1stWeightPeep[i], n1stBiasIG[i], n1stBiasFG[i], n1stBiasIN[i], n1stBiasOG[i], n1stdSdWBias[i], n1stdSdWIG[i], n1stdSdWFG[i], n1stdSdWIN[i], new SoftSign)
                             :
                             new LSTMLayer(recurSize[i], recurr_1st_new_link[i], recurr_n_new_links[i], recurr_1st_old_link[i], recurr_n_old_links[i], recurr_tot_links[i], recurr_pos[i], indIG[i], indFG[i], indOG[i], indState[i], n1stWeightIG[i], n1stWeightFG[i], n1stWeightIN[i], n1stWeightOG[i], n1stWeightPeep[i], n1stBiasIG[i], n1stBiasFG[i], n1stBiasIN[i], n1stBiasOG[i], n1stdSdWBias[i], n1stdSdWIG[i], n1stdSdWFG[i], n1stdSdWIN[i], new Linear)
                             ); // damnit, Susan!
            
        }
        if (layerSize[i]>0)
        {
            for (int w=n1stWeightHL[i]; w<n1stWeightHL[i]+normal_n_links[i]*layerSize[i]; w++)
            weights[w] = dis(gen)*sqrt(6)/(normal_n_links[i] + layerSize[i] + recurSize[i]);
            
            for (int w=n1stBiasHL[i]; w<n1stWeightHL[i]+normal_n_links[i]*layerSize[i]; w++)
            biases[w] = dis(gen)*sqrt(6)/(normal_n_links[i] + layerSize[i] + recurSize[i]);
            
            layers.push_back( (i+1!=nMixedLayers) ?
                             new NormalLayer(layerSize[i], normal_1st_link[i], normal_n_links[i], normal_pos[i], n1stWeightHL[i], n1stBiasHL[i], new SoftSign)
                             :
                             new NormalLayer(layerSize[i], normal_1st_link[i], normal_n_links[i], normal_pos[i], n1stWeightHL[i], n1stBiasHL[i], new Linear)
                             );
        }
    }
    nLayers = layers.size();
}

void FishNet::save(string fname)
{
    debug1("Saving into %s\n", fname.c_str());
    
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());
    
    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
    
    out.precision(20);
    
    out << nWeights << " "  << nBiases << " " << nLayers  << " " << nNeurons << endl;
    
    for (int i=0; i<nWeights; i++)
        out << *(weights + i) << " ";
    
    for (int i=0; i<nBiases; i++)
        out << *(biases + i) << " ";

    out.flush();
    out.close();
    
    // Prepare copying command
    string command = "cp ";
    string nameOriginal = fname;
    command = command + nameBackup + " " + nameOriginal;
    
    // Submit the command to the system
    system(command.c_str());
}

bool FishNet::restart(string fname)
{
    string nameBackup = fname + "_tmp";
    
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good())
    {
        error("WTF couldnt open file %s (ok keep going mofo)!\n", fname.c_str());
        return false;
    }
    
    int readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
    
    if (readTotWeights != nWeights || readTotBiases != nBiases || readNLayers != nLayers || readNNeurons != nNeurons)
    die("Network parameters differ!");
    
    for (int i=0; i<nWeights; i++)
        in >> *(weights + i);
    
    for (int i=0; i<nBiases; i++)
        in >> *(biases + i);

    in.close();
    return true;
}

void FishNet::predict(const vector<Real>& input, vector<Real>& output, int nAgent)
{ //updates memory of agent
    for (int n=0; n<nInputs; n++)
        *(outvals +n) = input[n]; /* inputs are the ovals of layer -1 */
    for (int n=0; n<nNeurons; n++)
        *(in_vals +n) = 0.; /* everything here is a += */
    for (int n=0; n<nGates; n++)
        *(igates +n) = 0.; /* everything here is a += */
    
    swap(Agents[nAgent].oldvals,oldvals);
    swap(Agents[nAgent].ostates,ostates);

#ifdef DBG_INPUT
    _info("oldvals before prediction of agent %d are: ", nAgent);
    for (int i=0; i<nNeurons; i++)
    _info("%f ", *(Agents[nAgent].oldvals[i]));
    _info("ostates before prediction of agent %d are: ", nAgent);
    for (int i=0; i<nStates; i++)
    _info("%f ", *(Agents[nAgent].ostates[i]));
#endif

    for (int i=0; i<layers.size(); i++)
        layers[i]->propagate(in_vals, outvals, oldvals, weights, biases, igates, ogates, ostates, nstates);

    for (int i=0; i<nOutputs; i++)
        output[i] = *(outvals +nNeurons -nOutputs +i);
    
    swap(Agents[nAgent].oldvals, outvals);
    swap(Agents[nAgent].ostates, nstates);
    
#ifdef DBG_INPUT
    _info("oldvals after prediction of agent %d are: ", nAgent);
    for (int i=0; i<nNeurons; i++)
    _info("%f ", *(Agents[nAgent].oldvals[i]));
    _info("ostates after prediction of agent %d are: ", nAgent);
    for (int i=0; i<nStates; i++)
    _info("%f ", *(Agents[nAgent].ostates[i]));
#endif
}

void FishNet::test(const vector<Real>& input, vector<Real>& output, int nAgent)
{ // does not affect memory of agents
    for (int n=0; n<nInputs; n++)
        *(outvals +n) = input[n]; /* inputs are the ovals of layer -1 */
    for (int n=0; n<nNeurons; n++)
        *(in_vals +n) = 0.; /* everything here is a += */
    for (int n=0; n<nGates; n++)
        *(igates +n) = 0.; /* everything here is a += */
    
    swap(Agents[nAgent].oldvals,oldvals);
    swap(Agents[nAgent].ostates,ostates);
    
#ifdef DBG_INPUT
    _info("oldvals before prediction of agent %d are: ", nAgent);
    for (int i=0; i<nNeurons; i++)
    _info("%f ", *(Agents[nAgent].oldvals[i]));
    _info("ostates before prediction of agent %d are: ", nAgent);
    for (int i=0; i<nStates; i++)
    _info("%f ", *(Agents[nAgent].ostates[i]));
#endif
    
    for (int i=0; i<layers.size(); i++)
    layers[i]->propagate(in_vals, outvals, oldvals, weights, biases, igates, ogates, ostates, nstates);
    
    for (int i=0; i<nOutputs; i++)
    output[i] = *(outvals +nNeurons -nOutputs +i);
    
    swap(Agents[nAgent].oldvals,oldvals);
    swap(Agents[nAgent].ostates,ostates);
    
#ifdef DBG_INPUT
    _info("oldvals after prediction of agent %d are: ", nAgent);
    for (int i=0; i<nNeurons; i++)
    _info("%f ", *(Agents[nAgent].oldvals[i]));
    _info("ostates after prediction of agent %d are: ", nAgent);
    for (int i=0; i<nStates; i++)
    _info("%f ", *(Agents[nAgent].ostates[i]));
#endif
}

void FishNet::improve(const vector<Real>& input, const vector<Real>& error, int nAgent)
{
    for (int n=0; n<nNeurons; n++)
        *(errvals +n) = 0.;
    
    /*if(kappa>0) // eligibility trace: one error at the time, only for RL
    {
        Real signal = 0.0;
        int Isignal;
        for (int i=0; i<nOutputs; i++)
        if(fabs(error[i])>fabs(signal))
        {
            Isignal = i;
            signal = error[i];
        }
        *(errvals +nNeurons -nOutputs +Isignal) = 1.;
    }
    else */
    
    for (int i=0; i<nOutputs; i++)
        *(errvals +nNeurons -nOutputs +i) = error[i];
  
    for (int i=nLayers-1; i>=0; i--)
        layers[i]->backPropagate(in_vals, outvals, oldvals, errvals, weights, igates, ogates, ostates, nstates, dsdw, Dw, Db);

    /*if(kappa>0)
    {
        for (int i=0; i<nWeights; i++)
        {
            *(trace_W) = kappa * *(trace_W) + *(Dw + i);
            *(Dw + i) = signal * trace_W;
        }
        for (int i=0; i<nBiases; i++)
        {
            *(trace_B) = kappa * *(trace_B) + *(Db + i);
            *(Db + i) = signal * trace_W;
        }
    }*/
    /* Adam update. It's good, I like it. */
    Real fac1 = 1./(1.-beta_t_1);
    Real fac2 = 1./(1.-beta_t_2);
    for (int i=0; i<nWeights; i++)
    {
        *(_1stMomW + i) = beta_1 * *(_1stMomW + i) + (1.-beta_1) * *(Dw + i);
        *(_2ndMomW + i) = beta_2 * *(_2ndMomW + i) + (1.-beta_2) * *(Dw + i) * *(Dw + i);
        *(weights + i) += eta * *(_1stMomW + i) * fac1 / (sqrt(*(_2ndMomW + i) * fac2) + epsilon);
    }
    
    for (int i=0; i<nBiases; i++)
    {
        *(_1stMomB + i) = beta_1 * *(_1stMomB + i) + (1.-beta_1) * *(Db + i);
        *(_2ndMomB + i) = beta_2 * *(_2ndMomB + i) + (1.-beta_2) * *(Db + i) * *(Db + i);
        *(biases + i) += eta * *(_1stMomB + i) * fac1 / (sqrt(*(_2ndMomB + i) * fac2) + epsilon);
    }
    
    beta_t_1 *= beta_2;
    beta_t_2 *= beta_1;
}

void NormalLayer::propagate(Real* in_vals, Real* outvals, Real* oldvals, Real* weights, Real* biases, Real* igates, Real* ogates, Real* ostates, Real* nstates)
{
    for (int n=0; n<nNeurons; n++)
    {
        for (int i=0; i<nLinks; i++)
            *(in_vals +n1stNeuron +n) += *(outvals +n1stLink +i) * *(weights +n1stWeight +n*nLinks +i);
        *(in_vals +n1stNeuron +n) += *(biases +n1stBias +n);
        *(outvals +n1stNeuron +n) = func->eval( (in_vals +n1stNeuron +n) );
    }
}

void NormalLayer::backPropagate(Real* in_vals, Real* outvals, Real* oldvals, Real* errvals, Real* weights, Real* igates, Real* ogates, Real* ostates, Real* nstates, Real* dsdw, Real* Dw, Real* Db)
{   //downstream gave delta, assumed we forward propagated just before so we have ivals and "upstream" errvals are 0
    for (int n=0; n<nNeurons; n++)
    {
        *(errvals +n1stNeuron +n) = *(errvals +n1stNeuron +n) * func->evalDiff( (in_vals +n1stNeuron +n) );
        for (int i=0; i<nLinks; i++)
        {
            *(errvals +n1stLink +i) += *(errvals +n1stNeuron +n) * *(weights +n1stWeight +n*nLinks +i);
            *(Dw +n1stWeight +n*nLinks +i) = *(errvals +n1stNeuron +n) * *(outvals +n1stLink +i);
        }
        *(Db +n1stBias +n) = *(errvals +n1stNeuron +n);
    }
}

void LSTMLayer::propagate(Real* in_vals, Real* outvals, Real* oldvals, Real* weights, Real* biases, Real* igates, Real* ogates, Real* ostates, Real* nstates)
{
    for (int n=0; n<nNeurons; n++)
    { /* compute IG, IN and FG, required for peeps */
        for (int i=0; i<nOldL; i++)
        { //this loops need to be split if cells per block =/= 1
            *(igates +nIG +n) += *(oldvals +n1stOld +i) * *(weights +n1stIG +n*nLinks +i);
            *(igates +nFG +n) += *(oldvals +n1stOld +i) * *(weights +n1stFG +n*nLinks +i);
            *(in_vals +n1stNeuron +n) += *(oldvals +n1stOld +i) * *(weights +n1stWeight +n*nLinks +i);
        }
        for (int i=0; i<nLinks; i++)
        {
            *(igates +nIG +n) += *(outvals +n1stLink +i) * *(weights +n1stIG +n*nLinks +i +nOldL);
            *(igates +nFG +n) += *(outvals +n1stLink +i) * *(weights +n1stFG +n*nLinks +i +nOldL);
            *(in_vals +n1stNeuron +n) += *(outvals +n1stLink +i) * *(weights +n1stWeight +n*nLinks +i +nOldL);
        }
        *(in_vals +n1stNeuron +n) += *(biases +n1stBias +n);
        *(igates +nIG +n) += *(ostates +n1stState +n) * *(weights +n1stPeep +3*n) + *(biases +n1stIGB +n);
        *(igates +nFG +n) += *(ostates +n1stState +n) * *(weights +n1stPeep +3*n +1) + *(biases +n1stFGB +n);
        
        *(ogates +nIG +n) = sigm->eval( (igates +nIG +n) );
        *(ogates +nFG +n) = sigm->eval( (igates +nFG +n) );
        
        *(nstates +n1stState +n) = *(ostates +n1stState +n) * *(ogates +nFG +n) + func->eval( (in_vals +n1stNeuron +n) ) * *(ogates +nIG +n);

        for (int i=0; i<nOldL; i++)
            *(igates +nOG +n) += *(oldvals +n1stOld +i) * *(weights +n1stOG +n*nLinks +i);

        for (int i=0; i<nLinks; i++)
            *(igates +nOG +n) += *(outvals +n1stLink +i) * *(weights +n1stOG +n*nLinks +i  +nOldL);
        
        *(igates +nOG +n) += *(nstates +n1stState +n) * *(weights +n1stPeep +3*n +2) + *(biases +n1stOGB +n);
        *(ogates +nOG +n) = sigm->eval( (igates +nOG +n) );
        *(outvals +n1stNeuron +n) = *(nstates +n1stState +n) * *(ogates +nOG +n);
    }
}

void LSTMLayer::backPropagate(Real* in_vals, Real* outvals, Real* oldvals, Real* errvals, Real* weights, Real* igates, Real* ogates, Real* ostates, Real* nstates, Real* dsdw, Real* Dw, Real* Db)
{
    for (int n=0; n<nNeurons; n++)
    {
        Real tmp1 = sigm->evalDiff( (igates +nOG +n) ) * *(errvals +n1stNeuron +n) * *(nstates +n1stState +n);
        Real tmp2 = func->evalDiff( (in_vals +n1stNeuron +n) ) * *(ogates +nIG +n);
        Real tmp3 = sigm->evalDiff( (igates +nIG +n) ) * func->eval( (in_vals +n1stNeuron +n));
        Real tmp4 = sigm->evalDiff( (igates +nFG +n) ) * *(ostates +n1stState +n);
        *(errvals +n1stNeuron +n) = *(errvals +n1stNeuron +n) * *(ogates +nOG +n);
        // multiply by * func->evalDiff( (nstates +n1stState +n) ) in case its not linear

        *(dsdw +n1dsdB +n*5   ) = *(dsdw +n1dsdB +n*5   ) * *(ogates +nFG +n) + tmp3;
        *(Db +n1stIGB +n) = *(dsdw +n1dsdB +n*5   ) * *(errvals +n1stNeuron +n);
        
        *(dsdw +n1dsdB +n*5 +1) = *(dsdw +n1dsdB +n*5 +1) * *(ogates +nFG +n) + tmp4;
        *(Db +n1stFGB +n) = *(dsdw +n1dsdB +n*5 +1) * *(errvals +n1stNeuron +n);
        
        *(dsdw +n1dsdB +n*5 +2) = *(dsdw +n1dsdB +n*5 +2) * *(ogates +nFG +n) + tmp2;
        *(Db +n1stBias +n) = *(dsdw +n1dsdB +n*5 +2) * *(errvals +n1stNeuron +n);
        
        *(Db +n1stOGB +n) = tmp1;
        
        // now the peepholes: note in this case mem is not [IG IG IG ...] [FG FG...] ... but [IG FG OG] [...]
        *(dsdw +n1dsdB +n*5 +3) = *(dsdw +n1dsdB +n*5 +3) * *(ogates +nFG +n) + tmp3 * *(ostates +n1stState +n);
        *(Dw +n1stPeep +3*n   ) = *(dsdw +n1dsdB +n*5 +3) * *(errvals +n1stNeuron +n);
        
        *(dsdw +n1dsdB +n*5 +4) = *(dsdw +n1dsdB +n*5 +4) * *(ogates +nFG +n) + tmp4 * *(ostates +n1stState +n);
        *(Dw +n1stPeep +3*n +1) = *(dsdw +n1dsdB +n*5 +4) * *(errvals +n1stNeuron +n);
        
        *(Dw +n1stPeep +3*n +2) = tmp1 * *(nstates +n1stState +n);

        for (int i=0; i<nOldL; i++)
        {
            *(dsdw +n1dsIG +n*nLinks +i) = *(dsdw +n1dsIG +n*nLinks +i) * *(ogates +nFG +n) + tmp3 * *(oldvals +n1stOld +i);
            *(Dw +n1stIG +n*nLinks +i) = *(dsdw +n1dsIG +n*nLinks +i) * *(errvals +n1stNeuron +n);
            
            *(dsdw +n1dsFG +n*nLinks +i) = *(dsdw +n1dsFG +n*nLinks +i) * *(ogates +nFG +n) + tmp4 * *(oldvals +n1stOld +i);
            *(Dw +n1stFG +n*nLinks +i) = *(dsdw +n1dsFG +n*nLinks +i) * *(errvals +n1stNeuron +n);
            
            *(dsdw +n1dsIN +n*nLinks +i) = *(dsdw +n1dsIN +n*nLinks +i) * *(ogates +nFG +n) + tmp2 * *(oldvals +n1stOld +i);
            *(Dw +n1stWeight +n*nLinks +i) = *(dsdw +n1dsIN +n*nLinks +i) * *(errvals +n1stNeuron +n);
            
            *(Dw +n1stOG +n*nLinks +i) = tmp1 * *(oldvals +n1stOld +i);
            // skip it because we dont propagate old errors to older inputs this way ... for now
            /* *(errvals +n1stOld +i) += tmp1 * *(weights +n1stOG +n*nLinks +i) + *(errvals +n1stNeuron +n) * (
                                         tmp2 * *(weights +n1stWeight +n*nLinks +i) +
                                         tmp3 * *(weights +n1stIG +n*nLinks +i) +
                                         tmp4 * *(weights +n1stFG +n*nLinks +i) ));*/
        }
        for (int i=0; i<nLinks; i++) 
        {
            *(dsdw +n1dsIG +n*nLinks +i +nOldL) = *(dsdw +n1dsIG +n*nLinks +i +nOldL) * *(ogates +nFG +n) + tmp3 * *(outvals +n1stLink +i);
            *(Dw +n1stIG +n*nLinks +i +nOldL) = *(dsdw +n1dsIG +n*nLinks +i +nOldL) * *(errvals +n1stNeuron +n);
            
            *(dsdw +n1dsFG +n*nLinks +i +nOldL) = *(dsdw +n1dsFG +n*nLinks +i +nOldL) * *(ogates +nFG +n) + tmp4 * *(outvals +n1stLink +i);
            *(Dw +n1stFG +n*nLinks +i +nOldL) = *(dsdw +n1dsFG +n*nLinks +i +nOldL) * *(errvals +n1stNeuron +n);
            
            *(dsdw +n1dsIN +n*nLinks +i +nOldL) = *(dsdw +n1dsIN +n*nLinks +i +nOldL) * *(ogates +nFG +n) + tmp2 * *(outvals +n1stLink +i);
            *(Dw +n1stWeight +n*nLinks +i +nOldL) = *(dsdw +n1dsIN +n*nLinks +i +nOldL) * *(errvals +n1stNeuron +n);
            
            *(Dw +n1stOG +n*nLinks +i +nOldL) = tmp1 * *(outvals +n1stOld +i);
            //controversial backprop:
            *(errvals +n1stLink +i) += tmp1 * *(weights +n1stOG +n*nLinks +i +nOldL) + *(errvals +n1stNeuron +n) * (
                                         tmp2 * *(weights +n1stWeight +n*nLinks +i +nOldL) +
                                         tmp3 * *(weights +n1stIG +n*nLinks +i +nOldL) +
                                         tmp4 * *(weights +n1stFG +n*nLinks +i +nOldL) );
        }
    }
}
