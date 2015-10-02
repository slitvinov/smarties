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
#include <cblas.h>

#include "Network.h"
#include "../ErrorHandling.h"
//#define DBG_EXEC
//#define DBG_BACK //panico
//#define DBG_ADJS
using namespace ErrorHandling;

Network::Network(vector<int>& layerSize, double eta, double alpha, double _lambda, int batchSize) :
nInputs(layerSize.front()), nOutputs(layerSize.back()), nLayers(layerSize.size()), eta(eta), alpha(alpha), batchSize(batchSize), rng(0)
{
    lambda = _lambda;
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
    
    totWeights = 0;
    nInBatch   = 0;
    for (int i=0; i<nLayers-1; i++)
    {
        totWeights += layers[i]->nNeurons * (layers[i+1]->nNeurons - 1);
    }
    
    if (batchSize == -1) this->batchSize = batchSize = totWeights;
    
    J.set_size(nOutputs*batchSize, totWeights);
    I.eye(totWeights, totWeights);
    
    e.set_size(batchSize*nOutputs);
    dw.set_size(totWeights);
    Je.set_size(totWeights);
}

void Network::predict(const vector<double>& inputs, vector<double>& outputs, int nAgent)
{
    for (int i=0; i<nInputs; i++)
    {
        *(this->inputs[i]) = inputs[i];
        //_info("%f\n", inputs[i]);
    }
    
    for (int i=0; i<nLayers; i++)
    layers[i]->propagate();
    
    for (int i=0; i<nOutputs; i++)
    outputs[i] = *(this->outputs[i]);
}

void Network::improve(const vector<double>& inputs, const vector<double>& errors, int nAgent)
{
    for (int i=0; i<nInputs; i++)
    {
        *(this->inputs[i]) = inputs[i];
        //_info("%f\n", inputs[i]);
    }
    
    for (int i=0; i<nOutputs; i++)
    {
        *(this->errors[i]) = errors[i];
        //_info("%f\n", errors[i]);
    }
    
    for (int i=nLayers-1; i>=0; i--)
    layers[i]->backPropagate();
    
    for (int i=0; i<nLayers; i++)
    layers[i]->adjust(eta, alpha,lambda);
}

void Network::setBatchsize(int size)
{
    batchSize = size;
    totWeights = 0;
    nInBatch   = 0;
    for (int i=0; i<nLayers-1; i++)
    {
        totWeights += layers[i]->nNeurons * (layers[i+1]->nNeurons - 1);
    }
    
    if (batchSize == -1) this->batchSize = batchSize = totWeights;
    
    J.set_size(nOutputs*batchSize, totWeights);
    I.eye(totWeights, totWeights);
    
    e.set_size(batchSize*nOutputs);
    dw.set_size(totWeights);
    Je.set_size(totWeights);
}

NetworkLM::NetworkLM(vector<int>& layerSize, double muFactor, int batchSize) :
Network(layerSize, 0, 0, 0, batchSize), muFactor(muFactor)
{
    mu = 0.01;
    muMax = 1e10;
    muMin = 1e-3;
}

void NetworkLM::improve(const vector<double>& inputs, const vector<double>& errors, int nAgent)
{
    openblas_set_num_threads(12);
    vector<double> tmpVec(nOutputs);
    predict(inputs, tmpVec);
    batch.push_back(inputs);
    batchOut.push_back(tmpVec);
    for (int i=0; i<nOutputs; i++)
    tmpVec[i] -= errors[i];
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
        debug("nInBatch %d\n", batchSize);
        
        nInBatch = 0;
        Q = 0;
        for (int i=0; i<nOutputs*batchSize; i++)
        Q += e(i) * e(i);
        
        double Q0 = Q;
        Q = Q0+1;
        
        JtJ = J.t() * J;
        Je  = J.t() * e;
        
        while (Q > Q0)
        {
            tmp = JtJ + mu*I;
            dw = solve(tmp, Je);
            
            bool _nan = false;
            for (int w=0; w<totWeights; w++)
            if (std::isnan((dw(w))) || std::isinf((dw(w))))
            _nan = true;
            if (_nan)
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
                debug6("l%d w%d%d = %f\n", l, lnk, n, layers[l]->neurons[n]->inLinks[lnk]->w);
                layers[l]->neurons[n]->inLinks[lnk]->w += dw(w++);
            }
            
            
            Q = 0;
            for (int i=0; i<batchSize; i++)
            {
                predict(batch[i], tmpVec);
                for (int j=0; j<nOutputs; j++)
                {
                    double diff = tmpVec[j] - batchExact[i][j];
                    Q += diff * diff;
                }
            }
            
            if (Q > Q0)
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

void NetworkLM::save(string fname)
{
    debug1("Saving into %s\n", fname.c_str());
    
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());
    
    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
    
    out.precision(20);
    
    out << totWeights << " " << nInputs << " " << nLayers << endl;
    for(int i=0; i<nLayers; i++)
    {
        out << layers[i]->nNeurons << "  ";
    }
    out << endl;
    
    //*************************************************************************
    for (int l=0; l<nLayers; l++)
    for (int n=0; n<layers[l]->nNeurons; n++)
    for (int lnk=0; lnk<layers[l]->neurons[n]->inLinks.size(); lnk++)
				out << layers[l]->neurons[n]->inLinks[lnk]->w << " ";
    //*************************************************************************
    
    out.flush();
    out.close();
    
    // Prepare copying command
    string command = "cp ";
    string nameOriginal = fname;
    command = command + nameBackup + " " + nameOriginal;
    
    // Submit the command to the system
    system(command.c_str());
}

bool NetworkLM::restart(string fname)
{
    string nameBackup = fname + "_tmp";
    
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good())
    {
        error("WTF couldnt open file %s (ok keep going mofo)!\n", fname.c_str());
        return false;
    }
    
    int readTotWeights, readNInputs, readNLayers;
    in >> readTotWeights >> readNInputs >> readNLayers;
    
    if (readTotWeights != totWeights || readNInputs != nInputs || readNLayers != nLayers)
    die("Network parameters differ!");
    
    for(int i=0; i<nLayers; i++)
    {
        int dummy;
        in >> dummy;
        if (dummy != layers[i]->nNeurons)
        die("Network layer parameters differ!");
    }
    
    //*************************************************************************
    for (int l=0; l<nLayers; l++)
    for (int n=0; n<layers[l]->nNeurons; n++)
    for (int lnk=0; lnk<layers[l]->neurons[n]->inLinks.size(); lnk++)
				in >> layers[l]->neurons[n]->inLinks[lnk]->w;
    //*************************************************************************
    
    in.close();
    return true;
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

void Layer::adjust(double eta, double alpha, double lambda)
{
    for (int i=0; i<nNeurons; i++)
    neurons[i]->adjust(eta, alpha, lambda);
}

Neuron::Neuron(ActivationFunction* func) :
hasInputs(false), hasOutputs(false), func(func) { };

void Neuron::exec()
{
    if (hasInputs)
    {
        ival = 0;
        for (int i=0; i<inLinks.size(); i++)
        {
            ival += inLinks[i]->val * inLinks[i]->w;
#ifdef DBG_EXEC
            _info("in val %d = %f, w = %f\n", i, inLinks[i]->val, inLinks[i]->w);
#endif
        }
    }
    
    oval = func->eval(ival);
#ifdef DBG_EXEC
    _info("out val %f\n", oval);
#endif
    if (hasOutputs)
    {
        for (int i=0; i<outLinks.size(); i++)
            outLinks[i]->val = oval;
    }
}

void MemoryCell::exec()
{ //Input acts just normal neuron -> get oval
    Input->exec();
    Sc_old = Sc_new;
    Sc_new = Sc_old * FG->oval + IG->oval * Input->oval;
    ScN->oval = Sc_new;
    for (int j=0; j<ScN->outLinks.size(); j++)
        ScN->outLinks[j]->val = func->eval(Sc_new);
#ifdef DBG_EXEC
    _info("Sc_old = %f Sc_new = %f \n",func->eval(Sc_old),func->eval(Sc_new));
#endif
    //oval = func->eval(Sc_new) * OG->oval;
    
    //for (int i=0; i<outLinks.size(); i++)
        //outLinks[i]->val = oval;
}

void MemoryBlock::exec()
{
    for (int i=0; i<nMemoryCells; i++)
    {
        //mCells[i]->ScO->oval = mCells[i]->Sc_new;
        mCells[i]->ScO->oval = mCells[i]->func->eval(mCells[i]->Sc_new);
        for (int j=0; j<mCells[i]->ScO->outLinks.size(); j++)
            mCells[i]->ScO->outLinks[j]->val = mCells[i]->ScO->oval;
    }
        
    //IG, FG, OG behave like neurons: get input from outside
    IG->exec();
    FG->exec();
#ifdef DBG_EXEC
    _info("IG oval = %f\n",IG->oval);
    _info("FG oval = %f\n",FG->oval);
#endif
    
    //oval of gates used by the cell "neuron"
    for (int i=0; i<nMemoryCells; i++)
        mCells[i]->exec();
    OG->exec();
    //_info("OG oval = %f\n",OG->oval);
    for (int i=0; i<nMemoryCells; i++)
    {
        mCells[i]->oval = mCells[i]->func->eval(mCells[i]->Sc_new) * OG->oval;
#ifdef DBG_EXEC
        _info("Sj oval = %f\n",mCells[i]->oval);
#endif
        for (int j=0; j<mCells[i]->outLinks.size(); j++)
            mCells[i]->outLinks[j]->val = mCells[i]->oval;
    }
}

void Neuron::backExec()
{
    if (hasOutputs)
    {
        err = 0;
        for (int i=0; i<outLinks.size(); i++)
        {
            err += outLinks[i]->err * outLinks[i]->w;
#ifdef DBG_BACK
            _info("err %d = %f, w = %f\n", i, outLinks[i]->err, outLinks[i]->w);
#endif
        }
    }
    
    err = func->evalDiff(ival) * err;
#ifdef DBG_BACK
    _info("out val %f\n", err);
#endif
    if (hasInputs)
    {
        for (int i=0; i<inLinks.size(); i++)
        inLinks[i]->err = err;
    }
}

void MemoryCell::backExec()
{
    sumwd = 0.0;
    //Sum_{over out links k of cell} w_{out link k - cell v of block j} delta_{k}
    for (int i=0; i<outLinks.size(); i++)
    {
        sumwd += outLinks[i]->err * outLinks[i]->w;
#ifdef DBG_BACK
        _info("err %d = %f, w = %f\n", i, outLinks[i]->err, outLinks[i]->w);
#endif
    }
    //fac = Sum_{over cells v in block} H(Sc_v) * fac_v
    OGerrfac = sumwd * func->eval(Sc_new);
    
    //error state of cell
    err = sumwd * OG->oval * func->evalDiff(Sc_new);
    
    //error of cell input
    
    for (int i=0; i<Input->inLinks.size(); i++)
    {
        Neuron* prev = Input->inLinks[i]->neuronFrom;
//        if (dsdw_IN.size() != Input->inLinks.size() || dsdw_INo.size() != Input->inLinks.size())
//            die("Mismatch inLinks Input and dsdw\n");
//        printf("dsdwIN %d old %f new %f\n", i, dsdw_INo[i], dsdw_IN[i]);
        dsdw_INo[i] = dsdw_IN[i];
        dsdw_IN[i]  = dsdw_INo[i] * FG->oval + Input->func->evalDiff(Input->ival) * IG->oval * prev->oval;
        
        Input->inLinks[i]->err = 0.0;
        
        if (fabs(prev->oval)>1e-9)
            Input->inLinks[i]->err = err * dsdw_IN[i] / prev->oval;
//#ifdef DBG_BACK
//        _info("Input out err %f\n", Input->inLinks[i]->err);
//#endif
    }
    
    for (int i=0; i<IG->inLinks.size(); i++)
    {
        Neuron* prev = IG->inLinks[i]->neuronFrom;
//        if (dsdw_IG.size() != IG->inLinks.size() || dsdw_IGo.size() != IG->inLinks.size())
//            die("Mismatch inLinks IG and dsdw\n");
//        printf("dsdwIG %d old %f new %f\n", i, dsdw_IGo[i], dsdw_IG[i]);
        dsdw_IGo[i] = dsdw_IG[i];
        dsdw_IG[i]  = dsdw_IGo[i] * FG->oval + Input->oval * IG->func->evalDiff(IG->ival) * prev->oval;
    }
    
    for (int i=0; i<FG->inLinks.size(); i++)
    {
        Neuron* prev = FG->inLinks[i]->neuronFrom;
//        if (dsdw_FG.size() != FG->inLinks.size() || dsdw_FGo.size() != FG->inLinks.size())
//            die("Mismatch inLinks FG and dsdw\n");
//        printf("dsdwFG %d old %f new %f\n", i, dsdw_FGo[i], dsdw_FG[i]);
        dsdw_FGo[i] = dsdw_FG[i];
        dsdw_FG[i]  = dsdw_FGo[i] * FG->oval + Sc_old * FG->func->evalDiff(FG->ival) * prev->oval;
    }
    
}

void MemoryBlock::backExec()
{
    // error of output gate and error state of cell
    
    double OGerrfac = 0.0;
    for (int i=0; i<nMemoryCells; i++)
    {
        mCells[i]->backExec();
        OGerrfac+=mCells[i]->OGerrfac;
    }
    //delta_OG = ...questo V
    OG->err = OG->func->evalDiff(OG->ival) * OGerrfac;
#ifdef DBG_BACK
    _info("OG out err %f\n", OG->err);
#endif
    
    for (int i=0; i<OG->inLinks.size(); i++)
        OG->inLinks[i]->err = OG->err;
    
    for (int i=0; i<IG->inLinks.size(); i++)
    {
        Neuron* prev = IG->inLinks[i]->neuronFrom;
        
        IG->inLinks[i]->err = 0.0;
        
        if (fabs(prev->oval)>1e-9)
        {
            for (int j=0; j<nMemoryCells; j++)
                IG->inLinks[i]->err += mCells[j]->err * mCells[j]->dsdw_IG[i];
            
            IG->inLinks[i]->err /= prev->oval;
#ifdef DBG_BACK
            _info("IG out err %f\n", IG->inLinks[i]->err);
#endif
        }
    }
    
    for (int i=0; i<FG->inLinks.size(); i++)
    {
        Neuron* prev = FG->inLinks[i]->neuronFrom;
        
        FG->inLinks[i]->err = 0.0;
        
        if (fabs(prev->oval)>1e-9)
        {
            for (int j=0; j<nMemoryCells; j++)
                FG->inLinks[i]->err += mCells[j]->err * mCells[j]->dsdw_FG[i];
            
            FG->inLinks[i]->err /= prev->oval;
#ifdef DBG_BACK
            _info("FG out err %f\n", FG->inLinks[i]->err);
#endif
        }
    }
}

void MemoryCell::_backExec()
{
    sumwd = 0.0;
    //Sum_{over out links k of cell} w_{out link k - cell v of block j} delta_{k}
    for (int i=0; i<outLinks.size(); i++)
    {
        sumwd += outLinks[i]->err * outLinks[i]->w;
#ifdef DBG_BACK
        _info("MCell err %d = %f, w = %f\n", i, outLinks[i]->err, outLinks[i]->w);
#endif
    }
    //fac = Sum_{over cells v in block} H(Sc_v) * fac_v
    OGerrfac = sumwd * func->eval(Sc_new);
    
    //error state of cell
    err = sumwd * OG->oval * func->evalDiff(Sc_new);
    
    //error of cell input
    Input->err = err * Input->func->evalDiff(Input->ival) * IG->oval;
#ifdef DBG_BACK
    _info("Input err = %f\n", Input->err);
#endif
    for (int i=0; i<Input->inLinks.size(); i++)
        Input->inLinks[i]->err = Input->err;
}

void MemoryBlock::_backExec()
{
    // error of output gate and error state of cell
    
    double OGerrfac = 0.0;
    for (int i=0; i<nMemoryCells; i++)
    {
        mCells[i]->_backExec();
        OGerrfac+=mCells[i]->OGerrfac;
    }
    //delta_OG = ...questo V
    OG->err = OG->func->evalDiff(OG->ival) * OGerrfac;
#ifdef DBG_BACK
    _info("OG out err %f\n", OG->err);
#endif
    
    for (int i=0; i<OG->inLinks.size(); i++)
        OG->inLinks[i]->err = OG->err;
    
    IG->err = 0.0;
    for (int j=0; j<nMemoryCells; j++)
        IG->err += mCells[j]->err * mCells[j]->Input->oval * IG->func->evalDiff(IG->ival);
    
#ifdef DBG_BACK
    _info("IG out err %f\n", IG->err);
#endif
    
    for (int i=0; i<IG->inLinks.size(); i++)
        IG->inLinks[i]->err = IG->err;
    
    FG->err = 0.0;
    for (int j=0; j<nMemoryCells; j++)
        FG->err += mCells[j]->err * mCells[j]->Sc_old * FG->func->evalDiff(FG->ival);
    
#ifdef DBG_BACK
    _info("FG out err %f\n", FG->err);
#endif
    for (int i=0; i<FG->inLinks.size(); i++)
        FG->inLinks[i]->err = FG->err;
}

void Neuron::adjust(double eta, double alpha, double lambda)
{
    if (hasInputs)
    {
        for (int i=0; i<inLinks.size(); i++)
        {
            Neuron* prev = inLinks[i]->neuronFrom;
            inLinks[i]->prevDw = inLinks[i]->Dw;
            inLinks[i]->Dw = - eta * err * prev->oval; // !!!!!!!!! Error = output - target !!!!!!!!!!
            inLinks[i]->w += inLinks[i]->Dw + alpha * inLinks[i]->prevDw - lambda*eta*inLinks[i]->w;
        }
    }
}

void Neuron::adjust(double error, double eta, double alpha, double lambda, double kappa)
{
    if (hasInputs)
    {
#ifdef DBG_ADJS
        printf("error %f, eta %f, alpha %f, lambda %f, kappa %f\n",error, eta, alpha, lambda, kappa);
#endif
        bool bRegularization(true);
        bool bMomentumLearn(true);
        bool bEligibTrace(true);
        bool bAdaptiveLearnR(false); //not working
        bool bRprop(true);
        bool bMix(true);
        for (int i=0; i<inLinks.size(); i++)
        {
            Neuron* prev = inLinks[i]->neuronFrom;
            inLinks[i]->prevDw = inLinks[i]->Dw;
            
            //here we assume we have just one error signal and the backprop has been done with error signal = 1 (hardcoded in LSTM)
            if (bEligibTrace && kappa>0.)
            {
                inLinks[i]->epsilon = (kappa * inLinks[i]->epsilon) + inLinks[i]->err * prev->oval; //eligibility
                inLinks[i]->Dw = eta * error * inLinks[i]->epsilon; // !!!!!!!!! Error = target - output !!!!!!!!!!
            }
            else
                inLinks[i]->Dw = eta * error * inLinks[i]->err * prev->oval;
            
            double dEdw = -inLinks[i]->Dw / eta;
            
            if (bAdaptiveLearnR)
            {
                double delta = 0.1;
                if (bMix)
                    inLinks[i]->Dw = 0.5*inLinks[i]->Dw - 0.5 * dEdw * inLinks[i]->etar;
                
                inLinks[i]->rr = (1-delta)*inLinks[i]->rr + delta*dEdw;
                inLinks[i]->maxrr = max(fabs(inLinks[i]->rr),inLinks[i]->maxrr);
                double alpha = 0.002;
                double beta  = 100./inLinks[i]->maxrr;
            
                inLinks[i]->etar = inLinks[i]->etar + alpha*inLinks[i]->etar* (beta*fabs(inLinks[i]->rr) - inLinks[i]->etar);
                //printf("%f\n",inLinks[i]->etar);
            }
            
            if (bRprop)
            {
                double Delta_w;
                if (dEdw * inLinks[i]->o_dEdw > 0.)
                {
                    inLinks[i]->Delta = min(1.2*inLinks[i]->Delta, eta*fabs(inLinks[i]->w));
                    inLinks[i]->o_dEdw = dEdw;
                    if (dEdw>0.)
                        Delta_w = -inLinks[i]->Delta;
                    if (dEdw<0.)
                        Delta_w =  inLinks[i]->Delta;
                }
                if (dEdw * inLinks[i]->o_dEdw < 0.)
                {
                    inLinks[i]->Delta = max(0.5*inLinks[i]->Delta, lambda*eta*fabs(inLinks[i]->w));
                    inLinks[i]->o_dEdw = 0.0;
                    Delta_w = - inLinks[i]->o_Delta_w;
                }
                if (dEdw * inLinks[i]->o_dEdw == 0.)
                {
                    inLinks[i]->o_dEdw = dEdw;
                    if (dEdw>0.)
                        Delta_w = -inLinks[i]->Delta;
                    if (dEdw<0.)
                        Delta_w =  inLinks[i]->Delta;
                }
                inLinks[i]->o_Delta_w = Delta_w;
                if (bMix)
                    inLinks[i]->Dw = 0.5*inLinks[i]->Dw + 0.5*Delta_w;
            }
            
            if (fabs(inLinks[i]->Dw)>fabs(eta*inLinks[i]->w))
            {
                if (inLinks[i]->Dw>0.)
                    inLinks[i]->Dw =  fabs(eta*inLinks[i]->w);
                if (inLinks[i]->Dw<0.)
                    inLinks[i]->Dw = -fabs(eta*inLinks[i]->w);
            }
            
            inLinks[i]->w += inLinks[i]->Dw;
            
            if (bMomentumLearn && alpha>0.)
            {
                if (inLinks[i]->prevDw*inLinks[i]->Dw > 0.)
                    inLinks[i]->w += alpha * inLinks[i]->prevDw;
                if (inLinks[i]->prevDw*inLinks[i]->Dw < 0.)
                {
                    inLinks[i]->w -= alpha * inLinks[i]->Dw;
                    inLinks[i]->Dw = 0.0;
                }
            }
            
            if (bRegularization && lambda>0.)
                inLinks[i]->w -= lambda*eta*inLinks[i]->w;
            
                //inLinks[i]->w += inLinks[i]->Dw + alpha * inLinks[i]->prevDw - lambda*eta*inLinks[i]->w;
            
        }
    }
}

void MemoryBlock::adjust(double error, double eta, double alpha, double lambda, double kappa)
{
    for (int i=0; i<nMemoryCells; i++)
        mCells[i]->Input->adjust(error, eta, alpha, lambda, kappa);
    
    OG->adjust(error, eta, alpha, lambda, kappa);
    FG->adjust(error, eta, alpha, lambda, kappa);
    IG->adjust(error, eta, alpha, lambda, kappa);
}

NetworkLSTM::NetworkLSTM(vector<int>& layerSize, vector<int>& memorySize, vector<int>& nCellpB, double eta, double alpha, double _lambda, double kappa, int nAgents = 1) :
nInputs(layerSize.front()), nOutputs(layerSize.back()), nLayers(layerSize.size()), eta(eta), alpha(alpha),  rng(0), nAgents(nAgents), nMems(0), nRecurr(0), kappa(kappa), olderr(1)
{
    //lambda(lambda),
    lambda = _lambda;
    //total number of memory cells in network
    for (int i=1; i<nLayers-1; i++) //no memory in input and output layers, recurrency only in hidden layers
    {
        nRecurr += memorySize[i]*nCellpB[i] + layerSize[i];
        nMems += memorySize[i]*nCellpB[i];
    }
    
    inputs.resize(nInputs);
    outputs.resize(nOutputs);
    errors.resize(nOutputs);
    
    //each agent has its memory (cell out signal) and cell state
    //needs the memory before the frwd prop and the memory after the frwd prop
    memory_in.resize(nRecurr);
    memory_out.resize(nRecurr);
    o_state.resize(nMems);
    n_state.resize(nMems);
    
    Agents.clear();
    for (int i = 0; i<nAgents; ++i)
    {
        Memory agent(nMems, nRecurr);
        Agents.push_back(agent);
    }
    
    debug7("Creating first layer with %d neurons\n", nInputs+nRecurr);
    HiddenLayer* first = new HiddenLayer(nInputs+nRecurr, new Linear); //this must be linear: input = output
    layers.push_back(first);
    debug7("- connecting first layer to inputs\n");
    first->connect2inputs(inputs, memory_in);
    
    int indMem = 0;
    int indRec = 0;
    for (int i=1; i<nLayers-1; i++)
    {
        debug7("Creating layer %d with neurons %d %d %d (B C N)\n",i, memorySize[i], nCellpB[i], layerSize[i]);
        HiddenLayer* hl = new HiddenLayer(memorySize[i], nCellpB[i], layerSize[i], new Tanh, &rng);
        layers.push_back(hl);
        
        debug7("- connecting to memory signal\n");
        hl->connect2memstate(memory_out, o_state, n_state, indMem, indRec);
        indRec += memorySize[i]*nCellpB[i] + layerSize[i];
        indMem += memorySize[i]*nCellpB[i];
        
        for (int j=0; j<i+1; j++)
        {
            debug7("- connecting back to layer %d\n", j);
            HiddenLayer* prev = layers[j];
            hl->connect2layers(prev, &rng, i-j-1);
        }
        
        debug7("- connecting to ground signal\n");
        hl->connect2ground(&rng);
        
        hl->normaliseWeights();
        hl->init_dsdw(); //derivatives of cell states wrt weights, required by real-time-recurrent-learning style weight update (blame Forget Gates)
    }
    
    debug7("Creating last layer with %d neurons\n", nOutputs);
    HiddenLayer* last = new HiddenLayer(nOutputs, new Linear);
    layers.push_back(last);
    
    for (int j=0; j<nLayers-1; j++)
    {
        debug7("- connecting back to layer %d\n", j);
        HiddenLayer* prev = layers[j];
        last->connect2layers(prev, &rng, nLayers-j-2);
    }
    
    debug7("- connecting to ground signal\n");
    last->connect2ground(&rng);
    
    last->normaliseWeights();
    debug7("- connecting last layer to outputs and errors\n");
    last->connect2outputs(outputs);
    last->connect2errors(errors);
}
/*
void NetworkLSTM::improveLM(const vector<double>& inputs, const vector<double>& errors, int nAgent)
{
    openblas_set_num_threads(12);
    vector<double> tmpVec(nOutputs);
    predict(inputs, tmpVec, nAgent);
    batch.push_back(inputs);
    batchOut.push_back(tmpVec);
    for (int i=0; i<nOutputs; i++)
        tmpVec[i] -= errors[i];
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
        debug("nInBatch %d\n", batchSize);
        
        nInBatch = 0;
        Q = 0;
        for (int i=0; i<nOutputs*batchSize; i++)
            Q += e(i) * e(i);
        
        double Q0 = Q;
        Q = Q0+1;
        
        JtJ = J.t() * J;
        Je  = J.t() * e;
        
        while (Q > Q0)
        {
            tmp = JtJ + mu*I;
            dw = solve(tmp, Je);
            
            bool _nan = false;
            for (int w=0; w<totWeights; w++)
                if (std::isnan((dw(w))) || std::isinf((dw(w))))
                    _nan = true;
            if (_nan)
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
                        debug6("l%d w%d%d = %f\n", l, lnk, n, layers[l]->neurons[n]->inLinks[lnk]->w);
                        layers[l]->neurons[n]->inLinks[lnk]->w += dw(w++);
                    }
            
            
            Q = 0;
            for (int i=0; i<batchSize; i++)
            {
                predict(batch[i], tmpVec);
                for (int j=0; j<nOutputs; j++)
                {
                    double diff = tmpVec[j] - batchExact[i][j];
                    Q += diff * diff;
                }
            }
            
            if (Q > Q0)
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
*/
void NetworkLSTM::predict(const vector<double>& input, vector<double>& output, int nAgent)
{
    for (int i=0; i<nInputs; i++)
    {
        *(this->inputs[i]) = input[i];
        //_info("%f\n", input[i]);
    }
    //cout << endl;
    
    for (int i=0; i<nMems; i++)
    { //memory and states are downloaded from agents (son of a french cow)
        *(this->o_state[i]) = Agents[nAgent].ostate[i];
        *(this->n_state[i]) = Agents[nAgent].nstate[i];
        //_info("Crapping memory before prediction of agent %d is [%f %f %f]\n", i, Agents[nAgent].memory[i], Agents[nAgent].ostate[i], Agents[nAgent].nstate[i]);
    }
    
    for (int i=0; i<nRecurr; i++)
    { //memory and states are downloaded from agents (son of a french cow)
        *(this->memory_in[i]) = Agents[nAgent].memory[i];
        //_info("Crapping memory before prediction of agent %d is [%f]\n", i, Agents[nAgent].memory[i]);
    }
    
    for (int i=0; i<nLayers; i++)
        layers[i]->propagate();
    
    for (int i=0; i<nOutputs; i++)
        output[i] = *(this->outputs[i]);
    
    for (int i=0; i<nMems; i++)
    { //new state is uploaded to agents
        Agents[nAgent].ostate[i] = *(this->o_state[i]);
        Agents[nAgent].nstate[i] = *(this->n_state[i]);
        //_info("Crapping memory after prediction of agent %d is [%f %f %f]\n", i, Agents[nAgent].memory[i], Agents[nAgent].ostate[i], Agents[nAgent].nstate[i]);
    }
    
    for (int i=0; i<nRecurr; i++)
    { //memory and states are downloaded from agents (son of a french cow)
        Agents[nAgent].memory[i] = *(this->memory_out[i]);
        //_info("Crapping memory before prediction of agent %d is [%f]\n", i, Agents[nAgent].memory[i]);
    }
}

void NetworkLSTM::predict(const vector<double>& input, const vector<double>& memoryin, const vector<double>& ostate,  vector<double>& nstate,  vector<double>& output)
{ // does not affect memory of agents
    for (int i=0; i<nInputs; i++)
    {
        *(this->inputs[i]) = input[i];
        //_info("%f\n", input[i]);
    }
    //cout << endl;
    
    for (int i=0; i<nMems; i++)
    {
        *(this->o_state[i]) = ostate[i];
        *(this->n_state[i]) = nstate[i];
        //_info("Crapping memory before testing of cell %d is [%f %f %f]\n", i, memoryin[i], ostate[i], nstate[i]);
    }
    for (int i=0; i<nRecurr; i++)
    {
        *(this->memory_in[i]) = memoryin[i];
        //_info("Crapping memory before testing of cell %d is [%f]\n", i, memoryin[i]);
    }
    
    for (int i=0; i<nLayers; i++)
        layers[i]->propagate();
    
    for (int i=0; i<nOutputs; i++)
        output[i] = *(this->outputs[i]);
    /*
    for (int i=0; i<nMems; i++)
    { //new state is uploaded to agents
        //_info("Crapping memory after testing of cell %d is [%f %f %f]\n", i, memoryin[i], ostate[i], nstate[i]);
    }
     */
}

void NetworkLSTM::improve(const vector<double>& input, const vector<double>& error, int nAgent)
{
    for (int i=0; i<nMems; i++)
    { //improve does not change memory state
        *(this->o_state[i]) = Agents[nAgent].ostate[i];
        *(this->n_state[i]) = Agents[nAgent].nstate[i];
        //_info("Crapping memory before improving of cell %d is [%f %f %f]\n", i, Agents[nAgent].memory[i], Agents[nAgent].ostate[i], Agents[nAgent].nstate[i]);
    }
    
    for (int i=0; i<nRecurr; i++)
    { //improve does not change memory state
        *(this->memory_in[i]) = Agents[nAgent].memory[i];
        //_info("Crapping memory before improving of cell %d is [%f]\n", i, Agents[nAgent].memory[i]);
    }
    
    double signal = 0.0;
    int Isignal;
    
    for (int i=0; i<nOutputs; i++)
    {
        *(this->errors[i]) = 0.0;
        if(fabs(error[i])>fabs(signal))
        {
            Isignal = i;
            signal = error[i];
        }
    }
    
    *(this->errors[Isignal]) = 1.;
    
    if (fabs(olderr)>fabs(signal))
    {
        eta*=0.99;
        olderr = signal;
    }
    
    for (int i=nLayers-1; i>=0; i--)
        layers[i]->backPropagate();

    for (int i=0; i<nLayers; i++)
        layers[i]->adjust(signal, eta, alpha, lambda, kappa);

    /*
    for (int i=0; i<nMems; i++)
    { //new state is uploaded to agents
        //_info("Crapping memory after improving of agent %d is [%f %f %f]\n", i, Agents[nAgent].memory[i], Agents[nAgent].ostate[i], Agents[nAgent].nstate[i]);
    }
     */
}

double NetworkLSTM::TotSumWeights()
{
    double sumW=0.;
    for (int i=0; i<nLayers; i++)
        sumW+=layers[i]->TotSumWeights();
    return sumW;
}

MemoryCell::MemoryCell() : Neuron(new Tanh), Sc_new(0.0), Sc_old(0.0)
{
    Input = new Neuron(new Tanh2);
    ScN = new Neuron(new Linear);
    ScO = new Neuron(new Linear);
}

void MemoryCell::init_dsdw()
{
    dsdw_IN.resize(Input->inLinks.size());
    dsdw_IG.resize(IG->inLinks.size());
    dsdw_FG.resize(FG->inLinks.size());
    dsdw_INo.resize(Input->inLinks.size());
    dsdw_IGo.resize(IG->inLinks.size());
    dsdw_FGo.resize(FG->inLinks.size());
}

MemoryBlock::MemoryBlock(int nCellpB) : nMemoryCells(nCellpB)
{
    IG = new Neuron(new Sigm);
    FG = new Neuron(new Sigm);
    OG = new Neuron(new Sigm);
    
    mCells.resize(this->nMemoryCells);
    for (int i=0; i < this->nMemoryCells; i++)
    {
        mCells[i] = new MemoryCell();
        
        mCells[i]->IG = IG;
        mCells[i]->FG = FG;
        mCells[i]->OG = OG;
    }
}

void MemoryBlock::init_dsdw()
{
    for (int i=0; i<nMemoryCells; i++)
        mCells[i]->init_dsdw();
}

HiddenLayer::HiddenLayer(int nBlocks, int nCellpB, int nNeurons, ActivationFunction* func, RNG* rng) : nNeurons(nNeurons), nMemoryBlocks(nBlocks), nCellpB(nCellpB)
{
    mBlocks.resize(this->nMemoryBlocks);
    for (int i=0; i < this->nMemoryBlocks; i++)
    {
        mBlocks[i] = new MemoryBlock(nCellpB);
        for (int j=0; j<nCellpB; j++)
        {
            link(mBlocks[i]->IG, mBlocks[i]->mCells[j]->ScO, rng, false);
            link(mBlocks[i]->FG, mBlocks[i]->mCells[j]->ScO, rng, false);
            link(mBlocks[i]->OG, mBlocks[i]->mCells[j]->ScN, rng, false);
        }
    }
    
    neurons.resize(this->nNeurons);
    for (int i=0; i < this->nNeurons; i++)
        neurons[i] = new Neuron(func);
    
    basePos = new Neuron(new Linear);
    baseNeg = new Neuron(new Linear);
    
    basePos->oval =  1.;
    baseNeg->oval = -1.;
    basePos->ival =  1.;
    baseNeg->ival = -1.;
}

HiddenLayer::HiddenLayer(int nNeurons, ActivationFunction* func) : nNeurons(nNeurons), nMemoryBlocks(0), nCellpB(0)
{
    mBlocks.clear(); //nothing to see here
    neurons.resize(this->nNeurons);
    for (int i=0; i < this->nNeurons; i++)
        neurons[i] = new Neuron(func);
    
    basePos = new Neuron(new Linear);
    baseNeg = new Neuron(new Linear);
    
    basePos->oval =  1.;
    baseNeg->oval = -1.;
    basePos->ival =  1.;
    baseNeg->ival = -1.;
}

void HiddenLayer::link(Neuron* Nto, Neuron* Nfrom, RNG* rng, double ground = 0.)
{
    Link* lnk = new Link;
    lnk->neuronTo   = Nto;
    lnk->neuronFrom = Nfrom;
    
    lnk->w = rng->uniform(-1., 1.);
    if(ground>0)
        lnk->w = ground; //why the hell not! (remember, we'll normalise w/=Ninputs later)
    
    lnk->Dw = 0;
    lnk->val = 0;
    lnk->prevDw = 0;
    lnk->etar = 10.;
    lnk->rr = 0.0;
    lnk->maxrr = 1e-9;
    lnk->Delta = 0.5;
    
    Nfrom->outLinks.push_back(lnk);
    Nfrom->hasOutputs = true;
    Nto->inLinks.push_back(lnk);
    Nto->hasInputs = true;
}

void HiddenLayer::connect2layers(HiddenLayer* prev, RNG* rng, int dist)
{
    //connect the memory
    if(dist>=0) //link to all PREVIOUS rows of neurons and memory
    for (int i=0; i<nMemoryBlocks; i++)
    {
        for (int j=0; j<prev->nNeurons; j++)
        {
            link(mBlocks[i]->OG, prev->neurons[j], rng);
            link(mBlocks[i]->IG, prev->neurons[j], rng);
            link(mBlocks[i]->FG, prev->neurons[j], rng);
            
            for (int k=0; k<nCellpB; k++)
            link(mBlocks[i]->mCells[k]->Input, prev->neurons[j], rng);
        }
        
        if(prev->nMemoryBlocks>0)
        for (int j1=0; j1<prev->nMemoryBlocks; j1++)
        for (int j2=0; j2<prev->nCellpB; j2++)
        {
            link(mBlocks[i]->OG, prev->mBlocks[j1]->mCells[j2], rng);
            link(mBlocks[i]->IG, prev->mBlocks[j1]->mCells[j2], rng);
            link(mBlocks[i]->FG, prev->mBlocks[j1]->mCells[j2], rng);
            
            for (int k=0; k<nCellpB; k++)
            link(mBlocks[i]->mCells[k]->Input, prev->mBlocks[j1]->mCells[j2], rng);
        }
    }
    //connect the neurons
    for (int i=0; i<nNeurons; i++)
    {
        if(dist==0) //link only to the previous hidden layer
        for (int j=0; j<prev->nNeurons; j++)
        link(neurons[i], prev->neurons[j], rng);
        //debug7("Coupling: %f to %f", neurons[i]->inLinks.back()->w, prev->neurons[j]->outLinks.back()->w);
        
        for (int j1=0; j1<prev->nMemoryBlocks; j1++) //link to ALL memory up to this one
        for (int j2=0; j2<prev->nCellpB; j2++)
        link(neurons[i], prev->mBlocks[j1]->mCells[j2], rng);
    }
}

void HiddenLayer::normaliseWeights()
{
    for (int i=0; i<nMemoryBlocks; i++)
    {
        for (int j=0; j<mBlocks[i]->OG->inLinks.size(); j++)
        mBlocks[i]->OG->inLinks[j]->w /= 2. * sqrt( mBlocks[i]->OG->inLinks.size() / 12.);
        
        for (int j=0; j<mBlocks[i]->FG->inLinks.size(); j++)
        mBlocks[i]->FG->inLinks[j]->w /= 2. * sqrt( mBlocks[i]->FG->inLinks.size() / 12.);
        
        for (int j=0; j<mBlocks[i]->IG->inLinks.size(); j++)
        mBlocks[i]->IG->inLinks[j]->w /= 2. * sqrt( mBlocks[i]->IG->inLinks.size() / 12.);
        
        for (int k=0; k<nCellpB; k++)
            for (int j=0; j<mBlocks[i]->mCells[k]->inLinks.size(); j++)
                mBlocks[i]->mCells[k]->inLinks[j]->w /= 2. * sqrt( mBlocks[i]->mCells[k]->inLinks.size() / 12.);

    }
    
    for (int i=0; i<nNeurons; i++)
        for (int j=0; j<neurons[i]->inLinks.size(); j++)
            neurons[i]->inLinks[j]->w /= 2. * sqrt( neurons[i]->inLinks.size() / 12.);
    
}

double HiddenLayer::TotSumWeights()
{
    double sumW(0.0);
    for (int i=0; i<nMemoryBlocks; i++)
    {
        for (int j=0; j<mBlocks[i]->OG->inLinks.size(); j++)
            sumW+= mBlocks[i]->OG->inLinks[j]->w*mBlocks[i]->OG->inLinks[j]->w;
        
        for (int j=0; j<mBlocks[i]->FG->inLinks.size(); j++)
            sumW+= mBlocks[i]->FG->inLinks[j]->w*mBlocks[i]->FG->inLinks[j]->w;
        
        for (int j=0; j<mBlocks[i]->IG->inLinks.size(); j++)
            sumW+= mBlocks[i]->IG->inLinks[j]->w*mBlocks[i]->IG->inLinks[j]->w;
        
        for (int k=0; k<nCellpB; k++)
            for (int j=0; j<mBlocks[i]->mCells[k]->inLinks.size(); j++)
                sumW+= mBlocks[i]->mCells[k]->inLinks[j]->w*mBlocks[i]->mCells[k]->inLinks[j]->w;
    }
    
    for (int i=0; i<nNeurons; i++)
        for (int j=0; j<neurons[i]->inLinks.size(); j++)
            sumW+= neurons[i]->inLinks[j]->w*neurons[i]->inLinks[j]->w;
    return sumW;
}

void HiddenLayer::connect2memstate(vector<double*>& memory, vector<double*>& Sc_old, vector<double*>& Sc_new, int firstm, int firstr)
{ //first = sum( memorySize[i]*nCellpB[i] )
    for (int i=0; i<nMemoryBlocks; i++)
    for (int j=0; j<nCellpB; j++)
    {
        int indr = i*nCellpB + j + firstr;
        int indm = i*nCellpB + j + firstm;
        memory[indr] = &(mBlocks[i]->mCells[j]->oval);
        Sc_new[indm] = &(mBlocks[i]->mCells[j]->Sc_new);
        Sc_old[indm] = &(mBlocks[i]->mCells[j]->Sc_old);
    }
    
    for (int i=0; i<nNeurons; i++)
    {
        int ind = i + nMemoryBlocks*nCellpB + firstr;
        memory[ind] = &(neurons[i]->oval);
    }
}

void HiddenLayer::connect2ground(RNG* rng)
{
    if(nMemoryBlocks>0)
    for (int i=0; i<nMemoryBlocks; i++)
    {
        link(mBlocks[i]->OG, baseNeg, rng, 10.); //at the beginning the OG and IG
        link(mBlocks[i]->IG, baseNeg, rng, 10.); //gates are closed: negative bias
        link(mBlocks[i]->FG, basePos, rng, 10.); //FG is open
        
        for (int k=0; k<nCellpB; k++)
        link(mBlocks[i]->mCells[k]->Input, baseNeg, rng);
    }
    
    for (int i=0; i<nNeurons; i++)
    link(neurons[i], baseNeg, rng);
    
    //ground can be propagated only once: no update of oval needed
    for (int i=0; i<baseNeg->outLinks.size(); i++)
    baseNeg->outLinks[i]->val = baseNeg->oval;
    for (int i=0; i<basePos->outLinks.size(); i++)
    basePos->outLinks[i]->val = basePos->oval;
}

void HiddenLayer::connect2inputs(vector<double*>& vals, vector<double*>& mems)
{
    for (int i=0; i<vals.size(); i++)
    vals[i] = &(neurons[i]->ival);
    
    for (int i=0; i<mems.size(); i++)
    mems[i] = &(neurons[i+vals.size()]->ival);
}

void HiddenLayer::connect2outputs(vector<double*>& vals)
{
    for (int i=0; i<nNeurons; i++)
    vals[i] = &(neurons[i]->oval);
}

void HiddenLayer::connect2errors(vector<double*>& errs)
{
    for (int i=0; i<nNeurons; i++)
    errs[i] = &(neurons[i]->err);
}

void HiddenLayer::propagate()
{   //memory first
    #pragma omp parallel for
    for (int i=0; i<nMemoryBlocks; i++)
    mBlocks[i]->exec();
    #pragma omp parallel for
    for (int i=0; i<nNeurons; i++)
    neurons[i]->exec();
}

void HiddenLayer::backPropagate()
{   //first in last out
    #pragma omp parallel for
    for (int i=0; i<nNeurons; i++)
    neurons[i]->backExec();
    #pragma omp parallel for
    for (int i=0; i<nMemoryBlocks; i++)
    mBlocks[i]->backExec();
}

void HiddenLayer::adjust(double error, double eta, double alpha, double lambda, double kappa)
{
    #pragma omp parallel for
    for (int i=0; i<nMemoryBlocks; i++)
        mBlocks[i]->adjust(error, eta, alpha, lambda, kappa);
    #pragma omp parallel for
    for (int i=0; i<nNeurons; i++)
        neurons[i]->adjust(error, eta, alpha, lambda, kappa);
}

void HiddenLayer::init_dsdw()
{
    for (int i=0; i<nMemoryBlocks; i++)
    mBlocks[i]->init_dsdw();
}

void NetworkLSTM::save(string fname)
{
    //TODO: save error derivatives dsdw
    //TODO: save eligibility trace epsilon
    //TODO: save memory structure numbers
    
    /*  here:
     1) save all inlinks neuron weights
     2) save all inlinks memory blocks weights
     3) save memstate and memory from agents
     */
    
    debug1("Saving into %s\n", fname.c_str());
    
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());
    
    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
    
    out.precision(20);
    
    out << nInputs << " " << nLayers << endl;
    for(int i=0; i<nLayers; i++)
        out << layers[i]->nNeurons << "  ";
    out << endl;
    
    //*************************************************************************
    for (int l=0; l<nLayers; l++)
        for (int n=0; n<layers[l]->nNeurons; n++)
            for (int lnk=0; lnk<layers[l]->neurons[n]->inLinks.size(); lnk++)
                out << layers[l]->neurons[n]->inLinks[lnk]->w << " ";
    out.flush();
    //*************************************************************************
    
    //*************************************************************************
    for (int l=0; l<nLayers; l++)
        for (int n=0; n<layers[l]->nMemoryBlocks; n++)
        {   //OG
            for (int lnk=0; lnk<layers[l]->mBlocks[n]->OG->inLinks.size(); lnk++)
                out << layers[l]->mBlocks[n]->OG->inLinks[lnk]->w << " ";
            //FG
            for (int lnk=0; lnk<layers[l]->mBlocks[n]->FG->inLinks.size(); lnk++)
                out << layers[l]->mBlocks[n]->FG->inLinks[lnk]->w << " ";
            //IG
            for (int lnk=0; lnk<layers[l]->mBlocks[n]->IG->inLinks.size(); lnk++)
                out << layers[l]->mBlocks[n]->IG->inLinks[lnk]->w << " ";
            //IN
            for (int i=0; i<layers[l]->mBlocks[n]->nMemoryCells; ++i)
                for (int lnk=0; lnk<layers[l]->mBlocks[n]->mCells[i]->inLinks.size(); lnk++)
                        out << layers[l]->mBlocks[n]->mCells[i]->inLinks[lnk]->w << " ";
        }
    out.flush();
    //*************************************************************************
    
    //*************************************************************************
    for (int a=0; a<Agents.size(); a++)
        for (int n=0; n<Agents[a].nMems; n++)
                out << Agents[a].ostate[n] << " " << Agents[a].nstate[n] << " ";
    
    for (int a=0; a<Agents.size(); a++)
        for (int n=0; n<Agents[a].nRecurr; n++)
            out << Agents[a].memory[n] << " ";
    out.flush();
    //*************************************************************************
    out.close();
    
    // Prepare copying command
    string command = "cp ";
    string nameOriginal = fname;
    command = command + nameBackup + " " + nameOriginal;
    
    // Submit the command to the system
    system(command.c_str());
}


