/*
 *  anntest.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <random>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <ostream>
#include <sstream>
#include <iostream>
#include "../Settings.h"
#include "../ANN/LSTMNet.h"
#include "../ArgumentParser.h"
#include "../ErrorHandling.h"

using namespace ErrorHandling;
//using namespace ArgumentParser;
using namespace std;
Settings settings;
int ErrorHandling::debugLvl;

double func(double x) { return x * std::sin(x); }

inline Real sqr(Real x)
{
	return x*x;
}
Real target1(vector<Real>& x)
{
	return 3*x[0]*x[0] + sin(x[1]);
}
Real target2(vector<Real>& x)
{
	Real res = 0;
	for (int i=0; i+1 < static_cast<int>(x.size()); i++)
		res += sqr(1-x[i]) + 100*sqr(x[i+1] - x[i]*x[i]);
		
	return 1.0 * res / 10.0 / x.size();
}
Real target3(vector<Real>& x)
{
    return cos(x[0]) + sin(x[1]);
}
Real target4(vector<Real>& x)
{
    return (2*x[1] - 5*x[0]);
}

Real target5(vector<Real>& _x)
{
    Real x = _x[0]*5.+5.;
    //Real x = _x[0];
    return (exp(cos(5*x))*sin(x) +sin(3*x) -cos(x)*x*x/10.0);
}

Real target5(Real _x)
{
    Real x = _x*5.+5.;
    return (exp(cos(5*x))*sin(x) +sin(3*x) -cos(x)*x*x/10.0);
}

const Real fact[14] = {8, 160, 80, 80, 160, 16, 8, 32000, 1024, 32, 1024, 1024, 8000, 36};

void vread(ifstream &in, vector<Real> &x, Real &res, bool train = true)
{
	Real val;
	char c;
	
	for (int i=0; i<14; i++)
	{
		in >> val >> c;
		x[i] = val/fact[i];
	}
	
	if (train)
	{
		in >> res;
		res /= 2e4;
	}
}

#define FISH
#define NIN 32
#define OUT 5
//#define ROSENSERIES
//#define NIN 1
//#define OUT 1

int main (int argc, char** argv)
{
    {
        struct timeval time;
        gettimeofday(&time, NULL);
        int seed = abs(84967194 + floor(time.tv_usec));
        settings.configFile=(string)"factory";
        settings.dt=0.01;
        settings.endTime=1e9;
        settings.gamma=0.95;
        settings.greedyEps=0.01;
        settings.lRate=0.1;
        settings.lambda=0.0;
        settings.randSeed=seed;
        settings.restart=(string)"none";
        settings.saveFreq=1000;
        debugLvl=0;
        settings.prefix=(string)"res/";
        settings.nnEta=0.001;
        settings.nnAlpha=0.5;
        settings.nnLambda=0.0;
        settings.nnKappa=0.0;
        settings.nnAdFac=1e-6;
        settings.AL_fac=0.0;
#if 1
        settings.nnLayer1 =0;
        settings.nnLayer2 =0;
        settings.nnLayer3 =0;
        settings.nnMemory1=32;
        settings.nnMemory2=16;
        settings.nnMemory3=8;
#else
        settings.nnLayer1 =32;
        settings.nnLayer2 =16;
        settings.nnLayer3 =8;
        settings.nnMemory1=0;
        settings.nnMemory2=0;
        settings.nnMemory3=0;
#endif
        settings.learner= (string)"NFQ";
    }
	debugLvl = 9;
    
    std::mt19937 gen(settings.randSeed);
    std::uniform_real_distribution<Real> dis(-1.,1.);
    std::uniform_real_distribution<Real> dis2(0.5,1.5);
    
	vector<int> lsize, mblocks;
    lsize.push_back(NIN);
    lsize.push_back(settings.nnLayer1);
    if (settings.nnLayer2>0 || settings.nnMemory2>0)
    {
        lsize.push_back(settings.nnLayer2);
        if (settings.nnLayer3>0 || settings.nnMemory3>0)
            lsize.push_back(settings.nnLayer3);
    }
    lsize.push_back(OUT);
    
    mblocks.push_back(0);
    mblocks.push_back(settings.nnMemory1);
    if (settings.nnLayer2>0 || settings.nnMemory2>0)
    {
        mblocks.push_back(settings.nnMemory2);
        if (settings.nnLayer3>0 || settings.nnMemory3>0)
            mblocks.push_back(settings.nnMemory3);
    }
    mblocks.push_back(0);
    
    FishNet ann(lsize, mblocks, settings, 1);
	vector<Real> in(NIN), tgt(OUT);
    int nepochs(1000);
    
#if defined(FISH)
    //vector<vector<vector<Real>>> inputs, targets;
    //vector<vector<Real>> testx, testy, test;
    vector<vector<vector<Real>>> input, target;
           vector<vector<Real>> tmp_i, tmp_o;
                  vector<Real>  testi(4), dmp_i(32), dmp_o(5);
    for (int i=1; i<= 563; i++)
    {
        string numbered_input, numbered_output;
        {
            char buf[500];
            sprintf(buf, "/cluster/home/novatig/2Fish_Data2/twofish_%03d/info.dat", (int)i);
            numbered_output = string(buf);
        }
        ifstream inout(numbered_output.c_str());
        {
            Real dmp;
            int j = 0;
            std::string line;
            while (std::getline(inout, line))
            {
                std::istringstream iss(line);
                iss >> dmp >> dmp_o[0] >> dmp_o[1] >> dmp_o[2] >> dmp_o[3] >> dmp_o[4];//)) { cout << endl << "Failed to open file " << numbered_output  << endl; break; } // error
                dmp_o[0]*= 3.0;
                dmp_o[1]*= 3.0;
                dmp_o[2]*= 2.4;
                dmp_o[3]*= 0.2;
                dmp_o[4]*= 1e6;
                
                if(j==0)
                    dmp_o[3] = 0.;
                if(j==0)
                    dmp_o[4] = 0.;
                //for (int k=0; k<5; k++)
                //    cout << dmp_o[k] << " ";
                //cout << endl;
                tmp_o.push_back(dmp_o);
                j++;
            }
            
            target.push_back(tmp_o);
            inout.close();
            //cout << target.size() << " " << tmp_o.size() << " " << dmp_o.size() << endl;
            tmp_o.clear();
        }
        {
            char buf[500];
            sprintf(buf, "/cluster/home/novatig/2Fish_Data2/twofish_%03d/data.dat", (int)i);
            numbered_input = string(buf);
        }
        ifstream ininp(numbered_input.c_str());
        {
            int j = 0;
            std::string line;
            while (std::getline(ininp, line))
            {
                std::istringstream iss(line);
                if (!(iss >> dmp_i[0] >> dmp_i[1] >> dmp_i[2] >> dmp_i[3] >> dmp_i[4] >> dmp_i[5] >> dmp_i[6] >> dmp_i[7] >> dmp_i[8] >> dmp_i[9] >> dmp_i[10] >> dmp_i[11] >> dmp_i[12] >> dmp_i[13] >> dmp_i[14] >> dmp_i[15] >> dmp_i[16] >> dmp_i[17] >> dmp_i[18] >> dmp_i[19] >> dmp_i[20] >> dmp_i[21] >> dmp_i[22] >> dmp_i[23] >> dmp_i[24] >> dmp_i[25] >> dmp_i[26] >> dmp_i[27] >> dmp_i[28] >> dmp_i[29] >> dmp_i[30] >> dmp_i[31])) { cout << endl << "Failed to open file " << numbered_input << endl; break; } // error

                //testi[0] = dis(gen); testi[1] = dis(gen); testi[2] = dis(gen); testi[3] = dis(gen);
                tmp_i.push_back(dmp_i);
                //for (int k=0; k<32; k++)
                //    cout << dmp_i[k] << " ";
                //cout << endl;
            }
            //cout << i << " "<< dmp_i[0] << " "<< dmp_i[1] << " "<< dmp_i[2] << " "<< dmp_i[3] << " "<< dmp_i[4] << endl;
        }
        ininp.close();
        input.push_back(tmp_i);
        //cout << input.size() << " " << tmp_i.size() << " " << dmp_i.size() << endl;
        tmp_i.clear();
    }
    
    ann.train(input, target, nepochs);
    ann.predict(input[192], tmp_o);
    
    ofstream outf1("LSTM.dat");
    for (int k=0; k<120; k++)
        outf1 << k << " "  << tmp_o[k][0] << " "  << tmp_o[k][1] << " "  << tmp_o[k][2] << " "  << tmp_o[k][3] << " "  << tmp_o[k][4] << " "  << endl;
    outf1.close();
    
    ofstream outf2("truth.dat");
    for (int k=0; k<120; k++)
        outf2 << k << " "  << target[192][k][0] << " "  << target[192][k][1] << " "  << target[192][k][2] << " "  << target[192][k][3] << " "  << target[192][k][4] << " "  << endl;
    outf2.close();
#elif defined(SERIES)
    Real dt(0.01);
    Real ndata(10000), series(5);
    vector<vector<vector<Real>>> inputs, targets;
    vector<vector<Real>> testx, testy, test;
    for (int j=0; j<ndata+1; j++)
    {
        Real x = 0;
        Real y = 0;
        Real theta = 0;
        Real velx  = 0;
        Real vely  = 0;
        vector<vector<Real>> inSeries;
        vector<vector<Real>> tgtSeries;
        for (int k=0; k<series; k++)
        {
            #if OUT == 2
            Real dtheta = dis(gen)*M_PI/6;
            theta += dtheta;
            #if NIN == 2
            Real acc    = dis2(gen);
            #else
            Real acc    = 1.;
            #endif
            
            Real accx = acc*cos(theta);
            Real accy = acc*sin(theta);
            vely += accy*dt;
            //y += vely*dt;
            y = dis(gen);
            tgt[1] = y;
            #else
            Real accx = dis(gen);
            #endif
            
            velx += accx*dt;
            //x += velx*dt;
            x = dis(gen);
            tgt[0] = x;
            
            #if NIN == 2
            in[1] = accy;
            in[0] = accx;
            #else
            #if OUT == 2
            in[0] = theta;
            #else
            in[0] = accx;
            #endif
            #endif
            
            tgtSeries.push_back(tgt);
            inSeries.push_back(in);
            //printf("%f %f %f %f\n",x,y,in[0],in[1]);
        }
        if (j==ndata)
        {
            testx = inSeries;
            testy = tgtSeries;
        }
        else
        {
            inputs.push_back(inSeries);
            targets.push_back(tgtSeries);
        }
    }
    
    ann.train(inputs, targets, nepochs);
    ann.predict(testx, test);
    
    ofstream outf1("sanity.dat");
    for (int k=0; k<series; k++)
    {
        #if OUT == 2
        outf1 << testy[k][0] << " " << testy[k][1] << " "
               << test[k][0] << " "  << test[k][1] << endl;
        #else
        outf1 << testx[k][0] << " " << testy[k][0] << " "  << test[k][0] << endl;
        #endif
        
    }
    outf1.close();
#elif defined(ROSENSERIES)
    int nseries = 10; int nrosen = 5;
                  vector<Real>   tmp(nrosen);
           vector<vector<Real>>  tmpo(nseries), tmpi(nseries);
    vector<vector<vector<Real>>> inputs, targets;
    
    for (int j=0; j<20000; j++)
    {
        for (int i=0; i<nrosen; i++)
            tmp[i] = 0;
        
        for (int k=0; k<nseries; k++)
        {
            tmp[NIN-1] = dis(gen);
            in[0] = tmp[NIN-1];
            tgt[0] = target2(tmp);
            tmpi[k] = in;
            tmpo[k] = tgt;
            
            for (int i=0; i+1<nseries; i++)
                tmp[i] = tmp[i+1];
        }
        
        inputs.push_back(tmpi);
        targets.push_back(tmpo);
    }
    ann.train(inputs, targets, nepochs);
    
#elif defined(ROSEN)
    int batchsize(1);
    vector<vector<Real>> inputs, targets;
    for (int j=0; j<70000; j++)
    {
        for (int i=0; i+1<NIN; i++)
            in[i] = in[i+1];
        
        in[NIN-1] = dis(gen);
        tgt[0] = target2(in);
        inputs.push_back(in);
        targets.push_back(tgt);
    }
    ann.train(inputs, targets, batchsize, nepochs);
    ofstream outf1("sanity.dat");
    for (in[0]=-1; in[0]<=1; in[0]+=.05)
    for (in[1]=-1; in[1]<=1; in[1]+=.05)
    {
        ann.predict(in, tgt, 0);
        Real exact = target2(in);
        outf1 << in[0] << " " << in[1] << " " << tgt[0] << " " << exact << endl;
    }
    outf1.close();
#else
    int batchsize(120);
    vector<vector<Real>> inputs, targets;
    for (int j=0; j<70000; j++)
    {
        for (int i=0; i<NIN; i++)
        {
            in[i] = dis(gen);
            tgt[i] = target5(in[i]);
        }
        inputs.push_back(in);
        targets.push_back(tgt);
    }
    ann.train(inputs, targets, batchsize, nepochs);
    
    ofstream outf1("sanity.dat");
    for (in[0]=-1; in[0]<=1; in[0]+=.01)
    {
        ann.predict(in, tgt, 0);
        Real exact = target5(in[0]);
        outf1 << in[0] << " " << tgt[0] << " " << exact << endl;
    }
    outf1.close();
#endif
    return 0;
}

/*
 nI 32, iI 0, nO 4, iO 32, iW 12, iC 0, iWI 140, iWF 268, iWO 396, idSdW 0
 nI 4, iI 32, nO 4, iO 32, iW 524, iC 0, iWI 540, iWF 556, iWO 572, idSdW 128
 nNeurons= 4, n1stNeuron= 32, n1stBias= 0
 n1stCell= 0, n1stPeep= 0, n1stBiasIG= 4, n1stBiasFG= 8, n1stBiasOG= 12, n1stdSdB= 0
 nI 4, iI 32, nO 4, iO 36, iW 600, iC 4, iWI 616, iWF 632, iWO 648, idSdW 144
 nI 4, iI 36, nO 4, iO 36, iW 664, iC 4, iWI 680, iWF 696, iWO 712, idSdW 160
 nNeurons= 4, n1stNeuron= 36, n1stBias= 16
 n1stCell= 4, n1stPeep= 588, n1stBiasIG= 20, n1stBiasFG= 24, n1stBiasOG= 28, n1stdSdB= 20
 nI 4, iI 36, nO 4, iO 40, iW 740, iC 8, iWI 756, iWF 772, iWO 788, idSdW 176
 nI 4, iI 40, nO 4, iO 40, iW 804, iC 8, iWI 820, iWF 836, iWO 852, idSdW 192
 nNeurons= 4, n1stNeuron= 40, n1stBias= 32
 n1stCell= 8, n1stPeep= 728, n1stBiasIG= 36, n1stBiasFG= 40, n1stBiasOG= 44, n1stdSdB= 40
 nI 4, iI 40, nO 5, iO 44, iW 868, iC -1, iWI -1, iWF -1, iWO -1, idSdW -1
 nNeurons= 5, n1stNeuron= 44, n1stBias= 48
 nNeurons= 49, nWeights= 888, nBiases= 53, ndSdW= 208, ndSdB= 60, nStates= 12
 */
