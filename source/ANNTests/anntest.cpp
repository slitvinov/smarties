/*
 *  anntest.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <cstdio>
#include <fstream>

#include "../ANN/Network.h"
#include "../ANN/WaveletNet.h"
#include "../rng.h"
#include "../ErrorHandling.h"

using namespace ErrorHandling;
int ErrorHandling::debugLvl;

inline double sqr(double x)
{
	return x*x;
}

double target1(vector<double>& x)
{
	return 3*x[0]*x[0];// + sin(x[1]);
}

double target2(vector<double>& x)
{
	double res = 0;
	for (int i=0; i<x.size()-1; i++)
		res += sqr(1-x[i]) + 100*sqr(x[i+1] - x[i]*x[i]);
		
	return 1.0 * res / 10.0 / x.size();
}

const double fact[14] = {8, 160, 80, 80, 160, 16, 8, 32000, 1024, 32, 1024, 1024, 8000, 36};

void vread(ifstream &in, vector<double> &x, double &res, bool train = true)
{
	double val;
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

//using namespace ANN;

int main (int argc, char** argv)
{
	debugLvl = 2;
	RNG rng(0);
	vector<int> lsize, mblocks, mcells;
    lsize.push_back(7);
    lsize.push_back(13);
    lsize.push_back(7);
    lsize.push_back(1);
    //memory blocks per layer (none in input and output)
    mblocks.push_back(0);
    mblocks.push_back(0);
    mblocks.push_back(0);
    mblocks.push_back(0);
    //num mememory cell per block on layer
    mcells.push_back(0);
    mcells.push_back(0);
    mcells.push_back(0);
    mcells.push_back(0);
    
    //NetworkLSTM ann(lsize, mblocks, mcells, 0.01, 0.1, 1);
	Network ann(lsize, 0.005, 0.9, 1);
	//WaveletNetLM ann(lsize);
	vector<double> res(1);
	vector<double> err(1);
	vector<double> x(7);
	
	double cerr = 0;
    for (int i=0; i<20000000; i++)
    {
        x[0] = rng.uniform(-2, 2);
        x[1] = rng.uniform(-2, 2);
        x[2] = rng.uniform(-2, 2);
        x[3] = rng.uniform(-2, 2);
        x[4] = rng.uniform(-2, 2);
        x[5] = rng.uniform(-2, 2);
        x[6] = rng.uniform(-2, 2);
        
        double exact = target2(x) + rng.normal(0, 0.002);
        //cout << "Exact is: "<< exact << endl;
        ann.predict(x, res);
        err[0] =  -(exact - res[0]);
        
        //printf("Res:  %f, exact  %f,   err: \t%f\n", res[0], exact, err[0]);
        ann.improve(x, err);
        
        cerr += fabs(err[0]);
        if (i % 500 == 0)
        {
            printf("Average error:  %f\n", cerr/5050.0);
            cerr = 0;
        }
        if (err[0]>1e9)
        abort();
    }
    
    return 0;
}