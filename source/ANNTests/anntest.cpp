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
#include <sys/time.h>
using namespace ErrorHandling;
int ErrorHandling::debugLvl;

inline double sqr(double x)
{
	return x*x;
}

double target1(vector<double>& x)
{
	return 3*x[0]*x[0] + sin(x[1]);
}

double target2(vector<double>& x)
{
	double res = 0;
	for (int i=0; i<x.size()-1; i++)
		res += sqr(1-x[i]) + 100*sqr(x[i+1] - x[i]*x[i]);
		
	return 1.0 * res / 10.0 / x.size();
}

double target3(vector<double>& x)
{
    return cos(x[0]) + sin(x[1]);
}

double target4(vector<double>& x)
{
    return (2*x[1] - 5*x[0]);
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
	debugLvl = 9;
    struct timeval time;
    gettimeofday(&time, NULL);
    cout << time.tv_usec << endl;
	RNG rng(426578 + floor(time.tv_usec));
	vector<int> lsize, mblocks, mcells;
    lsize.push_back(4);
    //lsize.push_back(3);
    lsize.push_back(0);
    lsize.push_back(2);
    lsize.push_back(1);
    //memory blocks per layer (none in input and output)
    mblocks.push_back(0);
    mblocks.push_back(8);
    //mblocks.push_back(6);
    mblocks.push_back(0);
    mblocks.push_back(0);
    //num mememory cell per block on layer
    mcells.push_back(0);
    mcells.push_back(1);
    //mcells.push_back(1);
    mcells.push_back(1);
    mcells.push_back(0);
    
    
    NetworkLSTM ann(lsize, mblocks, mcells, 0.001, 0.5, 0.000001, 0.0, 1);
	//Network ann(lsize, 0.1, 0.1, 0.0000001, 1);
    //NetworkLM ann(lsize, 5, 499);
	//WaveletNetLM ann(lsize);
	vector<double> res(1);
    vector<double> par(4);
	vector<double> err(1);
	vector<double> x(2);
    x[0] = 0.0;
    x[1] = 0.0;
    ofstream outf1("test.dat");
	double mean = 0.0;
    double cerr = 0.0;
    double M2 = 0;
    int count = 0;
    int nmean = 0;
    double oldmean = 0.;
    for (int i=0; i<300000000; i++)
    {
        x[0] = rng.uniform(-1, 1);
        x[1] = rng.uniform(-1, 1);
        par[0] = 10.0;
        par[1] = 10.0;
        par[2] = x[0];
        par[3] = x[1];
        
    for (int j=0; j<5; j++)
    {
        double exact = target4(x);// + rng.normal(0, 0.0);
        //printf("\n Exact is %f %f %f \n", exact, x[0], x[1]);
        //printf("Going to predict with %f %f %f %f \n", par[0], par[1], par[2], par[3]);
        ann.predict(par, res, 0);
        err[0] = (exact - res[0]);
        //printf("error %f\n", err[0]);
        ann.improve(par, err, 0);
        //printf("New loop? %d\n", i);
        //double ang = 3.14159265359*rng.uniform(-1., 1.);
        //double dx = 0.5*std::cos(ang);
        //double dy = 0.5*std::sin(ang);
        
        //dx -= 0.1*fabs(dx)*x[0]*rng.uniform(0., 1.); //slight correction to pull x,y inside the -2 2 domain
        //dy -= 0.1*fabs(dy)*x[1]*rng.uniform(0., 1.);
        //x[0] += dx;
        //x[1] += dy;
        double dx = -x[0];
        double dy = -x[1];
        x[0] = rng.uniform(-1, 1);
        x[1] = rng.uniform(-1, 1);
        dx += x[0];
        dy += x[1];
        
        par[0] = dx;
        par[1] = dy;
        par[2] = -10.0;
        par[3] = -10.0;
        
        //printf("\nDone improving\n");
        nmean++;
        double delta = fabs(err[0])-mean;
        mean += delta/(double)nmean;
        M2 += delta*(fabs(err[0])-mean);
        cerr += fabs(err[0]);
    }
        if (i % 5000 == 0 && i!=0)
        {
            count++;
            printf("Error:  %f, Variance: %f, SumWeights %f, Avg Eta %f\n", cerr/5000, M2/5000, ann.TotSumWeights(), ann.AvgLearnRate());
            //printf("Error:  %f, Variance: %f\n", cerr/5000, M2/5000);
            outf1 << count << " " << cerr/5000 << " " << M2/5000 << endl;
            nmean = 0;
            mean = cerr = M2 = 0.0;
        }
        //if (i % 5000 == 0)
        //   printf("\n \n \n");
    }
    outf1.close();
    /*
    ofstream outf1("sanity.dat");
    vector<double> y(2);
    for (y[0]=-1.; y[0]<1.05; y[0]+=.05)
        for (y[1]=-1.; y[1]<1.05; y[1]+=.05)
        {
            ann.predict(y, res, 0);
            double exact = target1(y);
            outf1 << y[0] << " " << y[1] << " " << res[0] << " " << exact << endl;
        }
    outf1.close();
    */
    return 0;
}
