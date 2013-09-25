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
		
	return 1.0 * res / 100.0 / x.size();
}

//using namespace ANN;

int main (int argc, char** argv)
{
	vector<double> x(3,0);
	
	debugLvl = 2;
	RNG rng(0);
	vector<int> lsize;
	lsize.push_back(x.size());
	lsize.push_back(20);
	lsize.push_back(1);

	
	//NetworkLM ann(lsize, 5.0);
	//Network ann(lsize, 0.005, 0.9);
	WaveletNetLM ann(lsize);
	vector<double> res(2);
	vector<double> err(2);
	
	double cerr = 0;
	for (int i=0; i<2500000; i++)
	{
		for (int i=0; i<x.size(); i++)
			x[i] = rng.uniform(-1, 1);
		
		double exact1 = target2(x) + rng.normal(0, 0.002);
		
		ann.predict(x, res);
		err[0] = (res[0] - exact1);
		//err[1] = res[1] - exact2;
		
		debug("Res:  %f, exact  %f,\terr: \t%f\n", res[0], exact1, (err[0]));
		ann.improve(x, err);
		
		cerr = max(fabs(err[0]), cerr);
		if (i % 500 == 0)
		{
			printf("%f\n", i, (cerr)/1.0);
			cerr = 0;
		}
	}
}