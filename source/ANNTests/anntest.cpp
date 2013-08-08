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

double target1(double x, double y)
{
	return 3*x*x + 0*y;
}

double target2(double x, double y)
{
	return -y*y*4.3 + 5* sin(x*3) - 1;
}

//using namespace ANN;

int main (int argc, char** argv)
{
	debugLvl = 2;
	RNG rng(0);
	vector<int> lsize;
	lsize.push_back(2);
	lsize.push_back(5);
	lsize.push_back(1);

	
	//NetworkLM ann(lsize, 10, 50);
	//Network ann(lsize, 0.01, 0.5);
	WaveletNet ann(lsize, 0.01, 0.8, 500);
	vector<double> x(6);
	vector<double> res(2);
	vector<double> err(2);
	
	double cerr = 0;
	for (int i=0; i<2000000; i++)
	{
		x[0] = rng.uniform(-1, 1);
		x[1] = rng.uniform(-1, 1);
		
		double exact1 = target1(x[0], x[1]);// + rng.normal(0, 0.00001);
		double exact2 = target2(x[0], x[0]) + rng.normal(0, 0.00001);
		
		ann.predict(x, res);
		err[0] = res[0] - exact1;
		//err[1] = res[1] - exact2;
		
		//printf("Res:  %f, exact  %f,\terr: \t%f\n", res[0], exact1, abs(err[0]));
		ann.improve(x, err);
		
		cerr = max(fabs(err[0]), cerr);
		if (i % 5000 == 0)
		{
			printf("Max L1 error:  %f\n", (cerr)/1);
			cerr = 0;
		}
	}
}