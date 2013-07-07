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
#include "../rng.h"
#include "../ErrorHandling.h"

using namespace ErrorHandling;
int ErrorHandling::debugLvl;

double target1(double x, double y)
{
	return x*x + 5* sin(x*3);
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
	lsize.push_back(6);
	lsize.push_back(6);
	lsize.push_back(6);
	lsize.push_back(2);
	
	NetworkLM ann(lsize, 1.1, 50);
	//Network ann(lsize, 0.1, 0.8);
	vector<double> x(6);
	vector<double> res(2);
	vector<double> err(2);
	
	double cerr = 0;
	for (int i=0; i<20000000; i++)
	{
		x[4] = rng.uniform(0, 1);
		x[5] = rng.uniform(0, 1);
		
		double exact1 = target1(x[4], x[5]) + rng.normal(0, 0.01);
		double exact2 = target2(x[4], x[5]) + rng.normal(0, 0.01);
		
		ann.predict(x, res);
		err[0] = res[0] - exact1;
		err[1] = res[1] - exact2;
		
		//printf("Res:  %f, exact  %f,\terr: \t%f\n", res[1], exact2, abs(err[1]));
		ann.improve(err);
		
		cerr = max(fabs(err[0]) + fabs(err[1]), cerr);
		if (i % 5000 == 0)
		{
			printf("Average L1 error:  %f\n", (cerr)/1);
			cerr = 0;
		}
	}
}