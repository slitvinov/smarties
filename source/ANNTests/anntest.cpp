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

double target(double x, double y)
{
	return x*x*5 - y*0.77 +1;
}

//using namespace ANN;

int main (int argc, char** argv)
{
	RNG rng(0);
	vector<int> lsize;
	lsize.push_back(2);
	lsize.push_back(5);
	lsize.push_back(5);
	lsize.push_back(5);
	lsize.push_back(3);
	
	Network ann(lsize, 0.1, 0.2);
	vector<double> x(2);
	vector<double> res(3);
	vector<double> err(3);
	
	double cerr = 0;
	for (int i=0; i<20000000; i++)
	{
		x[0] = rng.uniform(0, 1);
		x[1] = rng.uniform(0, 1);
		
		double exact = target(x[0], x[1]);
		
		ann.predict(x, res);
		err[1] = res[1] - exact;
		err[0] = 0;
		
		//printf("Res:  %f, exact  %f,\terr: \t%f\n", res[1], exact, err[1]);
		ann.improve(err);
		
		cerr += fabs(err[1]);
		if (i % 5000 == 0)
		{
			printf("Average error:  %f\n", cerr/5050.0);
			cerr = 0;
		}
	}
}