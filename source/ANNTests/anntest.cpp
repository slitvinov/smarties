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
#include "../ANN/lwpr.h"
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
		
	return 1.0 * res / 50.0 / x.size();
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
	vector<int> lsize;
	lsize.push_back(14);
	lsize.push_back(10);
	lsize.push_back(1);

	
	NetworkLM ann(lsize, 5.0);
	//Network ann(lsize, 0.005, 0.9);
	//WaveletNetLM ann(lsize);
	vector<double> res;
	vector<double> prediction(1);
	vector<double> err(1);
	vector<vector<double> > x;
	vector<double> tvec(14);
	
	ifstream inf("../training.csv");
	if (!inf.good())
	{
		die("ololo\n");
	}
	
	double tres;
	while (inf.good())
	{
		vread(inf, tvec, tres);
		x.push_back(tvec);
		res.push_back(tres);
	}
	
	info("Entries: %d\n", x.size());
	
	double cerr = 0;
	for (int i=0; i<x.size() * 500; i++)
	{
		int k = floor(rng.uniform(0, x.size()));
		double exact = res[k];//+ rng.normal(0, 0.002);
		
		ann.predict(x[k], prediction);
		err[0] = (prediction[0] - exact);
		
		debug("Res:  %f, exact  %f,\terr: \t%f\n", prediction[0], exact, err[0]);
		ann.improve(x[k], err);
		
		cerr = max(fabs(err[0]), cerr);
		if (i % 200 == 0)
		{
			printf("%f\n", i, (cerr)/1.0);
			cerr = 0;
		}
	}
	
	inf.close();
	inf.open("../validation.csv");
	ofstream outf("../res.cvs");
	if (!inf.good())
	{
		die("ololo\n");
	}
	
	while (inf.good())
	{
		vread(inf, tvec, tres, false);
		
		ann.predict(tvec, prediction);
		outf << prediction[0] * 2e4 << endl;
	}
	outf.close();
	inf.close();
	
	outf.open("../resTest.cvs");
	for (int i=0; i<x.size(); i++)
	{
		ann.predict(x[i], prediction);
		outf << (prediction[0]) * 2e4 << endl;
	}
	
}