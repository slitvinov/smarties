/*
 *  Settings.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <string>
using namespace std;

extern struct Settings
{
	int    saveFreq;
	int    videoFreq;
	string configFile;
	double dt;
	double endTime;
	int    randSeed;
	
	double lRate;
	double greedyEps;
	double gamma;
	double lambda;
	string restart;
	
	double nnEta;
	double nnAlpha;
    double nnKappa;
    double nnLambda;
    double nnAdFac;
	int    nnLayer1;
	int    nnLayer2;
    int    nnLayer3;
    int    nnLayer4;
    int    nnLayer5;
    int    nnMemory1;
    int    nnMemory2;
    int    nnMemory3;
    int    nnMemory4;
    int    nnMemory5;
    int    nnOuts;
    
    double  AL_fac;
    string learner;
    string network;
    
	bool best;
	bool immortal;
	string prefix;
	
} settings;
