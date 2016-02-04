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
//using Real=float;
#define Real double
extern struct Settings
{
	int    saveFreq;
	int    videoFreq;
	string configFile;
	Real dt;
	Real endTime;
	int    randSeed;
	
	Real lRate;
	Real greedyEps;
	Real gamma;
	Real lambda;
	string restart;
	
	Real nnEta;
	Real nnAlpha;
    Real nnKappa;
    Real nnLambda;
    Real nnAdFac;
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
    
    Real  AL_fac;
    string learner;
    string network;
    
	bool best;
	bool immortal;
	string prefix;
	
} settings;
