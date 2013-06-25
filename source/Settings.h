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
	double scale;
	double centerX, centerY;
	int    saveFreq;
	int    videoFreq;
	string configFile;
	double dt;
	double endTime;
	int    randSeed;
	
	double lRate;
	double greedyEps;
	double gamma;
	//double learnDump;
	//bool   shared;
	bool   restart;
	
	double nnEta;
	double nnAlpha;
	int    nnLayer1;
	int    nnLayer2;

} settings;
