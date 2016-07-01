/*
 *  Profiler.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 30.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "timer.h"
#include "Settings.h"
#include <map>

using namespace std;

struct Timings
{
	bool started;
	int iterations;
	Real start;
	Real total;
};

class Profiler
{	
	enum {SEC, MSEC, MCSEC} mode;
	
	map<string, Timings> timings;
	string ongoing;
	
public:
	Profiler();
	
	void start(string);
	void stop(string);
	void stop();
	string printStat();
	Real elapsed(string);
	
	void sec()      { mode = SEC; }
	void millisec() { mode = MSEC; };
	void microsec() { mode = MCSEC; };
};