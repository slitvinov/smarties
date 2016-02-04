/*
 *  Saver.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <errno.h>
#include <string>

#include "../Scheduler/Scheduler.h"

using namespace std;

class Master;

class Saver
{
protected:
	int period;
	ofstream* file;
	Master* master;
	
public:
    static string folder;

	Saver (ostream* cOut, int period = 1) : file((ofstream*)cOut), period(period) { };
	Saver (string fname, int period = 1) : period(period)
	{
		file = new ofstream((folder+fname).c_str());
	};
	Saver () : file(NULL) {};
	
    ~Saver() { if (file != NULL) file->close(); }

	void setMaster(Master* m)
	{
	    master = m;
	};
	
	inline static bool makedir(string name)
	{
		folder = name;
		if (mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) return false;
		return true;
	}
	
	virtual bool isReady(Real time, int iter)
	{
	    return (iter % period) == 0;
	}

	virtual void exec() = 0;
};
