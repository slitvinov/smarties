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

#include "../Environments/Environment.h"

using namespace std;

class Saver
{
protected:
	int period;
	ofstream* file;
	
public:
    static string folder;

	Saver (ostream* cOut) : file((ofstream*)cOut) { };
	Saver (string fname)
	{
		file = new ofstream((folder+fname).c_str());
	};
	Saver (){};
	
	virtual void setEnvironment(Environment*) { };
	inline  void setPeriod(int);
	inline  int  getPeriod();
	
	inline static bool makedir(string name)
	{
		folder = name;
		if (mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) return false;
		return true;
	}
	
	virtual void exec() = 0;
};

inline void Saver::setPeriod(int p)
{
	period = p;
}

inline int Saver::getPeriod()
{
	return period;
}
