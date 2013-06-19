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

#include "../Environments/Environment.h"

using namespace std;

class Saver
{
protected:
	int period;
	ofstream* file;
	
public:
	Saver (ofstream* newFile) : file(newFile) { };
	Saver (){};
	virtual void setEnvironment(Environment*) { };
	inline  void setPeriod(int);
	inline  int  getPeriod();
	
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
