/*
 *  AllSavers.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Saver.h"
#include "../AllSystems.h"
#include <string>
#include "../Screenshot.h"

class RewardSaver : public Saver
{
private:
	SelfAvoidEnvironment* env;
	
public:
	RewardSaver(ofstream* f) : Saver(f) { };
	~RewardSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = static_cast<SelfAvoidEnvironment*> (newEnv);
	}

	void exec()
	{
		(*file) << env->getAccumulatedReward() << endl;
		file->flush();
	}
};


class StateSaver : public Saver
{
private:
	SelfAvoidEnvironment* env;
	MultiTable* Q;
	
public:
	StateSaver(ofstream* f) : Saver(f) { };
	~StateSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = static_cast<SelfAvoidEnvironment*> (newEnv);
		Q   = static_cast<MultiTable*> (env->data[0]);
	}
	
	void exec()
	{
		ofstream& out(*file);
		map<long int, double>& data = Q->getData();
		
		info("Saving all the states... ");
		
		for (map<long int, double>::iterator it = data.begin(); it != data.end(); it++)
		{
			State s = decode(env->sI, it->first / 10);
			
			out << s.vals[4] << ", " << s.vals[5] << ", " << it->second << endl;
		}
		out << endl;
		out.flush();
		
		info("Done\n");
	}
};

class PhotoSaver : public Saver
{
private:
	int num;
	string fname;
	
public:
	PhotoSaver(string fname) : Saver(), num(0), fname(fname) { };
	
	void exec()
	{
#ifdef _RL_VIZ
		char buf[100];
		sprintf(buf, "%s%07d.tga", fname.c_str(), num);
		info("Saving screenshot to %s... ", buf);
		num++;
		if (gltWriteTGA(buf))
			info("Ok\n");
		else
			warn("Failed!\n");
#endif
	}
};



