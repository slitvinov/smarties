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
#include <iomanip>

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


class MomentumSaver : public Saver
{
private:
	Environment* env;
	
public:
	MomentumSaver(ofstream* f) : Saver(f) { };
	~MomentumSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = newEnv;
	}
	
	void exec()
	{
		int tot = 0;
		vector<CouzinAgent*>& agents = *(static_cast< vector<CouzinAgent*> *> (env->data["couzins"]));
		double resx = 0;
		double resy = 0;
		for (int i=0; i<agents.size(); i++)
		{
			if (agents[i]->getType() != DEAD)
			{
				resx += agents[i]->vx;
				resy += agents[i]->vy;
				tot++;
			}
		}
		
		(*file) << hypot(resx, resy) / tot << endl; 
	}
};

class EfficiencySaver : public Saver
{
private:
	Environment* env;
	
public:
	EfficiencySaver(ofstream* f) : Saver(f) { };
	~EfficiencySaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = newEnv;
	}
	
	void exec()
	{
		int tot = 0;
		vector<FluidAgent*>& agents = *(static_cast< vector<FluidAgent*> *> (env->data["fagents"]));
		double res = 0;
		double base = 0;
		for (int i=0; i<agents.size(); i++)
		{
			if (agents[i]->getType() != DEAD)
			{
				res += abs(agents[i]->vortices[0] + agents[i]->vortices[1]);
				base = 0.5*(agents[0]->vortices[0] - agents[0]->vortices[1]);
				tot++;
			}
		}
		
		(*file) << res / (tot * base) << endl; 
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

class NNSaver : public Saver
{
private:
	SelfAvoidEnvironment* env;
	ANNApproximator* Q;
	
public:
	NNSaver(ofstream* f) : Saver(f) { };
	~NNSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = static_cast<SelfAvoidEnvironment*> (newEnv);
		Q   = static_cast<ANNApproximator*> (env->data["QSelfAvoider"]);
	}
	
	void exec()
	{
		ofstream& out(*file);
		
		vector<double> inp(6);
		vector<double> scaledInp(6);
		inp[0] = inp[1] = 0;
		inp[2] = 0.05;
		inp[3] = 360;
		vector<double> outp(3);
		
		int ni = 15;
		int nj = 10;
		for (int i=0; i<ni; i++)
		{
			for (int j=0; j<nj; j++)
			{
				inp[4] = -0.001 + (0.002 * i) / ni;
				inp[5] = (360.0 * j) / nj;
				
				for (int i=0; i<6; i++)
				{
					scaledInp[i] = inp[i] - env->sI.bottom[i];
					scaledInp[i] /= env->sI.top[i] - env->sI.bottom[i];
					scaledInp[i] *= 4;
					scaledInp[i] -= 2;
				}
				
				
				out << "(";
				for (int k=0; k<3; k++)
				{
					Q->ann[k]->predict(scaledInp, outp);
					out << std::setprecision(4) << outp[0];
					if (k<2) out << " ";
				}
				out << "),  ";
			}
			out << endl;
		}
		
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



