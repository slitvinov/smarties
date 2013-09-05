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
	Environment* env;
	
public:
	RewardSaver(ostream* f) : Saver(f) { };
	RewardSaver(string   f) : Saver(f) { };
	~RewardSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = newEnv;
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
	MomentumSaver(ostream* f) : Saver(f) { };
	MomentumSaver(string   f) : Saver(f) { };
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
	EfficiencySaver(ostream* f) : Saver(f) { };
	EfficiencySaver(string   f) : Saver(f) { };
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
				base = 0.5*(agents[i]->vortices[0] - agents[i]->vortices[1]);
				tot++;
			}
		}
		
		(*file) << res / (tot * base) << endl; 
	}
};

class DipolesSaver : public Saver
{
private:
	Environment* env;
	
public:
	DipolesSaver(ostream* f) : Saver(f) { };
	DipolesSaver(string   f) : Saver(f) { };
	~DipolesSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = newEnv;
	}
	
	void exec()
	{
		vector<CouzinDipole*>& agents = *(static_cast< vector<CouzinDipole*> *> (env->data["fagents"]));
		double t = *(static_cast<double*> (env->data["time"]));
		
		for (int i=0; i<agents.size(); i++)
		{	
			CouzinDipole* agent = agents[i];
			(*file) << t << " " << agent->x << " " << agent->y  << " 0 0 0 0 ";
			(*file) << agent->alpha << " " << agent->vortices[0] << " " << agent->vortices[1] << " ";
			(*file) << agent->l << " " << agent->IvI << endl;
		}
	}
};

class CouzinsSaver : public Saver
{
private:
	Environment* env;
	
public:
	CouzinsSaver(ostream* f) : Saver(f) { };
	CouzinsSaver(string   f) : Saver(f) { };
	~CouzinsSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = newEnv;
	}
	
	void exec()
	{
		vector<CouzinAgent*>& agents = *(static_cast< vector<CouzinAgent*> *> (env->data["couzins"]));
		double t = *(static_cast<double*> (env->data["time"]));
		
		for (int i=0; i<agents.size(); i++)
		{	
			CouzinAgent* agent = agents[i];
			(*file) << t << " " << agent->x << " " << agent->y  << " 0 0 0 0 ";
			(*file) << ((agent->getType() != DEAD) ? atan2(agent->vy, agent->vx) : -100.0) << " " << 0 << " " << 0 << " ";
			(*file) << agent->d / (2*sqrt(2*M_PI)) << " " << agent->IvI << endl;
		}
	}
};

class CollisionSaver : public Saver
{
private:
	Environment* env;
	
public:
	CollisionSaver(ostream* f) : Saver(f) { };
	CollisionSaver(string   f) : Saver(f) { };
	~CollisionSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = newEnv;
	}
	
	void exec()
	{
		vector<Agent*>& agents = *(static_cast< vector<Agent*> *> (env->data["agents"]));
		
		int collisions = 0;
		for (int i=0; i<agents.size(); i++)
		{	
			if (agents[i]->getType() == DEAD)
				collisions++;
		}
		
		(*file) << collisions/2 << endl;
	}
};


class StateSaver : public Saver
{
private:
	SelfAvoidEnvironment* env;
	MultiTable* Q;
	
public:
	StateSaver(ostream* f) : Saver(f) { };
	StateSaver(string   f) : Saver(f) { };
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
	NNSaver(ostream* f) : Saver(f) { };
	NNSaver(string   f) : Saver(f) { };
	~NNSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = static_cast<SelfAvoidEnvironment*> (newEnv);
		Q   = static_cast<ANNApproximator*> (env->data["ann"]);
	}
	
	void exec()
	{
		if (Q != NULL)
		{
			ofstream& out(*file);
			
			vector<double> inp(9,0);
			vector<double> outp(3);
			
			int ni = 10;
			int nj = 10;
			for (int i=0; i<ni; i++)
			{
				for (int j=0; j<nj; j++)
				{
					//double s = 0.4;
					inp[2] = -2 + (4.0*i) / (ni-1.0);
					//inp[1] = -2 + (4.0*j) / (nj-1.0);
					
					out << "(";
					for (int k=0; k<3; k++)
					{
						Q->ann[k]->predict(inp, outp);         // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111111111111111
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
		sprintf(buf, "%s%07d.tga", (folder+fname).c_str(), num);
		info("Saving screenshot to %s... ", buf);
		num++;
		if (gltWriteTGA(buf))
		{
			info("Ok\n");
		}
		else
		{
			warn("Failed!\n");
		}
#endif
	}
};



