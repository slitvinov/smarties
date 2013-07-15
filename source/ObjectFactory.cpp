/*
 *  ObjectFactory.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <string>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "rng.h"
#include "Settings.h"
#include "ObjectFactory.h"
#include "ErrorHandling.h"
#include "AllSystems.h"
#include "PF_SwarmCylinders.h"

using namespace ErrorHandling;
using namespace std;

inline string ObjectFactory::_parse(string source, string pattern)
{
	int pos = source.find(pattern);
	if (pos == string::npos) die("Parsing factory file failed at required argument '%s'\n", pattern.c_str());
	
	pos += pattern.length();
	while (source[pos] == ' ') pos++;
	if (source[pos] != '=')  die("Parsing factory file failed at argument '%s'\n", pattern.c_str());
	while (source[pos] == ' ') pos++;
	
	pos++;
	int stpos = pos;
	while (source[pos] != ' ' && pos < source.length()) pos++;
	
	return source.substr(stpos, pos - stpos);
}
	
inline int ObjectFactory::_parseInt(string source, string pattern)
{
	return atoi(_parse(source, pattern).c_str());
}

inline double ObjectFactory::_parseDouble(string source, string pattern)
{
	return atof(_parse(source, pattern).c_str());
}

System ObjectFactory::getAgentVector()
{
	if  (settings.randSeed == -1 )  srand(time(NULL));
	else							srand(settings.randSeed);
	RNG rng(rand());
	
	double D = settings.scale;
	
	ifstream inFile;
	inFile.open(filename.c_str());
	
	if (!inFile.good()) die("Unable to open file '%s'!\n", filename.c_str());
	
	System system;
	
	while (inFile.good())
	{
		string name, envName;
		
		inFile >> envName;
		
		while (true)
		{
			inFile >> name;
			
			if (name == "CircularWall")
			{
				Agent* object = new CircularWall(settings.centerX, settings.centerY, 1.0);
				system.agents.push_back(object);
			}
			
			else if( name == "SmartyDodger" )
			{
				string s;
				getline(inFile, s);
				
				Agent* object = new SmartyDodger(_parseDouble(s, "xm")*D + settings.centerX,
												 _parseDouble(s, "ym")*D + settings.centerY,
												 _parseDouble(s, "d") *D,  _parseDouble(s, "T"), D/5.0, 0);
				
				system.agents.push_back(object);
			}
			
			else if( name == "SmartyDodgers" )
			{
				string s;
				getline(inFile, s);

				double d = _parseDouble(s, "d")*D;
				double T = _parseDouble(s, "T");
				int num  = _parseInt   (s, "n");
				
				for(int j=0; j<num; j++)
				{
					const double radius = rng.uniform(0.0,0.3);
					const double angle  = rng.uniform(0.0,2*M_PI);
					const double xx     = radius*cos(angle);
					const double yy     = radius*sin(angle);
					
					Agent* object = new SmartyDodger(xx+settings.centerX, yy+settings.centerY, d, T, D/5.0, 0);
					system.agents.push_back(object);
				}
			}
			
			else if( name == "SmartySelfAvoider" )
			{
				string s;
				getline(inFile, s);
				
				Agent* object = new SmartySelfAvoider(_parseDouble(s, "xm")*D + settings.centerX,
												      _parseDouble(s, "ym")*D + settings.centerY,
												      _parseDouble(s, "d") *D,  _parseDouble(s, "T"), D/5.0, 0);
				
				system.agents.push_back(object);
			}
			
			else if( name == "SmartySelfAvoiders" )
			{
				string s;
				getline(inFile, s);
				
				double d = _parseDouble(s, "d")*D;
				double T = _parseDouble(s, "T");
				int num  = _parseInt   (s, "n");
				
				for(int j=0; j<num; j++)
				{
					const double radius = rng.uniform(0.0,0.49-d);
					const double angle  = rng.uniform(0.0,2*M_PI);
					const double xx     = radius*cos(angle);
					const double yy     = radius*sin(angle);
					
					Agent* object = new SmartySelfAvoider(xx+settings.centerX, yy+settings.centerY, d, T, D/5.0, 0);
					system.agents.push_back(object);
				}
			}			
			
			else if( name == "DynamicColumn" )
			{
				string s;
				getline(inFile, s);

				Agent* object = new DynamicColumn(_parseDouble(s, "xm")*D + settings.centerX,
												  _parseDouble(s, "ym")*D + settings.centerY,
												  _parseDouble(s, "d") *D);
				system.agents.push_back(object);
			}
			
			else if( name == "DynamicColumns" )
			{
				string s;
				getline(inFile, s);
				
				double d = _parseDouble(s, "d")*D;
				int  num = _parseInt   (s, "n");
				
				for(int j=0; j<num; j++)
				{
					const double radius = rng.uniform(0.0,0.45);
					const double angle  = rng.uniform(0.0,2*M_PI);
					const double xx     = radius*cos(angle);
					const double yy     = radius*sin(angle);
					
					Agent* object = new DynamicColumn(xx+settings.centerY, yy+settings.centerY, d);
					system.agents.push_back(object);
				}
			}
			
			else if( name == "CouzinAgent" )
			{
				string s;
				getline(inFile, s);
				
				Agent* object = new CouzinAgent(_parseDouble(s, "xm") *D + settings.centerX,
												_parseDouble(s, "ym") *D + settings.centerY,
												_parseDouble(s, "d")  *D, _parseDouble(s, "T"), 1,
												_parseDouble(s, "zor")*D, _parseDouble(s, "zoo") *D,
												_parseDouble(s, "zoa")*D, _parseDouble(s, "ang"),
												_parseDouble(s, "tr"),    _parseDouble(s, "v")*D, 0, &rng);
				
				system.agents.push_back(object);
			}
			
			else if( name == "CouzinAgents" )
			{
				string s;
				getline(inFile, s);
				
				double d = _parseDouble(s, "d")*D;
				double T = _parseDouble(s, "T");
				int num  = _parseInt   (s, "n");
				
				double zor = _parseDouble(s, "zor")*D;
				double zoo = _parseDouble(s, "zoo")*D;
				double zoa = _parseDouble(s, "zoa")*D;
				double ang = _parseDouble(s, "ang");
				double tr  = _parseDouble(s, "tr");
				double v   = _parseDouble(s, "v")*D;
				
				for(int j=0; j<num; j++)
				{
					const double radius = rng.uniform(0.0,0.3);
					const double angle  = rng.uniform(0.0,2*M_PI);
					const double xx     = radius*cos(angle);
					const double yy     = radius*sin(angle);
					
					Agent* object = new CouzinAgent(xx+settings.centerX, yy+settings.centerY, d, T,
													1, zor, zoo, zoa, ang, tr, v, 0, &rng);
					system.agents.push_back(object);
				}
			}			
			
			
			else if( name == "CouzinDipole" )
			{
				string s;
				getline(inFile, s);
				
				Agent* object = new CouzinDipole(_parseDouble(s, "xm") *D + settings.centerX,
												_parseDouble(s, "ym") *D + settings.centerY,
												_parseDouble(s, "d")  *D, _parseDouble(s, "T"), 1,
												_parseDouble(s, "zor")*D, _parseDouble(s, "zoo") *D,
												_parseDouble(s, "zoa")*D, _parseDouble(s, "ang"),
												_parseDouble(s, "tr"),    _parseDouble(s, "v")*D, 0, &rng);
				
				system.agents.push_back(object);
			}
			
			else if( name == "CouzinDipoles" )
			{
				string s;
				getline(inFile, s);
				
				double d = _parseDouble(s, "d")*D;
				double T = _parseDouble(s, "T");
				int num  = _parseInt   (s, "n");
				
				double zor = _parseDouble(s, "zor")*D;
				double zoo = _parseDouble(s, "zoo")*D;
				double zoa = _parseDouble(s, "zoa")*D;
				double ang = _parseDouble(s, "ang");
				double tr  = _parseDouble(s, "tr");
				double v   = _parseDouble(s, "v")*D;
				
				for(int j=0; j<num; j++)
				{
					const double radius = rng.uniform(0.0,0.3);
					const double angle  = rng.uniform(0.0,2*M_PI);
					const double xx     = radius*cos(angle);
					const double yy     = radius*sin(angle);
					
					Agent* object = new CouzinDipole(xx+settings.centerX, yy+settings.centerY, d, T,
													 1, zor, zoo, zoa, ang, tr, v, 0, &rng);
					system.agents.push_back(object);
				}
			}
			
			else if (name == "CouzinSwarm" || name == "CouzinDipoleSwarm")
			{
				string s;
				getline(inFile, s);
				
				double d = _parseDouble(s, "d")*D;
				double T = _parseDouble(s, "T");
				int num  = _parseInt   (s, "n");
				
				double zor = _parseDouble(s, "zor")*D;
				double zoo = _parseDouble(s, "zoo")*D;
				double zoa = _parseDouble(s, "zoa")*D;
				double ang = _parseDouble(s, "ang");
				double tr  = _parseDouble(s, "tr");
				double v   = _parseDouble(s, "v")*D;

				SwarmCylinders swarm(num, 0.5 * d, 0.05, 1.0, 1.0, 1.0, M_PI / 2);
		
				swarm.placeCylinders();
				//swarm.printSpline();
				swarm.findEquilibrium();
				swarm.rotateClockwiseQuarterTurn();
				
				/// CALCULATE EQUILIBRIUM POSITIONS
				vector<double> xs;
				vector<double> ys;
				
				xs = swarm.getX();
				ys = swarm.getY();
				
				/// SCALE DIPOLES AND MOVE DIPOLES
				double const areaPerDipole = d * d * 4;
				double const totalSwarmArea = areaPerDipole * num;
				double const scaleRatio = sqrt(totalSwarmArea);
				
				double const yMin = *(std::min_element(ys.begin(), ys.end()));
				
				for (int i = 0; i < (int) xs.size(); i++)
				{
					xs[i] = xs[i] * scaleRatio;
					ys[i] = (ys[i] - yMin) * scaleRatio;
				}
				
				for(int j=0; j<num; j++)
				{
					const double xx     = xs[j];
					const double yy     = ys[j];
					
					Agent* object;
					if (name == "CouzinSwarm")
						object = new CouzinAgent(xx+settings.centerX, yy+settings.centerY, d, T,
												 1, zor, zoo, zoa, ang, tr, v, 0, &rng);
					else
						object = new CouzinDipole(xx+settings.centerX, yy+settings.centerY, d, T,
												  1, zor, zoo, zoa, ang, tr, v, 0, &rng);
					
					system.agents.push_back(object);
				}
				
				
			}
			
			else if (name == "END")
			{
				if      (envName == "DodgerEnvironment")       system.env = new DodgerEnvironment       (system.agents);
				else if (envName == "SelfAvoidEnvironment")    system.env = new SelfAvoidEnvironment    (system.agents);
				else if (envName == "CouzinEnvironment")       system.env = new CouzinEnvironment       (system.agents);
				else if (envName == "CouzinDipoleEnvironment") system.env = new CouzinDipoleEnvironment (system.agents);

				else die("Unsupported environment type %s\n", envName.c_str());
				break;
			}
			
			else die("Unsopported agent type: '%s'\n", name.c_str());
		}
	}
	
	return system;
}
