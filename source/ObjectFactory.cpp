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
#include <iostream>

#include "rng.h"
#include "Settings.h"
#include "ObjectFactory.h"
#include "ErrorHandling.h"
#include "AllSystems.h"

using namespace ErrorHandling;
using namespace std;

inline string ObjectFactory::_parse(string source, string pattern, bool req)
{
    int pos = source.find(((string)" ")+pattern);
    if (pos == string::npos)
    {
        if (req) die("Parsing factory file failed at required argument '%s' line '%s'\n", pattern.c_str(), source.c_str());
        else     return "";
    }

    pos += pattern.length()+1;
    while (source[pos] == ' ') pos++;
    if (source[pos] != '=')  die("Parsing factory file failed at argument '%s' line '%s'\n", pattern.c_str(), source.c_str());
    while (source[pos] == ' ') pos++;

    pos++;
    int stpos = pos;
    while (source[pos] != ' ' && pos < source.length()) pos++;

    return source.substr(stpos, pos - stpos);
}

inline int ObjectFactory::_parseInt(string source, string pattern, bool req)
{
    return atoi(_parse(source, pattern, req).c_str());
}

inline double ObjectFactory::_parseDouble(string source, string pattern, bool req)
{
    return atof(_parse(source, pattern, req).c_str());
}

Environment* ObjectFactory::createEnvironment(int rank, int index)
{
    RNG rng(rand());

    ifstream inFile;
    inFile.open(filename.c_str());
    Environment* env;

    if (!inFile.good()) die("Unable to open file '%s'!\n", filename.c_str());

    while (inFile.good())
    {
        string envStr, name;

        getline(inFile, envStr);
        string s;

        if (envStr.find("ExternalEnvironment ") != envStr.npos)
        {
            vector<Agent*> agents;

            string appType = _parse(envStr, "type", false);
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            
            StateType st;
            if (appType == "DISCR") st = DISCR;
            else if (appType == "ANN") st = ANN;
            else if (appType == "WAVE") st = WAVE;

            for (int i=0; i<n; i++)
            {
                agents.push_back(new ExternalAgent(1e-10, ACTOR, "ExternalAgent"));
            }

            env = new ExternalEnvironment(agents, execpath, st, rank, index);
            getline(inFile, s);
        }

        else if (envStr.find("SelfAvoidEnvironment ") != envStr.npos)
        {
            vector<Agent*> agents;
            vector<Column> columns;

            string appType = _parse(envStr, "type", false);
            double D = _parseDouble(envStr, "scale");
            double rWall = _parseDouble(envStr, "rWall")*D;

            StateType st;
            if (appType == "DISCR") st = DISCR;
            else if (appType == "ANN") st = ANN;
            else if (appType == "WAVE") st = WAVE;

            while (inFile.good())
            {
                getline(inFile, s);

                if (s.find("SmartySelfAvoider ") != s.npos)
                {
                    SmartySelfAvoider* agent = new SmartySelfAvoider(_parseDouble(s, "x")*D,
                            _parseDouble(s, "y")*D, _parseDouble(s, "d")*D,
                            _parseDouble(s, "T"), D/5.0, 0);
                    agents.push_back(agent);
                }
                else if( s.find("SmartySelfAvoiders ") != s.npos )
                {
                    double d = _parseDouble(s, "d")*D;
                    double T = _parseDouble(s, "T");
                    int num  = _parseInt   (s, "n");

                    for(int j=0; j<num; j++)
                    {
                        const double radius = rng.uniform(0.0, 0.6*rWall);
                        const double angle  = rng.uniform(0.0,2*M_PI);
                        const double xx     = radius*cos(angle);
                        const double yy     = radius*sin(angle);

                        SmartySelfAvoider* agent = new SmartySelfAvoider(xx, yy, d, T, 0.01*rWall, 0);
                        agents.push_back(agent);
                    }
                }
                else if( s.find("Column ") != s.npos )
                {
                    columns.push_back(make_tuple(_parseDouble(s, "x")*D, _parseDouble(s, "y")*D, _parseDouble(s, "d")*D));
                }
                else if( s.find("Columns ") != s.npos )
                {
                    double dmin = _parseDouble(s, "dmin")*D;
                    double dmax = _parseDouble(s, "dmax")*D;
                    int  num = _parseInt   (s, "n");

                    for(int j=0; j<num; j++)
                    {
                        const double radius = rng.uniform(0.0,0.45);
                        const double angle  = rng.uniform(0.0,2*M_PI);
                        const double xx     = radius*cos(angle);
                        const double yy     = radius*sin(angle);
                        const double d      = rng.uniform(dmin, dmax);

                        columns.push_back(make_tuple(xx, yy, d));
                    }
                }
                else die("Couldn't parse line '%s' for SelfAvoidEnvironment\n", s.c_str());
            }

            env = new SelfAvoidEnvironment(agents, columns, rWall, st);
        }
        else die("Unsupported environment type in line %s\n", envStr.c_str());
    }

    return env;
}
