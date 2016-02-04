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

inline Real ObjectFactory::_parseReal(string source, string pattern, bool req)
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
        
        else if (envStr.find("CartEnvironment ") != envStr.npos)
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
                agents.push_back(new CartAgent(1e-10, ACTOR, "CartAgent"));
            }
            
            env = new CartEnvironment(agents, execpath, st, rank, index);
            getline(inFile, s);
        }
        
        else if (envStr.find("HardEnvironment ") != envStr.npos)
        {
            vector<Agent*> agents;
            printf("Created the correct cart??\n");
            string appType = _parse(envStr, "type", false);
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            
            StateType st;
            if (appType == "DISCR") st = DISCR;
            else if (appType == "ANN") st = ANN;
            else if (appType == "WAVE") st = WAVE;
            
            for (int i=0; i<n; i++)
            {
                agents.push_back(new HardCartAgent(1e-10, ACTOR, "HardCartAgent"));
            }
            
            env = new HardCartEnvironment(agents, execpath, st, rank, index);
            getline(inFile, s);
        }
        
        else if (envStr.find("GlideEnvironment ") != envStr.npos)
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
                agents.push_back(new CartAgent(1e-10, ACTOR, "CartAgent"));
            }
            
            env = new GlideEnvironment(agents, execpath, st, rank, index);
            getline(inFile, s);
        }

        else if (envStr.find("SelfAvoidEnvironment ") != envStr.npos)
        {
            vector<Agent*> agents;
            vector<Column> columns;

            string appType = _parse(envStr, "type", false);
            Real D = _parseReal(envStr, "scale");
            Real rWall = _parseReal(envStr, "rWall")*D;

            StateType st;
            if (appType == "DISCR") st = DISCR;
            else if (appType == "ANN") st = ANN;
            else if (appType == "WAVE") st = WAVE;

            while (inFile.good())
            {
                getline(inFile, s);

                if (s.find("SmartySelfAvoider ") != s.npos)
                {
                    SmartySelfAvoider* agent = new SmartySelfAvoider(_parseReal(s, "x")*D,
                            _parseReal(s, "y")*D, _parseReal(s, "d")*D,
                            _parseReal(s, "T"), D/5.0, 0);
                    agents.push_back(agent);
                }
                else if( s.find("SmartySelfAvoiders ") != s.npos )
                {
                    Real d = _parseReal(s, "d")*D;
                    Real T = _parseReal(s, "T");
                    int num  = _parseInt   (s, "n");

                    for(int j=0; j<num; j++)
                    {
                        const Real radius = rng.uniform(0.0, 0.6*rWall);
                        const Real angle  = rng.uniform(0.0,2*M_PI);
                        const Real xx     = radius*cos(angle);
                        const Real yy     = radius*sin(angle);

                        SmartySelfAvoider* agent = new SmartySelfAvoider(xx, yy, d, T, 0.01*rWall, 0);
                        agents.push_back(agent);
                    }
                }
                else if( s.find("Column ") != s.npos )
                {
                    columns.push_back(make_tuple(_parseReal(s, "x")*D, _parseReal(s, "y")*D, _parseReal(s, "d")*D));
                }
                else if( s.find("Columns ") != s.npos )
                {
                    Real dmin = _parseReal(s, "dmin")*D;
                    Real dmax = _parseReal(s, "dmax")*D;
                    int  num = _parseInt   (s, "n");

                    for(int j=0; j<num; j++)
                    {
                        const Real radius = rng.uniform(0.0,0.45);
                        const Real angle  = rng.uniform(0.0,2*M_PI);
                        const Real xx     = radius*cos(angle);
                        const Real yy     = radius*sin(angle);
                        const Real d      = rng.uniform(dmin, dmax);

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
