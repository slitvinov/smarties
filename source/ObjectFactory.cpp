/*
 *  ObjectFactory.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Settings.h"
#include "ObjectFactory.h"
#include "AllSystems.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <iostream>

using namespace ErrorHandling;
using namespace std;

inline string ObjectFactory::_parse(string source, string pattern, bool req)
{
    int pos = source.find(((string)" ")+pattern);
    if (pos == string::npos) {
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
    ifstream inFile;
    inFile.open(filename.c_str());
    Environment* env = nullptr;

    string envStr;
    getline(inFile, envStr);

    if (!inFile.good()) die("Unable to open file '%s'!\n", filename.c_str());
    {
        if (envStr.find("TwoFishEnvironment ") != envStr.npos)
        {
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            env = new TwoFishEnvironment(n, execpath, rank, settings);
        }
        else if (envStr.find("TwoActFishEnvironment ") != envStr.npos)
        {
            string execpath = _parse(envStr, "exec", true);
            int n  = _parseInt(envStr, "n", true);
            env = new TwoActFishEnvironment(n, execpath, rank, settings);
        }
        else if (envStr.find("NewFishEnvironment ") != envStr.npos)
        {
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            env = new NewFishEnvironment(n, execpath, rank, settings);
        }
        else if (envStr.find("CartEnvironment ") != envStr.npos)
        {
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            env = new CartEnvironment(n, execpath, rank, settings);
        }
        else if (envStr.find("CMAEnvironment ") != envStr.npos)
        {
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            env = new CMAEnvironment(n, execpath, rank, settings);
        }
        else if (envStr.find("Environment ") != envStr.npos)
        {
            string execpath = _parse(envStr, "exec", true);
            int n = _parseInt(envStr, "n", true);
            env = new CartEnvironment(n, execpath, rank, settings);
        }
        else die("Unsupported environment type in line %s\n", envStr.c_str());

        //getline(inFile, s); used to be a while loop, but env vectors are not supported...
    }
    assert(env not_eq nullptr);
    env->setDims();
    if (rank != 0) env->setup_Comm();

    return env;
}
