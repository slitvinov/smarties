/*
 *  ObjectFactory.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

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
    size_t pos = source.find(((string)" ")+pattern);
    if (pos == string::npos) {
        if (req)
            _die("Parse factory file failed: required argument '%s' line '%s'\n",
                pattern.c_str(), source.c_str())
        else     return "";
    }

    pos += pattern.length()+1;
    while (source[pos] == ' ') pos++;
    if (source[pos] != '=')
          _die("Parse factory file failed: argument '%s' line '%s'\n",
              pattern.c_str(), source.c_str())
    while (source[pos] == ' ') pos++;

    pos++;
    size_t stpos = pos;
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

Environment* ObjectFactory::createEnvironment(int _rank, int index)
{
    assert(_rank>=0);
    Uint rank = static_cast<Uint>(_rank);
    ifstream inFile;
    inFile.open(filename.c_str());
    Environment* env = nullptr;

    string envStr;
    getline(inFile, envStr);

    if (!inFile.good())
      _die("Unable to open file '%s'!\n", filename.c_str());

    const string execpath = _parse(envStr, "exec", true);
    int _n = _parseInt(envStr, "n", true);
    if(_n<=0) die("Factory file requested environment without agents\n");
    const Uint n = static_cast<Uint>(_n);

    if (envStr.find("TwoFishEnvironment ") != envStr.npos) {
        printf("TwoFishEnvironment with %u agents per slave.\n",n);
        env = new TwoFishEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("TwoActFishEnvironment ") != envStr.npos) {
        printf("TwoActFishEnvironment with %u agents per slave.\n",n);
        env = new TwoActFishEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("NewFishEnvironment ") != envStr.npos)  {
        printf("NewFishEnvironment with %u agents per slave.\n",n);
        env = new NewFishEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("DeadFishEnvironment ") != envStr.npos) {
        printf("DeadFishEnvironment with %u agents per slave.\n",n);
        env = new DeadFishEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("AcrobotEnvironment ") != envStr.npos) {
        printf("AcrobotEnvironment with %u agents per slave.\n",n);
        env = new AcrobotEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("GliderEnvironment ") != envStr.npos) {
        printf("GliderEnvironment with %u agents per slave.\n",n);
        env = new GliderEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("CartEnvironment ") != envStr.npos) {
        printf("CartEnvironment with %u agents per slave.\n",n);
        env = new CartEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("CMAEnvironment ") != envStr.npos) {
        printf("CMAEnvironment with %u agents per slave.\n",n);
        env = new CMAEnvironment(n, execpath, rank, *settings);
    }
    else if (envStr.find("alebotEnvironment ") != envStr.npos) {
        printf("alebotEnvironment with %u agents per slave.\n",n);
        int _nactions = _parseInt(envStr, "nActions", true);
        if(_nactions<=0) die("Factory file requested environment without actions\n");
        const Uint nactions = static_cast<Uint>(_nactions);
        env = new alebotEnvironment(n, nactions, execpath, rank, *settings);
    }
    else if (envStr.find("TestEnvironment ") != envStr.npos) {
        printf("TestEnvironment with %u agents per slave.\n",n);
        env = new TestEnvironment(1, execpath, rank, *settings);
    }
    else _die("Unsupported environment type in line %s\n", envStr.c_str());

    //getline(inFile, s); used to be a while loop, but env vectors are not supported...

    if(env == nullptr) die("Env cannot be nullptr\n");
    env->setDims();

    return env;
}
