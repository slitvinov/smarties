/*
 *  SwarmEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 09.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "Environment.h"
#include "CellList.h"

class SmartySwarmer;
#include "../Agents/SmartySwarmer.h"

class SwarmEnvironment: public Environment
{
public:
    void setDims();
    double momentum, rWall;
    void calculateMomentum();

    vector<double> vortices;
    vector<pair<double, double> > vortCoos;
    vector<pair<double, double>* > targets;
    vector<SmartySwarmer*> myfAgents;

public:
    vector<SmartySwarmer*>   swarmers;
    Cells <SmartySwarmer>*   cells;
    CellsTraverser<SmartySwarmer>* getter;

    SwarmEnvironment(vector<Agent*> newAgents, double rWall, StateType tp);

    void findClosestNeighbours(vector<SmartySwarmer*>& res, SmartySwarmer* agent, double dist);
    SmartySwarmer* findClosestNeighbour(SmartySwarmer* agent);
    void computeVelocities();
    int evolve(double t);

    double getMomentum()
    {
        return momentum;
    }
};
