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
    Real momentum, rWall;
    void calculateMomentum();

    vector<Real> vortices;
    vector<pair<Real, Real> > vortCoos;
    vector<pair<Real, Real>* > targets;
    vector<SmartySwarmer*> myfAgents;

public:
    vector<SmartySwarmer*>   swarmers;
    Cells <SmartySwarmer>*   cells;
    CellsTraverser<SmartySwarmer>* getter;

    SwarmEnvironment(vector<Agent*> newAgents, Real rWall, StateType tp);

    void findClosestNeighbours(vector<SmartySwarmer*>& res, SmartySwarmer* agent, Real dist);
    SmartySwarmer* findClosestNeighbour(SmartySwarmer* agent);
    void computeVelocities();
    int evolve(Real t);

    Real getMomentum()
    {
        return momentum;
    }
};
