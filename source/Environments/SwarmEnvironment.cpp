/*
 *  SwarmEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 09.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <unistd.h>
#include <cmath>
#include <complex>

#include "SwarmEnvironment.h"
#include "../Misc.h"

using namespace ErrorHandling;

SwarmEnvironment::SwarmEnvironment(vector<Agent*> agents, double rWall, StateType tp): Environment(agents), rWall(rWall)
{
    for (auto a : agents)
        swarmers.push_back(static_cast<SmartySwarmer*>(a));

    cells =  new Cells<SmartySwarmer>(swarmers, 8*swarmers[0]->d, -rWall, -rWall, rWall, rWall);
    getter = new CellsTraverser<SmartySwarmer>(cells);

    sI.type = tp;
    setDims();
    for (auto& s : swarmers)
    {
        s->setDims(sI, aI);
        s->setEnvironment(this);
    }

    const int n = swarmers.size();
    vortices.resize(n);
    vortCoos.resize(n);
    targets.resize(n);
    myfAgents.resize(n);
}

void SwarmEnvironment::setDims()
{
    double d = swarmers[0]->d;
    double v = swarmers[0]->IvI;

    int nNeigh = 2;

    sI.dim = 1 + 2*nNeigh;

    for (int i=0; i<nNeigh; i++)
    {
        // dist to neigh
        sI.bounds.push_back(20);
        sI.top.push_back(10*d);
        sI.bottom.push_back(0);
        sI.aboveTop.push_back(true);
        sI.belowBottom.push_back(true);

        // angle to neigh
        sI.bounds.push_back(20);
        sI.top.push_back(180);
        sI.bottom.push_back(-180);
        sI.aboveTop.push_back(false);
        sI.belowBottom.push_back(false);
    }

    sI.bounds.push_back(20);
    sI.top.push_back(180);
    sI.bottom.push_back(-180);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);

    aI.dim = 1;
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}

SmartySwarmer* SwarmEnvironment::findClosestNeighbour(SmartySwarmer* agent)
{
    double min = 1e10;
    double xj, yj;

    getter->prepare(cells->getObjId(agent));
    SmartySwarmer *n, *closest = NULL;
    while (getter->getNextXY(xj, yj, n))
    {
        double dst = _dist(agent->x, agent->y, xj, yj);
        if (dst < min)
        {
            n->physX = xj;
            n->physY = yj;
            min = dst;
            closest = n;
        }
    }
    return closest;
}

void SwarmEnvironment::findClosestNeighbours(vector<SmartySwarmer*>& res, SmartySwarmer* agent, double dist)
{
    res.clear();
    double xj, yj;

    getter->prepare(cells->getObjId(agent));
    SmartySwarmer *n;

    while (getter->getNextXY(xj, yj, n))
    {
        double dst = _dist(agent->x, agent->y, xj, yj);
        if (dst < dist)
        {
            n->physX = xj;
            n->physY = yj;
            res.push_back(n);
        }
    }
}

void SwarmEnvironment::computeVelocities()
{
    // Prepare vortices - pack strengths, coordinates and desired velocities into vectors
    int k=0;
    double minDist = 1e10;
    for (int n = 0; n < (int) swarmers.size(); n++)
    {
        SmartySwarmer* agent = swarmers[n];

        for (int j = 0; j < (int) agent->nVort; j++)
        {
            vortices[k] = agent->vortices[j];
            vortCoos[k].first  = agent->vortCoos[j].first;
            vortCoos[k].second = agent->vortCoos[j].second;
            targets[k]  = &(agent->vortVels[j]);
            myfAgents[k] = agent;
            k++;
        }

        for (int i = 1; i <= (int) agent->nVort; i++)
            for (int j = i+1; j <= (int) agent->nVort; j++)
            {
                double dist = _dist(vortCoos[k-i].first, vortCoos[k-i].second, vortCoos[k-j].first, vortCoos[k-j].second);
                if (minDist > dist) minDist = dist;
            }
    }

    // Solve n^2 problem for all vortices
    int tot = vortices.size();
    for (int n = 0; n < (int) targets.size(); n++)
    {
        complex<double> velocity(0, 0);

        double xn = vortCoos[n].first;
        double yn = vortCoos[n].second;

        for (int j = 0; j < tot; j++)
        {
            if (n == j) continue;

            double xj = vortCoos[j].first;
            double yj = vortCoos[j].second;
            double gammaj = vortices[j];
            double X = xn - xj;
            double Y = yn - yj;

            double dist2 = (X * X + Y * Y);
            //if (!immortal && dist2 < minDist * minDist / 4) myfAgents[n]->type = myfAgents[j]->type = DEAD;

            velocity += -gammaj * (complex<double>(Y, X)) / dist2 ;
        }

        velocity /= 2 * M_PI;

        *(targets[n]) = pair<double, double> (real(velocity), imag(velocity)); /// beware, these are conjugate velocities
    }
}


void SwarmEnvironment::calculateMomentum()
{
    int tot = 0;
    double resx = 0;
    double resy = 0;
    for (int i=0; i<swarmers.size(); i++)
    {
        if (swarmers[i]->getType() != DEAD)
        {
            resx += swarmers[i]->vx;
            resy += swarmers[i]->vy;
            tot++;
        }
    }

    momentum = hypot(resx, resy) / tot;
}

int SwarmEnvironment::evolve(double t)
{
    cells->migrate();
    computeVelocities();
    calculateMomentum();
    return 0;
}
