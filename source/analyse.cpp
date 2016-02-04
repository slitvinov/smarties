/*
 *  analyse.cpp
 *  smarties
 *
 *  Created by Dmitry Alexeev on Jun 8, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "QApproximators/QApproximator.h"
#include "QApproximators/MultiTable.h"
#include "StateAction.h"

int ErrorHandling::debugLvl;

int main()
{
    StateInfo sI;
    ActionInfo aI;

    sI.dim = 4;
    // State: coordinate...
    sI.bounds.push_back(20);
    sI.top.push_back(1);
    sI.bottom.push_back(-1);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    // ...velocity...
    sI.bounds.push_back(20);
    sI.top.push_back(2);
    sI.bottom.push_back(-2);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    // ...angle...
    sI.bounds.push_back(100);
    sI.top.push_back(1);
    sI.bottom.push_back(-1);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    // ...and angular velocity
    sI.bounds.push_back(20);
    sI.top.push_back(4);
    sI.bottom.push_back(-4);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    aI.dim = 1;
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(5);

    QApproximator* q = new MultiTable(sI, aI);

    q->restart("res/policy");

    int ind[4];
    for (ind[2] = -1; ind[2] < sI.bounds[2]+1; ind[2]++)
    {
        int res[5] = {0, 0, 0, 0, 0};

        for (ind[1] = -1; ind[1] < sI.bounds[1]+1; ind[1]++)
            for (ind[0] = -1; ind[0] < sI.bounds[0]+1; ind[0]++)
                for (ind[3] = -1; ind[3] < sI.bounds[3]+1; ind[3]++)
                {
                    State s(sI);
                    for (int k = 0; k<sI.dim; k++)
                        s.vals[k] = sI.bottom[k] + (ind[k]+0.5)/sI.bounds[k] * (sI.top[k] - sI.bottom[k]);

                    Real best = -1e10;
                    ActionIterator actionsIt(aI);
                    actionsIt.reset();
                    while (!actionsIt.done())
                    {
                        Real val;
                        if ((val = q->get(s, actionsIt.next())) > best && fabs(val) > 1e-9)
                        {
                            best = val;
                            actionsIt.memorize();

                            //printf("     %d %d %d %d\n", ind[0], ind[1], ind[2], ind[3]);
                        }
                    }

                    if (best > -1e9)
                        res[ actionsIt.recall().vals[0] ]++;
                }

        printf("angle:  %f,  actions: %d %d %d %d %d\n", sI.bottom[2] + (ind[2]+0.5)/sI.bounds[2] * (sI.top[2] - sI.bottom[2]),
                res[0], res[1], res[2], res[3], res[4]);
    }

    return 0;
}


