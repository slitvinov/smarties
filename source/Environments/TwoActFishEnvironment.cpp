/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "TwoActFishEnvironment.h"

using namespace std;

TwoActFishEnvironment::TwoActFishEnvironment(const int nAgents, const string execpath, const int _rank, Settings & settings) :
Environment(nAgents, execpath, _rank, settings),
sight(settings.senses==0 || settings.senses==4), POV(settings.senses==1),
l_line(settings.senses==2), p_sensors(settings.senses==3 || settings.senses==4),
study(settings.rewardType), goalDY((settings.goalDY>1.)? 1.-settings.goalDY : settings.goalDY)
{
}

void TwoActFishEnvironment::setDims()
{
    {
        sI.bounds.clear(); sI.top.clear(); sI.bottom.clear(); sI.isLabel.clear(); sI.inUse.clear();
        {
            // State: Horizontal distance from goal point...
            sI.inUse.push_back(sight);
            // ...vertical distance...
            sI.inUse.push_back(sight);
            // ...inclination of1the fish...
            sI.inUse.push_back(sight);
            // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
            sI.inUse.push_back(true);
            // ...last action (HAX!)
            sI.inUse.push_back(true);
            // ...second last action (HAX!)
            sI.inUse.push_back(true); //if l_line i have curvature info
        }
        {
            //Dist 6
            sI.inUse.push_back(true);
            
            //Quad 7
            sI.inUse.push_back(true);
            
            // VxAvg 8
            sI.inUse.push_back(true);
            
            // VyAvg 9
            sI.inUse.push_back(true);
            
            // AvAvg 10
            sI.inUse.push_back(true);
        }
#if 1
            //Xabs 6
            sI.inUse.push_back(false);

            //Yabs 7
            sI.inUse.push_back(false);
#endif
        {
            //Dist 6
            sI.inUse.push_back(true);

            //Quad 7
            sI.inUse.push_back(true);

            // VxAvg 8
            sI.inUse.push_back(true);
            
            // VyAvg 9
            sI.inUse.push_back(true);
            
            // AvAvg 10
            sI.inUse.push_back(true);
        }
        {
            //Pout 11
            sI.inUse.push_back(false);
            
            //defPower 12
            sI.inUse.push_back(false);
            
            // EffPDef 13
            sI.inUse.push_back(false);
            
            // PoutBnd 14
            sI.inUse.push_back(false);
            
            // defPowerBnd 15
            sI.inUse.push_back(false);
            
            // EffPDefBnd 16
            sI.inUse.push_back(false);
            
            // Pthrust 17
            sI.inUse.push_back(false);
            
            // Pdrag 18
            sI.inUse.push_back(false);
            
            // ToD 19
            sI.inUse.push_back(false);
        }
        const int nSensors = 20;
        for (int i=0; i<nSensors; i++) {
            // (VelNAbove  ) x 5 [20]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (VelTAbove  ) x 5 [25]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (VelNBelow  ) x 5 [30]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (VelTBelow  ) x 5 [35]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (FPAbove  ) x 5 [40]
            sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<nSensors; i++) {
            // (FVAbove  ) x 5 [45]
            sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<nSensors; i++) {
            // (FPBelow  ) x 5 [50]
            sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<nSensors; i++) {
            // (FVBelow ) x 5 [55]
            sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<2*nSensors; i++) {
            // (FVBelow ) x 5 [55]
            sI.inUse.push_back(p_sensors);
        }
        /*
        sI.values.push_back(-.50);
        sI.values.push_back(-.25);
        sI.values.push_back(0.00);
        sI.values.push_back(0.25);
        sI.values.push_back(0.50);
         */
    }
    {
        aI.dim = 2;
        aI.values.resize(aI.dim);
        //curavture
        aI.bounds.push_back(5); //Number of possible actions to choose from
        aI.upperBounds.push_back(0.5);
        aI.lowerBounds.push_back(-.5);
        aI.values[0].push_back(-.50);
        aI.values[0].push_back(-.25);
        aI.values[0].push_back(0.00);
        aI.values[0].push_back(0.25);
        aI.values[0].push_back(0.50);
        //period:
        aI.bounds.push_back(5); //Number of possible actions to choose from
        aI.upperBounds.push_back(0.25);
        aI.lowerBounds.push_back(-.25);
        aI.values[1].push_back(-.25);
        aI.values[1].push_back(-.125);
        aI.values[1].push_back(0.00);
        aI.values[1].push_back(0.125);
        aI.values[1].push_back(0.250);
    }
    resetAll=true;
    commonSetup();
}

void TwoActFishEnvironment::setAction(const int & iAgent)
{
    if ( agents[iAgent]->a->vals[0] >0.75 ) {
    	printf("Action 0 is too large (>0), reassigned at random to prevent sim from crashing\n");
        std::normal_distribution<Real> dist(0.5,0.25);
        const Real uB = 0.75; const Real lB = -.75;
        agents[iAgent]->a->vals[0]=lB+.5*(std::tanh(dist(*g))+1.)*(uB-lB);
    }
    if ( agents[iAgent]->a->vals[0] <-.75 ) {
    	printf("Action 0 is too large (<0), reassigned at random to prevent sim from crashing\n");
        std::normal_distribution<Real> dist(-.5,0.25);
        const Real uB = 0.75; const Real lB = -.75;
        agents[iAgent]->a->vals[0]=lB+.5*(std::tanh(dist(*g))+1.)*(uB-lB);
    }
    if ( agents[iAgent]->a->vals[1] >0.25 ) {
    	printf("Action 1 is too large (>0), reassigned at random to prevent sim from crashing\n");
        std::normal_distribution<Real> dist(0.5,0.25);
        const Real uB = 0.25; const Real lB = -.25;
        agents[iAgent]->a->vals[0]=lB+.5*(std::tanh(dist(*g))+1.)*(uB-lB);
    }
    if ( agents[iAgent]->a->vals[1] <-.25 ) {
    	printf("Action 1 is too large (<0), reassigned at random to prevent sim from crashing\n");
        std::normal_distribution<Real> dist(-.5,0.25);
        const Real uB = 0.25; const Real lB = -.25;
        agents[iAgent]->a->vals[0]=lB+.5*(std::tanh(dist(*g))+1.)*(uB-lB);
    }
    
    for (int i=0; i<aI.dim; i++)
        dataout[i] = (double) agents[iAgent]->a->vals[i];
    
    send_all(sock, dataout, sizeout);
}

int TwoActFishEnvironment::getState(int & iAgent)
{
    int bStatus = 0;
    //printf("RECEIVING %d,%d\n",sock,sizein);
    if ((bytes = recv_all(sock, datain, sizein)) <= 0) {
        if (bytes == 0) printf("socket %d hung up\n", sock);
        else perror("(1) recv");
        
        close(sock);
        bStatus = -1;
    } else { // (bytes == nbyte)
        iAgent  = *((int*)  datain   );
        bStatus = *((int*) (datain+1)); //first (==1?), terminal (==2?), etc
        //printf("Receiving from agent %d %d: ", iAgent, bStatus);
        
        std::swap(agents[iAgent]->s,agents[iAgent]->sOld);
        
        int k = 2;
        for (int j=0; j<sI.dim; j++) {
            //printf(" %f (%d)",datain[k],k);
            agents[iAgent]->s->vals[j] = (Real) datain[k++];
            assert(not std::isnan(agents[iAgent]->s->vals[j]) && not std::isinf(agents[iAgent]->s->vals[j]));
            if (j>=187) { //sight sensors get non-dimensionalized differently depending on size of fish if no obstacle is found =(
                agents[iAgent]->s->vals[j] = min(agents[iAgent]->s->vals[j], 5.);
            }
        }
        
        //printf(" %f (%d)\n",datain[k],k);
        agents[iAgent]->r = (Real) datain[k++];
        assert(not std::isnan(agents[iAgent]->r) && not std::isinf(agents[iAgent]->r));
        //printf("Got from child %d: reward %f initial state %s\n", rank, agents[iAgent]->r, agents[iAgent]->s->print().c_str()); fflush(0);
    }
    fflush(0);
    return bStatus;
}

bool TwoActFishEnvironment::pickReward(const State & t_sO, const Action & t_a,
                                    const State & t_sN, Real & reward, const int info)
{/*
    if (fabs(t_sN.vals[5] -t_sO.vals[4])>0.001) {
        printf("Mismatch new and old state!!! %s === %s\n",t_sO.print().c_str(),t_sN.print().c_str());
        abort();
    }
    if (fabs(t_sN.vals[4] -t_a.vals[0])>0.001) {
        printf("Mismatch state and action!!! %s === %s\n",t_sN.print().c_str(),t_a.print().c_str());
        abort();
    }
    if ( fabs(t_sN.vals[3] -t_sO.vals[3])<1e-2 ) {
        printf("Same time for two states!!! %s === %s\n",t_sO.print().c_str(),t_sN.print().c_str());
        abort();
    }*/
    
    for (int i(0); i<20; i++) {
        max_scale[i] = std::max(max_scale[i], t_sN.vals[i]);
        min_scale[i] = std::min(min_scale[i], t_sN.vals[i]);
    }

    bool new_sample(false);
    if (reward<-9.9) new_sample=true;
    
    if (study == 0) {
        const Real scaledEfficiency = (t_sN.vals[18]-.4)/(1.-.4);
#ifndef _scaleR_
        reward = scaledEfficiency;
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
#else
        reward = (1.-gamma)*scaledEfficiency; //max cumulative reward = sum gamma^t r < 1/(1-gamma) = 1
        if (new_sample) reward = -1.;  // = - max cumulative reward
#endif
    }
    else if (study == 1) {
        const Real scaledEfficiency = (t_sN.vals[21]-.3)/(.6-.3);
#ifndef _scaleR_
        reward = scaledEfficiency;
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
#else
        reward = (1.-gamma)*scaledEfficiency; //max cumulative reward = sum gamma^t r < 1/(1-gamma) = 1
        if (new_sample) reward = -1.;  // = - max cumulative reward
#endif
    }
    else if (study == 2) {
#ifndef _scaleR_
        reward =  1.-fabs(t_sN.vals[1]-goalDY)/.5;
#else
        reward = (1.-gamma)*(1.-fabs(t_sN.vals[1]-goalDY));
        if (new_sample) reward = -1.;
#endif
    }
    else if (new_sample) reward = -10.;
         
    //    die("Wrong reward\n");
    //}
    
    return new_sample;
}
