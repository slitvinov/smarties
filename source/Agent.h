/*
 *  Agent.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "StateAction.h"
#include "Settings.h"

#ifdef _RL_VIZ
#ifdef __APPLE__
#include "GLUT/glut.h"
#endif
#endif

using namespace std;

class Agent
{
protected:
	StateInfo  sInfo;
	ActionInfo aInfo;
	
public:
    State *s, *sOld;
    Action *a;
    Real r;
    const int ID;
    int Status; 

    Agent(const int _ID = 0) : ID(_ID), Status(1) { }
    
	~Agent()
    {
        _dispose_object(s);
        _dispose_object(sOld);
        _dispose_object(a);
    }
    void getState(State& _s) const
    {
        _s = *s;
    }
    void getAction(Action& _a) const
    {
        _a = *a;
    }
    void getOldState(State& _s) const
    {
        _s = *sOld;
    }
    void act(Action& _a)
    {
        *a = _a;
    }
    int getStatus() const 
    {
        return Status;
    }
    Real getReward()
    {
        return r;
    }
	
	StateInfo getStateDims() {return sInfo;}
    ActionInfo getActionDims() {return aInfo;}
	
	void setDims(const StateInfo& stateInfo, const ActionInfo& actionInfo)
	{
        this->aInfo = actionInfo;
        this->sInfo = stateInfo;
	}
};
