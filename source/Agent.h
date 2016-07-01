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
    
    Agent(const int _ID = 0) : ID(_ID) { }
	
    virtual void getState(State& _s) const
    {
        _s = *s;
    }
    virtual void getAction(Action& _a) const
    {
        _a = *a;
    }
    virtual void getOldState(State& _s) const
    {
        _s = *sOld;
    }
    virtual void act(Action& _a)
    {
        *a = _a;
    }
    virtual Real getReward()
    {
        return r;
    }
	
	inline StateInfo getStateDims() {return sInfo;}
    inline ActionInfo getActionDims() {return aInfo;}
	
	inline void setDims(const StateInfo& stateInfo, const ActionInfo& actionInfo)
	{
        this->aInfo = actionInfo;
        this->sInfo = stateInfo;
	}
};
