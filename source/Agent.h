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
#include "Communicator.h"

class Agent
{
protected:
	StateInfo  sInfo;
	ActionInfo aInfo;

public:
	State *s = nullptr;
	State *sOld = nullptr;
	Action *a = nullptr;
	Real r = 0;
	Real cumulative_rewards = 0;
	const int ID;
	int Status = 1;
	int transitionID = 0;

	Agent(const int _ID = 0) : ID(_ID) { }

	~Agent()
	{
		_dispose_object(s);
		_dispose_object(sOld);
		_dispose_object(a);
	}
	void getState(State& _s) const
	{
		assert(s not_eq nullptr);
		_s = *s;
	}
	void setState(State& _s)
	{
		*s = _s;
	}
	void swapStates()
	{
		assert(s not_eq nullptr);
		assert(sOld not_eq nullptr);
		std::swap(s, sOld);
	}
	void getAction(Action& _a) const
	{
		assert(a not_eq nullptr);
		_a = *a;
	}
	void getOldState(State& _s) const
	{
		assert(sOld not_eq nullptr);
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

	void reset()
	{
		Status = 1; transitionID=0; cumulative_rewards=0; r=0;
	}

	void update(const _AGENT_STATUS _i, const vector<double>& _s, const double _r)
	{
		Status = _i;
		swapStates(); //swap sold and snew
		s->set(_s);
		r = _r;
		if(_i == _AGENT_FIRSTCOMM) {
			cumulative_rewards = 0;
			transitionID = 0;
		}
		else {
			cumulative_rewards += _r;
			transitionID++;
		}
	}
	StateInfo getStateDims() {return sInfo;}
	ActionInfo getActionDims() {return aInfo;}

	void setDims(const StateInfo& stateInfo, const ActionInfo& actionInfo)
	{
		this->aInfo = actionInfo;
		this->sInfo = stateInfo;
	}
};
