/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "StateAction.h"

class QApproximator
{
protected:
	StateInfo  sInfo;
	ActionInfo actInfo;
	
public:
	QApproximator(StateInfo newSInfo, ActionInfo newActInfo) : sInfo(newSInfo), actInfo(newActInfo) { };
	
	virtual double get(const State& s, const Action& a)				  = 0;
	virtual void   set(const State& s, const Action& a, double value) = 0;
	
	virtual void   save(string name)    = 0;
	virtual bool   restart(string name) = 0;
};
