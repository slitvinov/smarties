/*
 *  AllSavers.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Saver.h"
#include "../AllSystems.h"

class RewardSaver : public Saver
{
private:
	SelfAvoidEnvironment* env;
	
public:
	RewardSaver(ofstream* f) : Saver(f) { };
	~RewardSaver()
	{
		file->close();
	}
	
	void setEnvironment(Environment *newEnv)
	{
		env = static_cast<SelfAvoidEnvironment*> (newEnv);
	}

	void exec()
	{
		(*file) << env->getAccumulatedReward() << endl;
	}
};

