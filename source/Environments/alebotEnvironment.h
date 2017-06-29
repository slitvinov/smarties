
#pragma once

#include "Environment.h"

class alebotEnvironment : public Environment
{
public:
    alebotEnvironment(const Uint nAgents, const Uint nActions, const string execpath, Settings & settings);

    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;
	bool predefinedNetwork(Builder* const net) const override;

	const Uint legalActions;
};
