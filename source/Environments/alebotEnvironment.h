
#pragma once

#include "Environment.h"

class alebotEnvironment : public Environment
{
public:
    alebotEnvironment(const int nAgents, const string execpath,
                    const int _rank, Settings & settings);

    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a, 
                    const State& t_sN, Real& reward, const int info) override;
	bool predefinedNetwork(Network* const net) const override;
};
