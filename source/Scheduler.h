//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once

class Learner;
#include "Communicator.h"
#include "Learners/Learner.h"

class Master
{
private:
	const MPI_Comm slavesComm;
	Learner* const learner;
	Environment* const env;
	const ActionInfo aI;
	const StateInfo  sI;
	const vector<Agent*> agents;
	const int bTrain, nPerRank, nSlaves, nThreads, saveFreq, inSize, outSize, learn_rank, learn_size;
	double*const inbuf;
	double*const outbuf;
	State  sOld, sNew;
	Action aOld, aNew;
	Real meanR = 0, varR = 0;
	unsigned long iter = 0;
	vector<int> status;
	vector<Real> cumulative_rewards;

	MPI_Request request;
	void trackAgentsPerformance(const _AGENT_STATUS agentStatus, const int agent, const Real reward);
	void recvState(const int slave, int& iAgent, int& istatus, Real& reward);
	void sendAction(const int slave, const int iAgent);
	void save();

public:
	Master(MPI_Comm comm, Learner*const learner, Environment*const env, Settings& settings);
	~Master()
	{
		_dispose_object(env);
		_dealloc(inbuf);
		_dealloc(outbuf);
		_dispose_object(learner);
	}
	void sendTerminateReq(const double msg = -256)
	{
		//it's ugly, i send -256 to kill the slaves... but...
		//what are the chances that learner sends action -256.(+/- eps) to clients?
		outbuf[0] = msg;
		printf("nslaves %d\n",nSlaves);
		for (int slave=1; slave<=nSlaves; slave++)
		MPI_Ssend(outbuf, outSize, MPI_BYTE, slave, 0, slavesComm);
	}
	void run();
	void restart(string fname);
};

class Slave
{
private:
	Communicator* const comm;
	Environment* const env;
	const bool bTrain;
	vector<int> status;

public:
	Slave(Communicator*const c, Environment*const e, Settings& s);
	~Slave()
	{
		_dispose_object(env);
	}
	void run();
};

class Client
{
private:
	Learner* const learner;
	Communicator* const comm;
	Environment* const env;
	vector<Agent*> agents;
	const ActionInfo aI;
	const StateInfo  sI;
	State  sOld, sNew;
	Action aOld, aNew;
	vector<int> status;
	void prepareState(int& iAgent, int& istatus, Real& reward);
	void prepareAction(const int iAgent);

public:
	Client(Learner*const l,Communicator*const c,Environment*const e,Settings&s);
	~Client()
	{
		_dispose_object(env);
		_dispose_object(learner);
	}
	void run();
};
