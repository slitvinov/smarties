//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Worker_h
#define smarties_Worker_h

#include "Launcher.h"

#include <thread>

namespace smarties
{

class Worker
{
public:
  Worker(Settings& settings, DistributionInfo& distribinfo);

  void synchronizeEnvironments();

  void runTraining();
  void loopSocketToMaster() const;

  // may be called from application:
  void answerStateAction(const int bufferID) const;
  void stepWorkerToMaster(const Uint bufferID) const;

protected:
  const Settings& settings;
  DistributionInfo& distrib;
  TaskQueue tasks;

  const Launcher COMM(this, distrib, settings.bTrain);

  const MPI_Comm& master_workers_comm = distrib.master_workers_comm;
  const MPI_Comm& workerless_masters_comm = distrib.workerless_masters_comm;
  const MPI_Comm& learners_train_comm = distrib.learners_train_comm;

  const std::vector<std::unique_ptr<Learner>> learners;

  const Environment& ENV = COMM.ENV;
  const std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;

  const Uint nCallingEnvs = distrib.nOwnedEnvironments;
  const int bTrain = settings.bTrain;

  // small utility functions:
  Uint getLearnerID(const Uint agentIDlocal) const;
  bool learnersBlockingDataAcquisition() const;
  void dumpCumulativeReward(const Agent&, const Uint k, const Uint t) const;

  void answerStateActionCaller(const int bufferID);

  int getSocketID(const Uint worker) const {
    assert( worker>=0 && worker <= COMM.SOCK.clients.size() );
    return worker>0? COMM.SOCK.clients[worker-1] : COMM.SOCK.server;
  }
  const COMM_buffer& buffer getCommBuffer(const Uint worker) const {
    assert( worker>0 && worker <= COMM.BUFF.size() );
    return * COMM.BUFF[worker-1].get();
  }
};

} // end namespace smarties
#endif // smarties_Worker_h
