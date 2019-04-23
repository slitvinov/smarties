//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Launcher_h
#define smarties_Launcher_h

#include "../Communicators/Communicator.h"
#include "../Settings.h"

namespace smarties
{

class Launcher: public Communicator
{
 protected:
  DistributionInfo& distrib;

  std::string execPath    = distrib.launchFile;
  std::string setupFolder = distrib.setupFolder;

  std::vector<std::string> argsFiles;
  std::vector<Uint> argFilesStepsLimits;
/*
  std::mutex& mpi_mutex = S.mpi_mutex;
  const int bAsync        = S.bAsync;
  std::vector<MPI_Request> requests =
                        std::vector<MPI_Request>(nOwnWorkers, MPI_REQUEST_NULL);
*/

public:

  void initArgumentFileNames();
  void createGoRunDir(char* initDir, Uint folderID, MPI_Comm anvAppCom);
  std::vector<char*> readRunArgLst(const std::string paramfile);

  void forkApplication(const Uint nThreads, const Uint nOwnWorkers);
  void runApplication(const MPI_Comm envApplication_comm,
                      const Uint totalNumWorkers,
                      const Uint thisWorkerGroupID);

  //called by smarties
  Launcher(Worker* const W, DistributionInfo& D, bool isTraining);
};

} // end namespace smarties
#endif // smarties_Launcher_h
