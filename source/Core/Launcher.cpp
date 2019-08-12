//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Core/Launcher.h"
#include "Core/Worker.h"
#include "Utils/Warnings.h"
#include "Utils/SocketsLib.h"
#include "Utils/LauncherUtilities.h"
#include "Utils/SstreamUtilities.h"

#include <omp.h>
#include <fstream>

namespace smarties
{

Launcher::Launcher(Worker* const W, DistributionInfo& D, bool isTraining) :
  Communicator(W, D.generators[0], isTraining), distrib(D)
{
  initArgumentFileNames();
}

bool Launcher::forkApplication(const environment_callback_t & callback)
{
  const Uint nThreads = distrib.nThreads;
  const Uint nOwnWorkers = distrib.nOwnedEnvironments;
  const Uint totalNumWorkers = distrib.nWorker_processes;
  const environment_callback_MPI_t mpicallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc, argc, argv);
  };

  bool isChild = false;
  #pragma omp parallel num_threads(nThreads)
  for(int i = 0; i < (int) nOwnWorkers; ++i)
  {
    const int thrID = omp_get_thread_num(), thrN = omp_get_num_threads();
    const int tgtCPU =  ( ( (-1-i) % thrN ) + thrN ) % thrN;
    const int workloadID = i +totalNumWorkers * MPICommSize(distrib.world_comm);
    assert(nThreads == (Uint) omp_get_num_threads());
    if( thrID==tgtCPU ) {
      #pragma omp critical
      if ( fork() == 0 ) {
        isChild = true;
        launch(mpicallback, workloadID, MPI_COMM_SELF);
      } else assert(isChild == false);
    }
  }

  if(not isChild) SOCKET_serverConnect(nOwnWorkers, SOCK.clients);
  return isChild;
}

void Launcher::runApplication(const environment_callback_MPI_t & callback )
{
  const Sint thisWorkerGroupID = distrib.thisWorkerGroupID;
  const MPI_Comm envApplication_comm = distrib.environment_app_comm;
  if(thisWorkerGroupID<0) die("Error in setup of envApplication_comm");
  assert(envApplication_comm not_eq MPI_COMM_NULL);
  launch(callback, thisWorkerGroupID, envApplication_comm);
}

void Launcher::launch(const environment_callback_MPI_t & callback,
                      const Uint workLoadID,
                      const MPI_Comm envApplication_comm)
{
  const Uint totalNumWorkers = distrib.nWorker_processes;
  const Uint appSize = MPICommSize(envApplication_comm);
  const Uint appRank = MPICommRank(envApplication_comm);
  // app only needs lower level functionalities:
  // ie. send state, recv action, specify state/action spaces properties...
  Communicator* const commptr = static_cast<Communicator*>(this);

  while(true)
  {
    char currDirectory[512];
    // create dedicated directory for the process:
    createGoRunDir(currDirectory, workLoadID, envApplication_comm);

    Uint settInd = 0;
    for(size_t i=0; i<argsFiles.size(); ++i)
      if(globalTstepCounter >= argFilesStepsLimits[i]) settInd = i;

    Uint numTstepSett = argFilesStepsLimits[settInd+1] - globalTstepCounter;
    numTstepSett = numTstepSett * appSize / totalNumWorkers;
    std::vector<char*> args = readRunArgLst(argsFiles[settInd]);

    // process stdout file descriptor, so that we can revert:
    std::pair<int, fpos_t> currOutputFdescriptor;
    redirect_stdout_init(currOutputFdescriptor, appRank);
    callback(commptr, envApplication_comm, args.size()-1, args.data());
    redirect_stdout_finalize(currOutputFdescriptor);

    for(size_t i = 0; i < args.size()-1; ++i) delete[] args[i];
    chdir(currDirectory);  // go to original directory
    if(bTrainIsOver) break;
  }
}

void Launcher::initArgumentFileNames()
{
  // appSettings is a list of text files separated by commas
  // e.g. settings1.txt,settings2.txt,...
  argsFiles = split(distrib.appSettings, ',');
  if(argsFiles.size() == 0) {
    if(distrib.appSettings not_eq "")
      _die("error in splitting appSettings %s", distrib.appSettings.c_str());
    argsFiles.push_back("");
  }
  assert(argsFiles.size() > 0);

  // nStepPerFile is a list of numbers representing how many timesteps we should
  // run with a settings file before switching to the next one. e.g. 1000,...
  // Here '0' means : run the settings file for ever
  // If empty, we assume the settings file should be run for ever
  if(distrib.nStepPappSett == "") distrib.nStepPappSett = "0";
  std::vector<std::string> stepNmbrs = split(distrib.nStepPappSett, ',');
  using Utilities::vec2string;
  if(argsFiles.size() not_eq stepNmbrs.size())
    _die("mismatch in sizes: argsFiles=%s stepNmbrs=%s",
      vec2string(argsFiles).c_str(), vec2string(stepNmbrs).c_str());

  argFilesStepsLimits = std::vector<Uint>(argsFiles.size(), 0);
  argFilesStepsLimits[0] = 0; // first settings file is used from step 0
  for (size_t i=1; i<stepNmbrs.size(); ++i)
    argFilesStepsLimits[i]= argFilesStepsLimits[i-1] +std::stol(stepNmbrs[i-1]);
  //last setup used for ever:
  argFilesStepsLimits.push_back(std::numeric_limits<Uint>::max());
  assert(argFilesStepsLimits.size() == argsFiles.size() + 1);
}

void Launcher::createGoRunDir(char* initDir, Uint folderID, MPI_Comm envAppCom)
{
  char newDir[1024];
  getcwd(initDir, 512);
  struct stat fileStat;
  unsigned long iter = 0;

  while(true)
  {
    sprintf(newDir,"%s/%s_%03lu_%05lu", initDir, "simulation", folderID, iter);
    if ( stat(newDir, &fileStat) >= 0 ) iter++; // directory already exists
    else
    {
      if( MPICommSize(envAppCom)>1 ) MPI_Barrier(envAppCom);

      if( MPICommRank(envAppCom)<1 ) // app's root sets up dir
      {
        mkdir(newDir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(setupFolder not_eq "") //copy any file in the setup dir
        {
          if (copy_from_dir(("../"+setupFolder).c_str()) not_eq 0 )
            _die("Error in copy from dir %s\n", setupFolder.c_str());
        }
      }

      if( MPICommSize(envAppCom)>1 ) MPI_Barrier(envAppCom);

      chdir(newDir);
      break;
    }
  }
}

std::vector<char*> Launcher::readRunArgLst(const std::string& paramFile)
{
  std::vector<char*> args;
  const auto addArg = [&](const std::string & token)
  {
    char *arg = new char[token.size() + 1];
    copy(token.begin(), token.end(), arg);  // write into char array
    arg[token.size()] = '\0';
    args.push_back(arg);
  };

  // first put argc argv into args:
  for(int i=0; i<distrib.argc; ++i) addArg ( std::string( distrib.argv[i] ) );

  if (paramFile not_eq "")
  {
    std::ifstream t( ("../"+paramFile).c_str() );
    std::string linestr((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());
    if(linestr.size() == 0) die("did not find parameter file");
    std::istringstream iss(linestr); // params file is read into iss
    std::string token;
    while(iss >> token)
    {
      // If one runs an executable and provides a runarg like ./exec 'foo bar'
      // then `foo bar' is put in its entirety in argv[1]. However, when we ask
      // user to write a settingsfile, apostrophes are read as characters, not
      // special symbols, therefore we must do the following workaround to put
      // anything that is written between parenteses in a single argv entry.
      if(token[0]=='\'')
      {
        token.erase(0, 1); // remove apostrophe ( should have been read as \' )
        std::string continuation;
        while(token.back() not_eq '\'') { // if match apostrophe, we are done
          if(!(iss >> continuation)) die("missing matching apostrophe");
          token += " " + continuation; // add next line to argv entry
        }
        token.erase(token.end()-1, token.end()); // remove trailing apostrophe
      }

      addArg ( token );
    }
  }

  args.push_back(nullptr); // push back nullptr as last entry
  return args; // remember to deallocate it!
}

}
