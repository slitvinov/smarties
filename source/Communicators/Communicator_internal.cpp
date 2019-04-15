//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator_internal.h"
#include "CommunicatorUtils.h"

// main function call for user's application
// arguments are: - the communicator with smarties
//                - the mpi communicator to use within the app
//                - number of (action-state) timesteps to perform
//                  before returning (it can be ignored)
//                - argc and argv read from settings file
int app_main(Communicator*const smartiesCommunicator,
             const MPI_Comm mpicom,
             const Uint numDesiredTimeSteps,
             int argc,
             char**argv);

Communicator_internal::Communicator_internal(Settings& _S) :
  Communicator(_S.generators[0], _S.bTrain, _S.totNumSteps), S(_S)
{
  comm_learn_pool = S.workersComm;
  update_rank_size();
}

void Communicator_internal::answerTerminateReq(const double answer)
{
  data_action[0] = answer;
   //printf("I think im givign the goahead %f\n",data_action[0]);
  send_all(clientSockets[0], data_action, size_action);
}

void Communicator_internal::sendTerminateReq()
{
  //it's awfully ugly, i send -256 to kill the workers... but...
  //what are the chances that learner sends action -256.(+/- eps) to clients?
  for (int i = 1; i <= nOwnWorkers; i++)
  {
    outBufs[i-1][0] = AGENT_KILLSIGNAL;
    if(bSpawnApp) send_all(clientSockets[i-1], outBufs[i-1], size_action);
    else MPI(Ssend, outBufs[i-1], size_action, MPI_BYTE, i, 0, comm_learn_pool);
  }
}

void Communicator_internal::forkApplication()
{
  #pragma omp parallel num_threads(S.nThreads)
  for(int i = 0; i<nOwnWorkers; i++)
  {
    const int thrID = omp_get_thread_num(), thrN = omp_get_num_threads();
    const int tgtCPU =  ( ( (-1-i) % thrN ) + thrN ) % thrN;
    const int workloadID = i + nOwnWorkers * MPICommSize(MPI_COMM_WORLD);

    #pragma omp critical
    if ( ( thrID==tgtCPU ) && ( fork() == 0 ) )
    {
      char currDirectory[512];
      createGo_rundir(currDirectory, workloadID);
      printf("About to exec %s.... \n", exec.c_str());

      //App output file descriptor:
      std::pair<int, fpos_t> currOutputFdescriptor;
      redirect_stdout_init(currOutputFdescriptor, workloadID);
      // TODO ARGS!
      const std::string exec = "../" + execPath; // TODO : TEST currDirectory
      const int res = execlp(exec.c_str(), exec.c_str(), NULL);

      redirect_stdout_finalize(currOutputFdescriptor);
      //int res = execvp(*largv, largv);
      if(res < 0)
        fprintf(stderr, "Unable to exec file '%s'!"
                "Make sure it exists and it is executable.\n", exec.c_str());
      printf("Client application returned. Aborting..."); fflush(0); abort();
    }
  }
}

void Communicator_internal::runApplication()
{
  assert(workerGroup>=0 && comm_inside_app not_eq MPI_COMM_NULL);
  // app only needs lower level functionalities:
  // ie. send state, recv action, specify state/action spaces properties...
  Communicator* const commptr = static_cast<Communicator*>(this);

  while(true)
  {
    char currDirectory[512];
    createGoRunDir(currDirectory, workerGroup);

    Uint settingsInd = 0;
    for(size_t i=0; i<argsFiles.size(); i++)
      if(learner_step_id >= stepPrefix[i]) settingsInd = i;
    Uint numStepTSet = stepPrefix[settingsInd+1] - learner_step_id;
    numStepTSet = numStepTSet * size_inside_app / S.nWorkers;
    std::vector<char*> args = readRunArgLst(argsFiles[settingsInd]);

    //App output file descriptor:
    std::pair<int, fpos_t> currOutputFdescriptor;
    redirect_stdout_init(currOutputFdescriptor, 0);

    app_main(commptr, numStepTSet, comm_inside_app, args.size()-1, args.data());

    redirect_stdout_finalize(currOutputFdescriptor);

    for(size_t i = 0; i < args.size()-1; i++) delete[] args[i];
    chdir(currDirectory);  // go to original directory
  }
}

void Communicator_internal::initArgumentFileNames()
{
  // paramfile is a list of text files separated by commas
  // e.g. settings1.txt,settings2.txt,...
  argsFiles = split(paramfile, ',');
  if(argsFiles.size() == 0) {
    if(paramfile not_eq "")
      _die("error in splitting paramfile %s", paramfile.c_str());
    argsFiles.push_back("");
  }
  assert(argsFiles.size() > 0);

  // nStepPerFile is a list of numbers representing how many timesteps we should
  // run with a settings file before switching to the next one. e.g. 1000,...
  // Here '0' means : run the settings file for ever
  // If empty, we assume the settings file should be run for ever
  if(nStepPerFile == "") nStepPerFile = "0";
  std::vector<std::string> stepNmbrs = split(nStepPerFile, ',');
  if(argsFiles.size() not_eq stepNmbrs.size())
    _die("mismatch in sizes: argsFiles=%s stepNmbrs=%s", argsFiles, stepNmbrs);

  argFilesStepsLimits = std::vector<Uint>(argsFiles.size(), 0);
  argFilesStepsLimits[0] = 0; // first settings file is used from step 0
  for (size_t i=1; i<stepNmbrs.size(); i++)
    stepPrefix[i] = stepPrefix[i-1] + std::stol(stepNmbrs[i-1]);
  stepPrefix.push_back(std::numeric_limits<Uint>::max()); //last setup used for ever
  assert(stepPrefix.size() == argsFiles.size() + 1);
}

void Communicator_internal::createGoRunDir(char[] initDir, Uint folderID)
{
  char newd[1024];
  getcwd(initd, 512);
  struct stat fileStat;
  unsigned long iter = 0;

  while(true)
  {
    sprintf(newd, "%s/%s_%03d_%05lu", initd, "simulation", folderID, iter);
    //cout << newd << endl; fflush(0); fflush(stdout);
    if ( stat(newd, &fileStat) >= 0 ) iter++; // directory already exists
    else
    {
      if( MPICommSize(workers_application_comm)>1 )
          MPI_Barrier(workers_application_comm);

      if( MPICommRank(workers_application_comm)>1 ) // app's root sets up dir
      {
        mkdir(newd, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(setupFolder not_eq "") //copy any file in the setup dir
        {
          if (copy_from_dir(("../"+setupfolder).c_str()) not_eq 0 )
            _die("Error in copy from dir %s\n", setupfolder.c_str());
        }
      }

      if( MPICommSize(workers_application_comm)>1 )
          MPI_Barrier(workers_application_comm);

      chdir(newd);
      break;
    }
  }
}

std::vector<char*> Communicator_internal::readRunArgLst(const std::string _paramfile)
{
  std::vector<char*> args;
  if (_paramfile == "") {
    warn("empty parameter file path");
    args.push_back(0);
    return args;
  }
  std::ifstream t(("../"+_paramfile).c_str());
  std::string linestr((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
  if(linestr.size() == 0) die("did not find parameter file");
  std::istringstream iss(linestr); // params file is read into iss
  std::string token;
  while(iss >> token)
  {
    // If one runs an executable and provides an argument like ./exec 'foo bar'
    // then `foo bar' is put in its entirety in argv[1]. However, when we ask
    // user to write a settingsfile, apostrophes are read as characters, not as
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
    char *arg = new char[token.size() + 1];
    copy(token.begin(), token.end(), arg);  // write into char array
    arg[token.size()] = '\0';
    args.push_back(arg);
  }
  args.push_back(0); // push back nullptr as last entry
  return args; // remember to deallocate it!
}
