//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_pytorch.h"
#include <chrono>
#include "../ReplayMemory/Collector.h"

// PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

#include <iostream>


namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(smarties::NNvec);
PYBIND11_MAKE_OPAQUE(std::vector< smarties::NNvec >);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::NNvec > >);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::Rvec* > >);
// PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::Real > >);
// PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::nnReal > >);  // <--- Do not convert to lists.
PYBIND11_MAKE_OPAQUE(std::vector<smarties::MiniBatch*>);

namespace smarties
{

PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
    py::class_<MiniBatch>(m, "MiniBatch")
        // .def(py::init<>())
        .def_readwrite("S", &MiniBatch::S)
        .def_readwrite("A", &MiniBatch::A)
        .def_readwrite("MU", &MiniBatch::MU)
        .def_readwrite("R", &MiniBatch::R)
        .def_readwrite("W", &MiniBatch::W);

    py::bind_vector<NNvec>(m, "NNvec");
    py::bind_vector<std::vector<NNvec>>(m, "VectorNNvec");
    py::bind_vector<std::vector<std::vector<NNvec>>>(m, "VectorVectorNNvec");
    // py::bind_vector<std::vector<std::vector<Real>>>(m, "VectorVectorReal");
    // py::bind_vector<std::vector< std::vector< nnReal > > >(m, "VectorVectornnReal");
    py::bind_vector<std::vector< std::vector< Rvec* > > >(m, "VectorVectorRvecPointer");
    py::bind_vector<std::vector<MiniBatch*>>(m, "VectorMiniBatch");
}


py::scoped_interpreter guard{};

Learner_pytorch::Learner_pytorch(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner(MDP_, S_, D_)
{
  std::cout << "PYTORCH: STARTING NEW LEARNER." << std::endl;
  std::cout << "PYTORCH: Initializing pytorch scope..." << std::endl;

  py::module sys = py::module::import("sys");
  std::string path = "/home/pvlachas/smarties/source/Learners/Pytorch";
  sys.attr("path").attr("insert")(0,path);
  auto module = py::module::import("mlp_module");
  auto Net = module.attr("MLP");

  int input_dim = MDP_.dimStateObserved;
  int time_steps = 1;
  int output_dim = aInfo.dimPol();

  // Outputing the mean action and the standard deviation of the action (possibly also the value function)
  std::cout << "PYTORCH: Output dimension (aInfo.dimPol())=" << output_dim << std::endl;

  auto net = Net(time_steps, input_dim, S_.nnLayerSizes, output_dim);
  Nets.emplace_back(Net(time_steps, input_dim, S_.nnLayerSizes, output_dim));
  Nets[0] = net;
  std::cout << "PYTORCH: NEW LEARNER STARTED." << std::endl;

  std::cout << "PYTORCH: PROPAGATING THROUGH THE LEARNER." << std::endl;
  auto torch = py::module::import("torch");
  int batch_size = 7;
  auto input_ = torch.attr("randn")(batch_size, time_steps, input_dim);
  auto output = Nets[0].attr("forwardVector")(input_);
  std::cout << "PYTORCH: PROPAGATION WORKED!" << std::endl;

}

Learner_pytorch::~Learner_pytorch() {
  // _dispose_object(input);
}


void Learner_pytorch::setupTasks(TaskQueue& tasks)
{
  std::cout << "PYTORCH: SETTING UP TASKS..." << std::endl;

  // If not training (e.g. evaluate policy)
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization
  auto stepInit = [&]()
  {
    // conditions to start the initialization task:
    if ( algoSubStepID >= 0 ) return; // we done with init
    if ( data->readNData() < nObsB4StartTraining ) return; // not enough data to init

    debugL("Initialize Learner");
    initializeLearner();
    algoSubStepID = 0;
  };
  tasks.add(stepInit);

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data

    debugL("Sample the replay memory and compute the gradients");
    spawnTrainTasks();
    // // debugL("Gather gradient estimates from each thread and Learner MPI rank");
    // // prepareGradient();
    // debugL("Search work to do in the Replay Memory");
    // processMemoryBuffer(); // find old eps, update avg quantities ...
    // debugL("Update Retrace est. for episodes sampled in prev. grad update");
    // updateRetraceEstimates();
    // debugL("Compute state/rewards stats from the replay memory");
    // finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev
    logStats();
    
    algoSubStepID = 1;
  };
  tasks.add(stepMain);

  // these are all the tasks I can do before the optimizer does an allreduce
  auto stepComplete = [&]()
  {
    if ( algoSubStepID not_eq 1 ) return;
    // if ( networks[0]->ready2ApplyUpdate() == false ) return;

    // debugL("Apply SGD update after reduction of gradients");
    // applyGradient();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
  };
  tasks.add(stepComplete);

  std::cout << "PYTORCH: TASKS ALL SET UP..." << std::endl;
  std::cout << "PYTORCH: Data: " << data->readNSeq() << std::endl;
}



void Learner_pytorch::spawnTrainTasks()
{
  // std::cout << "PYTORCH: SPAWNING TRAINING TASKS. " << std::endl;
  // std::cout << "PYTORCH: Number of sequences: " << data->readNSeq() << std::endl;

  if(settings.bSampleSequences && data->readNSeq() < (long) settings.batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  const Uint nThr = distrib.nThreads, CS =  batchSize / nThr;
  const MiniBatch MB = data->sampleMinibatch(batchSize, nGradSteps() );

  // std::cout << "MB.episodes.size()=" << MB.episodes.size() << std::endl;

  if(settings.bSampleSequences)
  {
    // #pragma omp parallel for collapse(2) schedule(dynamic,1) num_threads(nThr)
    for (Uint wID=0; wID<ESpopSize; ++wID)
    {
      // std::cout << "PYTORCH: READING AN EPISODE!" << std::endl;
      // std::cout << MB.episodes[0]->states[0] << std::endl;
    }
    // for (Uint bID=0; bID<batchSize; ++bID) {
      // for (const auto & net : networks ) net->load(MB, bID, wID);
      // Train(MB, wID, bID);
      // // backprop, from last net to first for dependencies in gradients:
      // for (const auto & net : Utilities::reverse(networks) ) net->backProp(bID);
    // }
  }


  // std::vector<Uint> samp_seq = std::vector<Uint>(batchSize, -1);
  // std::vector<Uint> samp_obs = std::vector<Uint>(batchSize, -1);
  // data->sample(samp_seq, samp_obs);

  // for(Uint i=0; i<batchSize && bSampleSequences; i++)
  //   assert( samp_obs[i] == data->get(samp_seq[i])->ndata() - 1 );

  // for(Uint i=0; i<batchSize; i++)
  // {
  //   Sequence* const S = data->get(samp_seq[i]);
  //   std::vector<memReal> state = S->states[samp_obs[i]];
  //   std::vector<Real> action = S->actions[samp_obs[i]];
  //   std::vector<Real> policy = S->policies[samp_obs[i]];
  //   Real reward = S->rewards[samp_obs[i]];

  //   std::cout << "STATE:: " << std::endl;
  //   std::cout << "state SIZE: " << state.size() << std::endl;
  //   std::cout << "action SIZE: " << action.size() << std::endl;
  //   std::cout << "policy SIZE: " << policy.size() << std::endl;
  //   std::cout << "reward: " << reward << std::endl;

  //   // py::print(state[0]);
  //   // py::print(state[1]);
  //   // py::print(state[2]);

  // // int input_dim = 2;
  // // auto input = torch.attr("randn")(input_dim);
  // // py::print(input);
  // // py::print(input.attr("size")());

  // // auto output = net.attr("forward")(input);
  // // py::print(output);
  // // py::print(output.attr("size")());
  //   // std::cout << "OUTPUT:: " << std::endl;
  //   // py::print(state[0]);
  // }

}


void Learner_pytorch::select(Agent& agent)
{
  std::cout << "PYTORCH: AGENT SELECTING ACTION!" << std::endl;

  data_get->add_state(agent);
  Sequence& EP = * data_get->get(agent.ID);

  MiniBatch MB = data->agentToMinibatch(&EP);

  std::cout << "MB.episodes.size()=" << MB.episodes.size() << std::endl;
  std::cout << "MB.S[0].size()=" << MB.S[0].size() << std::endl;
  std::cout << "MB.A[0].size()=" << MB.A[0].size() << std::endl;
  std::cout << "MB.MU[0].size()=" << MB.MU[0].size() << std::endl;
  std::cout << "MB.R[0].size()=" << MB.R[0].size() << std::endl;
  std::cout << "MB.W[0].size()=" << MB.W[0].size() << std::endl;

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    // IMPORTANT !
    std::cout << "PYTORCH: IMPORTING THE EMBEDDER!" << std::endl;
    py::module::import("pybind11_embed");

    std::vector<MiniBatch*> vectorMiniBatch;

    vectorMiniBatch.push_back(&MB);

    std::reference_wrapper<std::vector<MiniBatch*>> vectorMiniBatch_ref{vectorMiniBatch};

    // Initializing action to be taken with zeros
    Rvec action = Rvec(aInfo.dim(), 0);
    // Initializing mu to be taken with zeros
    Rvec mu = Rvec(aInfo.dimPol(), 0);

    std::reference_wrapper<Rvec> action_ref{action};
    std::reference_wrapper<Rvec> mu_ref{mu};


    auto locals = py::dict("vectorMiniBatch"_a=vectorMiniBatch_ref, "action"_a=action_ref, "mu"_a=mu_ref);
    // {
      auto output = Nets[0].attr("forwardCPPDict")(locals);
      // py::gil_scoped_release();
    // }

    std::cout << "### OUPUT GOOD. ###" << std::endl;

    agent.act(action);
    data_get->add_action(agent, mu);

    std::cout << "### NEXT ###" << std::endl;


// {
//     auto action = Nets[0].attr("forward")(py::array(state.size(), state.data()));
//     py::print(action.attr("data"));
//     py::print(action.attr("__dir__")());
//     py::gil_scoped_release();
// }

      // py::print("BRACKE");



  // int input_dim = sInfo.dimUsed;
  // auto input = torch.attr("randn")(input_dim);
  // py::print(input);
  // py::print(input.attr("size")());
  // auto output = Nets[0].attr("forward")(input);
  // py::print(output);
  // py::print(output.attr("size")());




    // auto action = Nets[0].attr("forward")(py::list(state.size(), state.data()));

    // py::print(py::module::import("pybind11").attr("__version__"));
    // py::print(py::module::import("pybind11").attr("__file__"));

    // py::print("ARAASASA");
    // auto temp = py::array(state.size(), state.data());
    // auto action = Nets[0].attr("forward")(temp);
    // py::print("NET FEEDED 1!");

  // py::print(action);

  // auto input = torch.attr("randn")(sInfo.dimUsed);
  // py::print(input);
  // py::print(input.attr("size")());
  // auto output = Nets[0].attr("forward")(input);
  // py::print(output);

  // std::vector<Real> temp(sInfo.dimUsed);
  // auto output = Nets[0].attr("forward")(py::array(temp.size(), temp.data()));

  // py::print(output);
  // py::print(output.attr("size")());

  // auto g = output.attr("detach")();
  // py::print(g);
  // auto g2 = g.attr("numpy")();
  // py::print(g2);



    // Nets[0].attr("forward")(py::array(state.size(), state.data()));

    // auto action = Nets[0].attr("forward")(state);

    // py::print("NET FEEDED 2!");

    // py::print(action);

    // std::cout << "action" << action[0] << std::endl;
    // py::print(action[0]);

    // std::cout << "SELECTING ACTION:" << std::endl;
    // std::cout << "ACTION SIZE:" << aInfo.dim << std::endl;
    // std::cout << "policyVecDim:" << policyVecDim << std::endl;

    // auto torch = py::module::import("torch");
//  \hat{V}(s_t) = V(s_t) + \rho(s_t, a_t) * (r_{t+1} + gamma V(s_{t-1}) - V(s_t) )
    // auto state = agent.getState()
    // std::cout << "state.sInfo.dimUsed:" << state.sInfo.dimUsed << std::endl;
    // std::cout << "state.vals[0]:" << state.vals[0] << std::endl;


    // auto input = torch.attr("randn")(sInfo.dimUsed);
    // py::print(input);
    // py::print(input.attr("size")());

    // auto output = Nets[0].attr("forward")(input);
    // py::print(output);
    // py::print(output.attr("size")());

    // py::print("ACTION SELECTED !");


    // Rvec fakeAction = Rvec(aInfo.dim(), 0);
    // Rvec mu(aInfo.dimPol(), 0);
    // agent.act(fakeAction);
    // data_get->add_action(agent, mu);

    // py::print("PYTORCH: DONE SELECTING ACTION!");
    // std::cout << "PYTORCH: DONE SELECTING ACTION!" << std::endl;

  // delete action;

  }else
  {
    data_get->terminate_seq(agent);
  }

}

void Learner_pytorch::getMetrics(std::ostringstream& buf) const
{
  Learner::getMetrics(buf);
}
void Learner_pytorch::getHeaders(std::ostringstream& buf) const
{
  Learner::getHeaders(buf);
}

void Learner_pytorch::restart()
{
  // if(settings.restart == "none") return;
  // if(!learn_rank) printf("Restarting from saved policy...\n");

  // for(auto & net : F) net->restart(settings.restart+"/"+learner_name);
  // input->restart(settings.restart+"/"+learner_name);

  // for(auto & net : F) net->save("restarted_"+learner_name, false);
  // input->save("restarted_"+learner_name, false);

  Learner::restart();

  // if(input->opt not_eq nullptr) input->opt->nStep = _nGradSteps;
  // for(auto & net : F) net->opt->nStep = _nGradSteps;
}

void Learner_pytorch::save()
{
  // const Uint currStep = nGradSteps()+1;
  // const Real freqSave = freqPrint * PRFL_DMPFRQ;
  // const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  // const bool bBackup = currStep % freqBackup == 0;
  // for(auto & net : F) net->save(learner_name, bBackup);
  // input->save(learner_name, bBackup);

  Learner::save();
}

}
