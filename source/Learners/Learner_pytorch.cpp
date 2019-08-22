//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_pytorch.h"
#include <chrono>

// PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;
using namespace py::literals;

namespace smarties
{

struct PybindSequence
{
    std::vector<double> states;
    std::vector<double> actions;
};

//   // std::vector<std::vector<memReal>> states;
//   // std::vector<std::vector<Real>> actions;
//   // std::vector<std::vector<Real>> policies;
//   // std::vector<Real> rewards;



// PYBIND11_MAKE_OPAQUE(std::vector<double>);  // <--- Do not convert to lists.
// PYBIND11_MAKE_OPAQUE(std::vector<PybindSequence*>);   // <--- Works even without this, because bind_vector is used, but keep it just in case.

// PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
//     py::class_<PybindSequence>(m, "PybindSequence")
//         .def(pybind11::init<>())  // <--- If you want to create PybindSequence object from Python.
//         .def_readwrite("states", &PybindSequence::states)  // <--- Expose member variables.
//         .def_readwrite("actions", &PybindSequence::actions);  // <--- Expose member variables.

//     // Bind vectors without conversion to lists, but keep list interface (e.g. `append` instead of `push_back`).
//     py::bind_vector<std::vector<double>>(m, "VectorDouble");
//     py::bind_vector<std::vector<PybindSequence*>>(m, "VectorSequenceptr");
// }





PYBIND11_MAKE_OPAQUE(std::vector<double>);  // <--- Do not convert to lists.
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);  // <--- Do not convert to lists.
PYBIND11_MAKE_OPAQUE(std::vector<Sequence*>);   // <--- Works even without this, because bind_vector is used, but keep it just in case.

PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
    py::class_<Sequence>(m, "Sequence")
        .def(pybind11::init<>())  // <--- If you want to create Sequence object from Python.
        .def_readwrite("states", &Sequence::states)  // <--- Expose member variables.
        .def_readwrite("actions", &Sequence::actions)  // <--- Expose member variables.
        .def_readwrite("policies", &Sequence::policies)  // <--- Expose member variables.
        .def_readwrite("rewards", &Sequence::rewards);  // <--- Expose member variables.

    // Bind vectors without conversion to lists, but keep list interface (e.g. `append` instead of `push_back`).
    py::bind_vector<std::vector<double>>(m, "VectorDouble");
    py::bind_vector<std::vector<Sequence*>>(m, "VectorSequenceptr");
    py::bind_vector<std::vector<std::vector<double>>>(m, "VectorVectorDouble");
}


py::scoped_interpreter guard{};

// Learner_pytorch::Learner_pytorch(Environment*const E, Settings&S): Learner(E, S)

Learner_pytorch::Learner_pytorch(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner(MDP_, S_, D_)
{
  std::cout << "STARTING NEW LEARNER." << std::endl;

  std::cout << "Initializing pytorch scope..." << std::endl;

  py::module sys = py::module::import("sys");
  std::string path = "/home/pvlachas/smarties/source/Learners/Pytorch";
  py::print(sys.attr("path").attr("insert")(0,path));

  auto module = py::module::import("mlp_module");
  // py::print(module);
  auto Net = module.attr("MLP");

  std::cout << "state dimension: " << sInfo.dimUsed << std::endl;

  int input_dim = sInfo.dimUsed;
  int L1 = 10;
  int L2 = 10;
  // Outputing the value function, the mean action and the standard deviation of the action
  int output_dim = 1 + aInfo.dim + aInfo.dim;

  auto net = Net(input_dim, L1, L2, output_dim);
  Nets.emplace_back(Net(input_dim, L1, L2, output_dim));
  Nets[0] = net;
  std::cout << "NEW LEARNER STARTED." << std::endl;

  // py::print(Net);
  // Nets.push_back(Net())

  // auto net = Net();
  // auto net = Net();
  // Nets = &net;
  // Nets = new auto Net();
  // Nets.push_back(new Net());

  // auto torch = py::module::import("torch");

  // auto input = torch.attr("randn")(input_dim);
  // py::print(input);
  // py::print(input.attr("size")());
  // auto output = net.attr("forward")(input);
  // auto output = Nets.attr("forward")(input);
  // auto output = Nets[0].attr("forward")(input);
  // py::print(output);
  // py::print(output.attr("size")());

  // Nets = new  Net() ;
  // Nets = &net;

}

Learner_pytorch::~Learner_pytorch() {
  // _dispose_object(input);
}

void Learner_pytorch::setupTasks(TaskQueue& tasks) {
  std::cout << "PYTORCH SETTING UP TASKS..." << std::endl;

  // If not training (e.g. evaluate policy)
  if(not bTrain ) return;
  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization
  // Initializing a pipeline (one algorithmic step)
  {
    // Define a condition to initialize
    auto condInit = [&]() {
      if ( algoSubStepID>=0 ) return false;
      return data->readNData() >= nObsB4StartTraining;
    };
    auto stepInit = [&]() {
      debugL("Initialize Learner");
      initializeLearner();
      algoSubStepID = 0;
    };
    tasks.add(condInit, stepInit);
  }

  {
    auto condMain = [&](){
      if (algoSubStepID != 0) return false;
      else return unblockGradientUpdates();
    };
    auto stepMain = [&]() {
      debugL("Sample the replay memory and compute the gradients");
      spawnTrainTasks();

      debugL("Search work to do in the Replay Memory");
      processMemoryBuffer(); // find old eps, update avg quantities ...

      debugL("Update Retrace est. for episodes sampled in prev. grad update");
      updateRetraceEstimates();

      debugL("Compute state/rewards stats from the replay memory");
      finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev

      algoSubStepID = 1;
    };
    tasks.add(condMain, stepMain);
  }
  // these are all the tasks I can do before the optimizer does an allreduce
  {
    auto condComplete = [&]() {
      if(algoSubStepID != 1) return false;
      return true;
    };
    auto stepComplete = [&]() {
      algoSubStepID = 0; // rinse and repeat
      globalGradCounterUpdate(); // step ++
    };
    tasks.add(condComplete, stepComplete);
  }
  std::cout << "PYTORCH TASKS ALL SET UP..." << std::endl;
  std::cout << "Data: " << data->readNSeq() << std::endl;
}

void Learner_pytorch::spawnTrainTasks()
{

  std::cout << "SPAWNING TRAINING TASKS. " << std::endl;
  std::cout << "Number of sequences: " << data->readNSeq() << std::endl;
  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  std::vector<Uint> samp_seq = std::vector<Uint>(batchSize, -1);
  std::vector<Uint> samp_obs = std::vector<Uint>(batchSize, -1);
  data->sample(samp_seq, samp_obs);

  for(Uint i=0; i<batchSize && bSampleSequences; i++)
    assert( samp_obs[i] == data->get(samp_seq[i])->ndata() - 1 );

  for(Uint i=0; i<batchSize; i++){
    Sequence* const S = data->get(samp_seq[i]);
    std::vector<memReal> state = S->states[samp_obs[i]];
    std::vector<Real> action = S->actions[samp_obs[i]];
    std::vector<Real> policy = S->policies[samp_obs[i]];
    Real reward = S->rewards[samp_obs[i]];

    std::cout << "STATE:: " << std::endl;
    std::cout << "state SIZE: " << state.size() << std::endl;
    std::cout << "action SIZE: " << action.size() << std::endl;
    std::cout << "policy SIZE: " << policy.size() << std::endl;
    std::cout << "reward: " << reward << std::endl;



    // py::print(state[0]);
    // py::print(state[1]);
    // py::print(state[2]);
    // py::print(state[3]);
    // py::print(state[4]);
    // py::print(state[5]);
    // py::print(state[6]);


  // int input_dim = 2;
  // auto input = torch.attr("randn")(input_dim);
  // py::print(input);
  // py::print(input.attr("size")());

  // auto output = net.attr("forward")(input);
  // py::print(output);
  // py::print(output.attr("size")());



    // std::cout << "OUTPUT:: " << std::endl;
    // py::print(state[0]);
  }

}

void Learner_pytorch::select(Agent& agent)
{
  // Sequence* const traj = data_get->get(agent.ID);
  Sequence* traj = data_get->get(agent.ID);
  data_get->add_state(agent);

  if( agent.Status < TERM_COMM )
  {
    std::vector<Sequence*> seqVec;
    seqVec.push_back(traj);

    if(seqVec[0]->actions.size() == 0)
    {
      std::cout << "PYTORCH: No action taken, selecting DONE 222 !" << std::endl;
      Rvec fakeAction = Rvec(aInfo.dim, 0);
      Rvec mu(policyVecDim, 0);
      agent.act(fakeAction);
      data_get->add_action(agent, mu);
      py::print("DONE 2!");
      std::cout << "DONE 222 !" << std::endl;

    }

    std::cout << "TRY THIS: " << std::endl;
    std::cout << seqVec.size() << std::endl;
    std::cout << seqVec[0]->states.size() << std::endl;
    std::cout << seqVec[0]->actions.size() << std::endl;
    std::cout << seqVec[0]->states[0].size() << std::endl;
    std::cout << seqVec[0]->actions[0].size() << std::endl;
    // std::cout << seqVec[0]->actions[0][0] << std::endl;
    std::cout << "WORKED: " << std::endl;







    py::print("PYTORCH: Agent selecting action!");

    // std::vector<memReal> state = traj->states[0];
    // std::cout << "State[0] = " << state[0] << std::endl;
    // std::cout << "state.size() = " << state.size() << std::endl;
    // std::cout << "traj->states[0][0] = " << traj->states[0][0] << std::endl;
    // std::cout << "traj->states[0].size() = " << traj->states[0].size() << std::endl;
    // std::cout << "traj->actions[0].size() = " << traj->actions[0].size() << std::endl;


    py::print("PYTORCH: TRYING TO FEED NET");

    // auto torch = py::module::import("torch");

    // auto input = torch.attr("randn")(sInfo.dimUsed);
    // auto action = Nets[0].attr("forward")(input);

    // auto action = Nets[0].attr("forward")(state);

    py::module::import("pybind11_embed");  // <--- Important!

    {
    // PybindSequence s1;
    // s1.states.push_back(1.0);
    // s1.states.push_back(2.0);
    // s1.states.push_back(3.0);
    // s1.states.push_back(4.0);
    // s1.states.push_back(5.0);
    // PybindSequence s2;
    // s2.states.push_back(11.0);
    // s2.states.push_back(12.0);
    // s2.states.push_back(13.0);
    // s2.states.push_back(14.0);
    // s2.states.push_back(15.0);

    // s1.actions.push_back(0.0);
    // s1.actions.push_back(0.0);
    // s1.actions.push_back(0.0);

    // s2.actions.push_back(0.0);
    // s2.actions.push_back(0.0);
    // s2.actions.push_back(0.0);

    // std::vector<PybindSequence*> seqVec;
    // seqVec.push_back(&s1);
    // seqVec.push_back(&s2);

    // std::reference_wrapper<std::vector<PybindSequence*>> seqVec_ref{seqVec};
    // bool policyType = 1;

    // // auto locals = py::dict("var1"_a=seqVec_ref, "var2"_a=seqVec_ref);
    // auto locals = py::dict("seqVec"_a=seqVec_ref, "policyType"_a=policyType);


    // py::print("PROP1:");
    // // std::vector<double> temp(input_dim);
    // auto output = Nets[0].attr("forward")(locals);


    // py::print("DID IT MODIFY THE C++ ?:");
    // std::cout << s1.states[0] << std::endl;

    // py::print("s1.actions:");
    // std::cout << s1.actions[0] << std::endl;
    // std::cout << s1.actions[1] << std::endl;
    // std::cout << s1.actions[2] << std::endl;
    // py::print("OUTPUT:");
    // // std::cout << output << std::endl;



    std::reference_wrapper<std::vector<Sequence*>> seqVec_ref{seqVec};
    bool policyType = 1;

    // auto locals = py::dict("var1"_a=seqVec_ref, "var2"_a=seqVec_ref);
    auto locals = py::dict("seqVec"_a=seqVec_ref, "policyType"_a=policyType);


    // py::print("PROP1:");
    // std::vector<double> temp(input_dim);
    // auto output = Nets[0].attr("forward")(locals);


    // py::print("DID IT MODIFY THE C++ ?:");
    // std::cout << s1.states[0] << std::endl;


    }


    py::print("PYTORCH: PROPAGATION TEST FINISHED.");



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


    Rvec fakeAction = Rvec(aInfo.dim, 0);
    Rvec mu(policyVecDim, 0);
    agent.act(fakeAction);
    data_get->add_action(agent, mu);
    py::print("DONE 2!");
    std::cout << "DONE 222 !" << std::endl;

  // delete action;

  } else
  {
    data_get->terminate_seq(agent);
  }
  std::cout << "EXITING 222 !" << std::endl;
  py::print("EXITING 2!");

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
