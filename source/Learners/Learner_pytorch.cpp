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
// #include <pybind11/numpy.h>

#include <iostream>


struct PybindMiniBatch
{
  std::vector<int> example;

  // std::vector< std::vector< std::vector <double> > > S;  // state
  // episodes | time steps | dimensionality
  // std::vector< std::vector< NNvec > > S;  // state
  // std::vector< std::vector< Rvec* > > A;  // action pointer
  // std::vector< std::vector< Rvec* > > MU; // behavior pointer
  // std::vector< std::vector< Real  > > R;  // reward
  // std::vector< std::vector< nnReal> > W;  // importance sampling
};

PYBIND11_MAKE_OPAQUE(std::vector<int>);
// PYBIND11_MAKE_OPAQUE(std::vector<PybindMiniBatch*>);


namespace py = pybind11;
using namespace py::literals;

namespace smarties
{


PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
    py::class_<PybindMiniBatch>(m, "PybindMiniBatch")
        .def(py::init<>())  // <--- If you want to create PybindMiniBatch object from Python.
        .def_readwrite("example", &PybindMiniBatch::example); // <--- Expose member variables.

    // Bind vectors without conversion to lists, but keep list interface (e.g. `append` instead of `push_back`).
    py::bind_vector<std::vector<int>>(m, "VectorInt");
    py::bind_vector<std::vector<PybindMiniBatch*>>(m, "PybindMiniBatchPointer");
}





// PYBIND11_MAKE_OPAQUE(std::vector<double>);  // <--- Do not convert to lists.
// // PYBIND11_MAKE_OPAQUE(std::vector< std::vector<double> >);  // <--- Do not convert to lists.
// // PYBIND11_MAKE_OPAQUE(std::vector< std::vector< std::vector<double> > >);  // <--- Do not convert to lists.

// // PYBIND11_MAKE_OPAQUE(std::vector< std::vector< NNvec > >);  // <--- Do not convert to lists.
// // PYBIND11_MAKE_OPAQUE(std::vector< std::vector< Rvec* > >);  // <--- Do not convert to lists.
// // PYBIND11_MAKE_OPAQUE(std::vector< std::vector< Real > >);  // <--- Do not convert to lists.
// // PYBIND11_MAKE_OPAQUE(std::vector< std::vector< nnReal > >);  // <--- Do not convert to lists.
// PYBIND11_MAKE_OPAQUE(std::vector<PybindMiniBatch*>);   // <--- Works even without this, because bind_vector is used, but keep it just in case.

// PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
//     py::class_<PybindMiniBatch>(m, "PybindMiniBatch")
//         .def(py::init<>())  // <--- If you want to create PybindMiniBatch object from Python.
//         .def_readwrite("S", &PybindMiniBatch::S);
//         // .def_readwrite("S", &PybindMiniBatch::S)  // <--- Expose member variables.
//         // .def_readwrite("A", &PybindMiniBatch::A)  // <--- Expose member variables.
//         // .def_readwrite("MU", &PybindMiniBatch::MU)  // <--- Expose member variables.
//         // .def_readwrite("R", &PybindMiniBatch::R)  // <--- Expose member variables.
//         // .def_readwrite("W", &PybindMiniBatch::W);  // <--- Expose member variables.

//     // Bind vectors without conversion to lists, but keep list interface (e.g. `append` instead of `push_back`).
//     py::bind_vector<std::vector<double>>(m, "VectorVectorVectorDouble");
//     // py::bind_vector<std::vector< std::vector< std::vector<double> > > >(m, "VectorVectorVectorDouble");
//     // py::bind_vector<std::vector< std::vector< NNvec > > >(m, "VectorVectorNNvec");
//     // py::bind_vector<std::vector< std::vector< Real > > >(m, "VectorVectorReal");
//     // py::bind_vector<std::vector< std::vector< nnReal > > >(m, "VectorVectornnReal");
//     // py::bind_vector<std::vector< std::vector< Rvec* > > >(m, "VectorVectornnRvecPointer");

//     py::bind_vector<std::vector<PybindMiniBatch*>>(m, "VectorPybindMiniBatch");
// }




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



// PYBIND11_MAKE_OPAQUE(std::vector<double>);  // <--- Do not convert to lists.
// // PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);  // <--- Do not convert to lists.
// PYBIND11_MAKE_OPAQUE(std::vector<Sequence*>);   // <--- Works even without this, because bind_vector is used, but keep it just in case.

// PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
//     py::class_<Sequence>(m, "Sequence")
//         .def(pybind11::init<>())  // <--- If you want to create Sequence object from Python.
//         .def_readwrite("states", &Sequence::states)  // <--- Expose member variables.
//         .def_readwrite("actions", &Sequence::actions)  // <--- Expose member variables.
//         .def_readwrite("policies", &Sequence::policies)  // <--- Expose member variables.
//         .def_readwrite("rewards", &Sequence::rewards);  // <--- Expose member variables.

//     // Bind vectors without conversion to lists, but keep list interface (e.g. `append` instead of `push_back`).
//     py::bind_vector<std::vector<double>>(m, "VectorDouble");
//     py::bind_vector<std::vector<Sequence*>>(m, "VectorSequenceptr");
//     py::bind_vector<std::vector<std::vector<double>>>(m, "VectorVectorDouble");
// }


py::scoped_interpreter guard{};

Learner_pytorch::Learner_pytorch(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner(MDP_, S_, D_)
{
  std::cout << "PYTORCH: STARTING NEW LEARNER." << std::endl;
  std::cout << "PYTORCH: Initializing pytorch scope..." << std::endl;

  py::module sys = py::module::import("sys");
  std::string path = "/home/pvlachas/smarties/source/Learners/Pytorch";
  py::print(sys.attr("path").attr("insert")(0,path));
  auto module = py::module::import("mlp_module");
  auto Net = module.attr("MLP");

  std::cout << "PYTORCH: MDP_.dimStateObserved= " << MDP_.dimStateObserved << std::endl;

  int input_dim = MDP_.dimStateObserved ;
  int L1 = 10;
  int L2 = 10;
  // Outputing the mean action and the standard deviation of the action (possibly also the value function)
  int output_dim = 2*MDP_.dimAction;
  std::cout << "PYTORCH: Output dimension=" << output_dim << std::endl;

  auto net = Net(input_dim, L1, L2, output_dim);
  Nets.emplace_back(Net(input_dim, L1, L2, output_dim));
  Nets[0] = net;
  std::cout << "PYTORCH: NEW LEARNER STARTED." << std::endl;


  std::cout << "PYTORCH: PROPAGATING THROUGH THE LEARNER." << std::endl;
  auto torch = py::module::import("torch");
  auto input = torch.attr("randn")(input_dim);
  auto output = Nets[0].attr("forwardVector")(input);
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
  // Sequence* const traj = data_get->get(agent.ID);
  // Sequence* traj = data_get->get(agent.ID);

  data_get->add_state(agent);
  Sequence& EP = * data_get->get(agent.ID);

  const MiniBatch MB = data->agentToMinibatch(&EP);
  // NET->load(MB, agent, 0);

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

    PybindMiniBatch pybindMB;
    std::vector<int> vec;
    vec.push_back(0);
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    pybindMB.example = vec;
    std::cout << "pybindMB.example.size()=" << pybindMB.example.size() << std::endl;

    std::cout << "PYTORCH: 1" << std::endl;
    std::cout << pybindMB.example[2] << std::endl;

    std::vector<PybindMiniBatch*> vectorMiniBatch;
    std::reference_wrapper<std::vector<PybindMiniBatch*>> vectorMiniBatch_ref{vectorMiniBatch};
    vectorMiniBatch.push_back(&pybindMB);
    std::cout << "PYTORCH: 2" << std::endl;

    bool policyType = 1;
    // auto locals = py::dict("vectorMiniBatch"_a=vectorMiniBatch_ref, "policyType"_a=policyType);
    auto locals = py::dict("vectorMiniBatch"_a=vectorMiniBatch_ref);

    std::cout << "PYTORCH: 3" << std::endl;

    std::cout << "PYTORCH: SELECTING ACTION FROM THE NET" << std::endl;
    // auto output = Nets[0].attr("forward")(locals);

    std::cout << "PYTORCH: ACTION SELECTED!" << std::endl;




    // PybindMiniBatch pybindMB;
    // pybindMB.S = MB.S;
    // // pybindMB.A = MB.A;
    // // pybindMB.MU = MB.MU;
    // // pybindMB.R = MB.R;
    // // pybindMB.W = MB.W;

    // std::vector<PybindMiniBatch*> miniBatch;
    // std::reference_wrapper<std::vector<PybindMiniBatch*>> miniBatch_ref{miniBatch};
    // miniBatch.push_back(&pybindMB);

    // bool policyType = 1;
    // auto locals = py::dict("miniBatch"_a=miniBatch_ref, "policyType"_a=policyType);


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





    // //Compute policy and value on most recent element of the sequence.
    // const Rvec output = NET->forward(agent);
    // auto pol = prepare_policy<Policy_t>(output);
    // Rvec mu = pol.getVector(); // vector-form current policy for storage

    // // if explNoise is 0, we just act according to policy
    // // since explNoise is initial value of diagonal std vectors
    // // this should only be used for evaluating a learned policy
    // const bool bSamplePolicy = settings.explNoise>0 && agent.trackSequence;
    // auto act = pol.finalize(bSamplePolicy, &generators[nThreads+agent.ID], mu);
    // const auto adv = prepare_advantage<Advantage_t>(output, &pol);
    // const Real advantage = adv.computeAdvantage(pol.sampAct);
    // EP.action_adv.push_back(advantage);
    // EP.state_vals.push_back(output[VsID]);
    // agent.act(act);
    // data_get->add_action(agent, mu);

    // std::vector<Sequence*> seqVec;
    // seqVec.push_back(traj);

    // if(seqVec[0]->actions.size() == 0)
    // {
    //   std::cout << "PYTORCH: No action taken, selecting DONE 222 !" << std::endl;
    //   Rvec fakeAction = Rvec(aInfo.dim, 0);
    //   Rvec mu(policyVecDim, 0);
    //   agent.act(fakeAction);
    //   data_get->add_action(agent, mu);
    //   py::print("DONE 2!");
    //   std::cout << "DONE 222 !" << std::endl;

    // }

    // std::cout << "TRY THIS: " << std::endl;
    // std::cout << seqVec.size() << std::endl;
    // std::cout << seqVec[0]->states.size() << std::endl;
    // std::cout << seqVec[0]->actions.size() << std::endl;
    // std::cout << seqVec[0]->states[0].size() << std::endl;
    // std::cout << seqVec[0]->actions[0].size() << std::endl;
    // // std::cout << seqVec[0]->actions[0][0] << std::endl;
    // std::cout << "WORKED: " << std::endl;







    // py::print("PYTORCH: Agent selecting action!");

    // // std::vector<memReal> state = traj->states[0];
    // // std::cout << "State[0] = " << state[0] << std::endl;
    // // std::cout << "state.size() = " << state.size() << std::endl;
    // // std::cout << "traj->states[0][0] = " << traj->states[0][0] << std::endl;
    // // std::cout << "traj->states[0].size() = " << traj->states[0].size() << std::endl;
    // // std::cout << "traj->actions[0].size() = " << traj->actions[0].size() << std::endl;


    // py::print("PYTORCH: TRYING TO FEED NET");

    // // auto torch = py::module::import("torch");

    // // auto input = torch.attr("randn")(sInfo.dimUsed);
    // // auto action = Nets[0].attr("forward")(input);

    // // auto action = Nets[0].attr("forward")(state);

    // py::module::import("pybind11_embed");  // <--- Important!

    // {
    // // PybindSequence s1;
    // // s1.states.push_back(1.0);
    // // s1.states.push_back(2.0);
    // // s1.states.push_back(3.0);
    // // s1.states.push_back(4.0);
    // // s1.states.push_back(5.0);
    // // PybindSequence s2;
    // // s2.states.push_back(11.0);
    // // s2.states.push_back(12.0);
    // // s2.states.push_back(13.0);
    // // s2.states.push_back(14.0);
    // // s2.states.push_back(15.0);

    // // s1.actions.push_back(0.0);
    // // s1.actions.push_back(0.0);
    // // s1.actions.push_back(0.0);

    // // s2.actions.push_back(0.0);
    // // s2.actions.push_back(0.0);
    // // s2.actions.push_back(0.0);

    // // std::vector<PybindSequence*> seqVec;
    // // seqVec.push_back(&s1);
    // // seqVec.push_back(&s2);

    // // std::reference_wrapper<std::vector<PybindSequence*>> seqVec_ref{seqVec};
    // // bool policyType = 1;

    // // // auto locals = py::dict("var1"_a=seqVec_ref, "var2"_a=seqVec_ref);
    // // auto locals = py::dict("seqVec"_a=seqVec_ref, "policyType"_a=policyType);


    // // py::print("PROP1:");
    // // // std::vector<double> temp(input_dim);
    // // auto output = Nets[0].attr("forward")(locals);


    // // py::print("DID IT MODIFY THE C++ ?:");
    // // std::cout << s1.states[0] << std::endl;

    // // py::print("s1.actions:");
    // // std::cout << s1.actions[0] << std::endl;
    // // std::cout << s1.actions[1] << std::endl;
    // // std::cout << s1.actions[2] << std::endl;
    // // py::print("OUTPUT:");
    // // // std::cout << output << std::endl;



    // std::reference_wrapper<std::vector<Sequence*>> seqVec_ref{seqVec};
    // bool policyType = 1;

    // // auto locals = py::dict("var1"_a=seqVec_ref, "var2"_a=seqVec_ref);
    // auto locals = py::dict("seqVec"_a=seqVec_ref, "policyType"_a=policyType);


    // // py::print("PROP1:");
    // // std::vector<double> temp(input_dim);
    // // auto output = Nets[0].attr("forward")(locals);


    // // py::print("DID IT MODIFY THE C++ ?:");
    // // std::cout << s1.states[0] << std::endl;


    // }


    // py::print("PYTORCH: PROPAGATION TEST FINISHED.");



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


    Rvec fakeAction = Rvec(aInfo.dim(), 0);
    Rvec mu(aInfo.dimPol(), 0);
    agent.act(fakeAction);
    data_get->add_action(agent, mu);
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
