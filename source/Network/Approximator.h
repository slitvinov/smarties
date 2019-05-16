//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Approximator_h
#define smarties_Approximator_h

#include "Builder.h"
#include "ThreadContext.h"

namespace smarties
{

struct Aggregator;

struct Approximator
{
  //when this flag is true, specification of network properties is disabled:
  bool bCreatedNetwork = false;
public:
  void setNumberOfAddedSamples(const Uint nSamples = 0)
  {
    if(bCreatedNetwork) die("cannot modify network setup after it was built");
    m_numberOfAddedSamples = nSamples;
  }
  //specify type (and size) of auxiliary input
  void setAddedInput(const ADDED_INPUT type, Sint size = -1)
  {
    if(bCreatedNetwork) die("cannot modify network setup after it was built");
    if(type == NONE)
    {
      if(size>0) die("No added input must have size 0");
      if(auxInputNet) die("Given auxInputNet Approximator but specified no added inputyo");
      m_auxInputSize = 0;
    }
    else if (type == NETWORK)
    {
      if(not auxInputNet) die("auxInputNet was not given on construction");
      if(size<0) m_auxInputSize = auxInputNet->nOutputs();
      else {
        m_auxInputSize = size;
        if(auxInputNet->nOutputs() < (Uint) size)
          die("Approximator allows inserting the first 'size' outputs of "
              "another 'auxInputNet' Approximator as additional input (along "
              "with the state or the output of 'preprocessing' Approximator). "
              "But auxInputNet's output must be at least of size 'size'.");
      }
    }
    else if (type == ACTION || type == VECTOR)
    {
      if(size<=0) die("Did not specify size of the action/vector");
      m_auxInputSize = size;
    } else die("type not recognized");
    if(m_auxInputSize<0) die("m_auxInputSize cannot be negative at this point");
  }
  // specify whether we are using target networks
  void setUseTargetNetworks(const Sint targetNetworkSampleID = -1,
                            const bool bTargetNetUsesTargetWeights = true)
  {
    if(bCreatedNetwork) die("cannot modify network setup after it was built");
    m_UseTargetNetwork = true;
    m_bTargetNetUsesTargetWeights = bTargetNetUsesTargetWeights;
    m_targetNetworkSampleID = targetNetworkSampleID;
  }

  Builder buildFromSettings(const Uint outputSize) {
    return buildFromSettings( std::vector<Uint>(1, outputSize) );
  }
  Builder buildFromSettings(const std::vector<Uint> outputSizes)
  {
    Builder build(settings, distrib);
    const MDPdescriptor & MDP = replay->MDP;
    Uint inputSize = preprocessing? preprocessing->nOutputs()
                                  : (1+MDP.nAppendedObs) * MDP.dimStateObserved;
    if(auxInputNet && m_auxInputSize<=0) {
      assert(m_auxInputSize not_eq 0 && "Default is -1, what set it to 0?");
      m_auxInputSize = auxInputNet->nOutputs();
    }
    if(m_auxInputSize>0) inputSize += m_auxInputSize;
    build.stackSimple( inputSize, outputSizes );
    return build;
  }

  void initializeNetwork(Builder& build)
  {
    const MDPdescriptor & MDP = replay->MDP;
    net = build.build();
    opt = build.opt;
    std::vector<std::shared_ptr<Parameters>>& grads = build.threadGrads;
    assert(opt && net && grads.size() == nThreads);

    contexts.reserve(nThreads);
    #pragma omp parallel num_threads(nThreads)
    for (Uint i=0; i<nThreads; i++)
    {
      if(i == (Uint) omp_get_thread_num())
        contexts.emplace_back(i, grads[i], m_numberOfAddedSamples,
          m_UseTargetNetwork, m_bTargetNetUsesTargetWeights ? -1 : 0);
      #pragma omp barrier
    }

    agentsContexts.reserve(nAgents);
    for (Uint i=0; i<nAgents; i++) agentsContexts.emplace_back(i);

    const auto& layers = net->layers;
    if (m_auxInputSize>0) // If we have an auxInput (eg policy for DPG) to what
    {                     // layer does it attach? Then we can grab gradient.
      auxInputAttachLayer = 0; // preprocessing/state and aux in one input layer
      for(Uint i=1; i<layers.size(); i++) if(layers[i]->bInput) {
        if(auxInputAttachLayer>0) die("too many input layers, not supported");
        auxInputAttachLayer = i;
      }
      if (auxInputAttachLayer > 0) {
        if(layers[auxInputAttachLayer]->nOutputs() != auxInputNet->nOutputs())
          die("Size of layer to which auxInputNet does not match auxInputNet");
        if(preprocessing && layers[0]->nOutputs() != preprocessing->nOutputs())
          die("Mismatch in preprocessing output size and network input");
        const Uint stateInpSize = (1+MDP.nAppendedObs) * MDP.dimStateObserved;
        if(not preprocessing && layers[0]->nOutputs() != stateInpSize)
          die("Mismatch in state size and network input");
      }
      if(MDP.dimStateObserved > 0 and not layers[0]->bInput)
        die("Network does not have input layer.");
    }


    if (m_blockInpGrad or not preprocessing)
    {
      // Skip backprop to input vector or to preprocessing if 'm_blockInpGrad'
      // Three cases of interest:
      // 1) (most common) no aux input or both both preprocessing and aux input
      //    are given at layer 0 then block backprop at layer 1
      // 2) aux input given at layer greater than 1:  block backprop at layer 1
      // 3) aux input is layer 1 and layer 2 is joining (glue) layer, then
      //    gradient blocking is done at layer 3
      const Uint skipBackPropLayerID = auxInputAttachLayer==1? 3 : 1;
      if (auxInputAttachLayer==1) // check logic of statement 3)
        assert(layers[1]->bInput && not net->layers[2]->bInput);

      if (layers.size() > skipBackPropLayerID) {
        const Uint inputSize = preprocessing? preprocessing->nOutputs()
                             : (1+MDP.nAppendedObs) * MDP.dimStateObserved;
        if(auxInputAttachLayer==0) // check statement 1)
          assert(layers[1]->spanCompInpGrads == inputSize + m_auxInputSize);
        else if(auxInputAttachLayer==1) // check statement 3)
          assert(layers[3]->spanCompInpGrads == inputSize + m_auxInputSize);
        assert(layers[skipBackPropLayerID]->spanCompInpGrads >= inputSize);
        // next two lines actually tell the network to skip backprop to input:
        layers[skipBackPropLayerID]->spanCompInpGrads -= inputSize;
        layers[skipBackPropLayerID]->startCompInpGrads = inputSize;
      }
    }

    #ifdef __CHECK_DIFF //check gradients with finite differences
      net->checkGrads();
    #endif
    gradStats = new StatsTracker(net->getnOutputs(), distrib);
  }

  Uint nOutputs() const
  {
    return net->getnOutputs();
  }
  void updateGradStats(const std::string base, const Uint iter) const
  {
    gradStats->reduce_stats(base+name, iter);
  }

  Approximator(std::string name_, const Settings&S, const DistributionInfo&D,
               const std::shared_ptr<MemoryBuffer> replay_,
               const std::shared_ptr<Approximator> preprocessing_ = nullptr,
               const std::shared_ptr<Approximator> auxInputNet_ = nullptr) :
    settings(S), distrib(D), name(name_), replay(replay_),
    preprocessing(preprocessing_), auxInputNet(auxInputNet_)
  { }

  void load(const MiniBatch& B, const Uint batchID, const Sint wghtID) const
  {
    const Uint thrID = omp_get_thread_num();
    // ensure we allocated enough workspaces:
    assert(contexts.size()>thrID && threadsPerBatch.size()>batchID);
    ThreadContext&C = * contexts[thrID].get();
    threadsPerBatch[batchID] = thrID;
    assert(C.endBackPropStep(0)<0 && "Previous backprop did not finish?");
    C.load(net, B, batchID, wghtID);
    if(preprocessing) preprocessing->load(B, batchID, wghtID);
    if(auxInputNet) auxInputNet->load(B, batchID, wghtID);
  }

  void load(const MiniBatch& B, const Agent& agent, const Sint wghtID) const
  {
    assert(agentsContexts.size() > agent.ID);
    AgentContext & C = * agentsContexts[agent.ID].get();
    C.load(net, B, agent, wghtID);
    if(preprocessing) preprocessing->load(B, agent, wghtID);
    if(auxInputNet) auxInputNet->load(B, agent, wghtID);
  }

  template< typename contextid_t, typename val_t>
  void setAddedInput(const std::vector<val_t> addedInput,
                     const contextid_t& contextID,
                     const Uint t, Sint sampID = 0) const
  {
    assert(addedInput.size());
    getContext(contextID).addedInputVec(t, sampID) = NNvec( addedInput.begin(),
                                                            addedInput.end() );
  }
  template< typename contextid_t>
  void setAddedInputType(const ADDED_INPUT type,
                         const contextid_t& contextID,
                         const Uint t, Sint sampID = 0) const
  {
    getContext(contextID).addedInput(t, sampID) = type;
  }

  // forward: compute net output taking care also to gather additional required
  // inputs such as recurrent connections and auxiliary input networks.
  // It expects as input either the index over a previously loaded minibatch
  // or a previously loaded agent.
  template< typename contextid_t >
  Rvec forward(const contextid_t& contextID,
               const Uint t, Sint sampID = 0) const
  {
    const auto& C = getContext(contextID);
    if(sampID > (Sint) C.nAddedSamples) { sampID = 0; }
    if(C.activation(t, sampID)->written)
      return C.activation(t, sampID)->getOutput();
    const Uint ind = C.mapTime2Ind(t);
    // compute previous outputs if needed by recurrencies. limitation. what should
    // we do for target net / additional samples?
    // this next line assumes we want to use curr W and sample 0 for recurrencies:
    if(ind>0 && not C.net(t-1, 0)->written) forward(contextID, t-1, 0);
    //if(ind>0 && not C.net(t, samp)->written) forward(C, t-1, samp);
    const Activation* const recur = ind>0? C.activation(t-1, 0) : nullptr;
    const Activation* const activation = C.activation(t, sampID);
    const Parameters* const W = opt->getWeights(C.usedWeightID(sampID));
    //////////////////////////////////////////////////////////////////////////////
    NNvec INP;
    if(preprocessing)
    {
      INP = preprocessing->forward(contextID, t, sampID);
    } else INP = C.getState(t);

    if(C.addedInput(sampID) == NETWORK)
    {
      assert(auxInputNet not_eq nullptr);
      const NNvec addedinp = auxInputNet->forward(contextID, t, sampID);
      assert(addedinp.size());
      INP.insert(INP.end(), addedinp.begin(), addedinp.end());
      //if(!thrID) cout << "relay "<<print(addedinp) << endl;
    }
    else if(C.addedInput(sampID) == ACTION)
    {
      const ActionInfo & aI = replay->aI;
      const NNvec addedinp = aI.scaledAction2action<nnReal>( C.getAction(t) );
      INP.insert(INP.end(), addedinp.begin(), addedinp.end());
    }
    else if(C.addedInput(sampID) == VECTOR)
    {
      const auto& addedinp = C.addedInputVec(t, sampID);
      INP.insert(INP.end(), addedinp.begin(), addedinp.end());
    }
    assert(INP.size() == net->getnInputs());
    ////////////////////////////////////////////////////////////////////////////
    return net->predict(INP, recur, activation, W);
  }

  Rvec forward(const Agent& agent) const // run network for agent's recent step
  {
    const auto& C = getContext(agent);
    return forward(agent, C.episode->nsteps() - 1, 0);
  }

  void setGradient(      Rvec gradient,
                   const Uint batchID,
                   const Uint t, Sint sampID = 0) const
  {
    ThreadContext& C = getContext(batchID);
    if(sampID > (Sint) C.nAddedSamples) { sampID = 0; }
    //for(Uint i=0; i<grad.size(); i++) grad[i] *= PERW;
    gradStats->track_vector(gradient, C.threadID);
    const Sint ind = C.mapTime2Ind(t);
    //ind+1 because we use c-style for loops in other places:
    C.endBackPropStep(sampID) = std::max(C.endBackPropStep(sampID), ind+1);
    assert( C.activation(t, sampID)->written );
    if(ESpopSize > 1) debugL("Skipping backward because we use ES.");
    else C.activation(t, sampID)->addOutputDelta(gradient);
  }

  Rvec oneStepBackProp(const Rvec gradient,
                       const Uint batchID,
                       const Uint t, Sint sampID) const
  {
    assert(auxInputNet && "improperly set up the aux input net");
    assert(auxInputAttachLayer >= 0 && "improperly set up the aux input net");
    if(ESpopSize > 1) {
      debugL("Skipping relay_backprop because we use ES optimizers.");
      return Rvec(m_auxInputSize, 0);
    }
    ThreadContext& C = getContext(batchID);
    if(sampID > (Sint) C.nAddedSamples) { sampID = 0; }
    const Parameters* const W = opt->getWeights(C.usedWeightID(sampID));
    Activation* const A = C.activation(t, sampID);
    //const std::vector<Activation*>& act = series_tgt[thrID];
    //const int ind = mapTime2Ind(samp, thrID);
    //assert(act[ind]->written == true && relay not_eq nullptr);
    const Rvec ret = net->backPropToLayer(gradient, auxInputAttachLayer, A, W);
    //if(!thrID)
    //{
    //  const auto pret = Rvec(&ret[nInp], &ret[nInp+relay->nOutputs()]);
    //  const auto inp = act[ind]->getInput();
    //  const auto pinp = Rvec(&inp[nInp], &inp[nInp+relay->nOutputs()]);
    //  cout <<"G:"<<print(pret)<< " Inp:"<<print(pinp)<<endl;
    //}
    if(auxInputAttachLayer>0) return ret;
    else return Rvec(& ret[preprocessing->nOutputs()],
                     & ret[preprocessing->nOutputs() + m_auxInputSize]);
  }

  void backProp(const Uint batchID) const
  {
    ThreadContext& C = getContext(batchID);
    assert( C.endBackPropStep(samp) > 0 );

    if(ESpopSize > 1)
    {
      debugL("Skipping gradient because we use ES (derivative-free) optimizers.");
    }
    else
    {
      const auto& activations = C.activations;
      //loop over all the network samples, each may need different BPTT window
      for(Uint samp = 0; samp < activations.size(); ++samp)
      {
        const Sint last_error = C.endBackPropStep(samp);
        const auto& timeSeries = activations[samp];
        for (Sint i=0; i<last_error; ++i)
          assert(timeSeries[i]->written == true);

        const Parameters* const W = opt->getWeights(C.usedWeightID(samp));
        net->backProp(timeSeries, last_error, C.partialGradient.get(), W);

        //for(int i=0;i<last_error&&!thrID;i++)cout<<i<<" inpG:"<<print(act[i]->getInputGradient(0))<<endl;
        if(preprocessing and not m_blockInpGrad)
        {
          for(Sint k=0; k<last_error; ++k)
          {
            const Uint t = C.mapInd2Time(k);
            // assume that preprocessing is layer 0:
            Rvec inputGrad = C.activation(k, samp)->getInputGradient(0);
            // we might have added inputs, therefore trim those:
            inputGrad.resize(preprocessing->nOutputs());
            preprocessing->setGradient(inputGrad, batchID, t, samp);
          }
        }
        C.endBackPropStep(samp) = -1; //to stop additional backprops
      }
    }

    nAddedGradients++;
  }

  void prepareUpdate()
  {
    #ifndef NDEBUG
    for(const auto& C : contexts)
      for(const Sint todoBackProp : C->lastGradTstep)
        assert(todoBackProp<0 && "arrived into prepareUpdate() before doing backprop on all workspaces");
    #endif

    if(nAddedGradients==0) die("No-gradient update. Revise hyperparameters.");

    if(preprocessing and not m_blockInpGrad)
      for(Uint i=0; i<ESpopSize; i++) preprocessing->losses[i] += losses[i];

    opt->prepare_update(losses);
    losses = Rvec(ESpopSize, 0);
    reducedGradients = 1;
    nAddedGradients = 0;
  }

  bool ready2ApplyUpdate()
  {
    if(reducedGradients == 0) return true;
    else return opt->ready2UpdateWeights();
  }

  void applyUpdate()
  {
    if(reducedGradients == 0) return;

    opt->apply_update();
    reducedGradients = 0;
  }

  void getHeaders(std::ostringstream& buff) const
  {
    return opt->getHeaders(buff, name);
  }
  void getMetrics(std::ostringstream& buff) const
  {
    return opt->getMetrics(buff);
  }

  void save(const std::string base, const bool bBackup)
  {
    if(opt == nullptr) die("Attempted to save uninitialized net!");
    opt->save(base + name, bBackup);
  }
  void restart(const std::string base)
  {
    if(opt == nullptr) die("Attempted to restart uninitialized net!");
    opt->restart(base+name);
  }

private:
  const Settings& settings;
  const DistributionInfo & distrib;
  const std::string name;
  const Uint   nAgents =  distrib.nAgents,    nThreads =  distrib.nThreads;
  const Uint ESpopSize = settings.ESpopSize, batchSize = settings.batchSize;
  const std::shared_ptr<MemoryBuffer> replay;
  const std::shared_ptr<Approximator> preprocessing;
  const std::shared_ptr<Approximator> auxInputNet;
  Sint auxInputAttachLayer = -1;
  Sint m_auxInputSize = -1;
  Uint m_numberOfAddedSamples = 0;
  bool m_UseTargetNetwork = false;
  bool m_bTargetNetUsesTargetWeights = true;
  Sint m_targetNetworkSampleID = -1;
  // whether to backprop gradients in the input network.
  // work by DeepMind (eg in D4PG) indicates it's best to not propagate
  // policy net gradients towards input conv layers
  bool m_blockInpGrad = false;

  //const Aggregator* const relay;
  std::shared_ptr<Optimizer> opt = nullptr;
  std::shared_ptr<Network>   net = nullptr;

  mutable std::vector<Uint> threadsPerBatch = std::vector<Uint>(batchSize, -1);
  std::vector<std::unique_ptr<ThreadContext>> contexts;
  std::vector<std::unique_ptr< AgentContext>> agentsContexts;
  StatsTracker* gradStats = nullptr;

  mutable std::atomic<Uint> nAddedGradients{0};
  Uint reducedGradients=0;

  // For CMAES based optimization. Keeps track of total loss associate with
  // Each weight vector sample:
  mutable Rvec losses = Rvec(ESpopSize, 0);

  ThreadContext& getContext(const Uint batchID) const
  {
    assert(threadsPerBatch.size() > batchID);
    return * contexts[ threadsPerBatch[batchID] ].get();
  }
  AgentContext&  getContext(const Agent& agent) const
  {
    assert(agentsContexts.size() > agent.ID);
    return * agentsContexts[agent.ID].get();
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h


/*
Rvec forward(const Uint samp, const Uint thrID,
  const int USE_WGT, const int USE_ACT, const int overwrite=0) const;
inline Rvec forward(const Uint samp, const Uint thrID, int USE_ACT=0) const {
  assert(USE_ACT>=0);
  return forward(samp, thrID, thread_Wind[thrID], USE_ACT);
}
template<NET USE_A = CUR>
inline Rvec forward_cur(const Uint samp, const Uint thrID) const {
  const int indA = USE_A==CUR? 0 : -1;
  return forward(samp, thrID, thread_Wind[thrID], indA);
}
template<NET USE_A = TGT>
inline Rvec forward_tgt(const Uint samp, const Uint thrID) const {
  const int indA = USE_A==CUR? 0 : -1;
  return forward(samp, thrID, -1, indA);
}
// relay backprop requires gradients: no wID, no sorting based opt algos
Rvec relay_backprop(const Rvec grad, const Uint samp, const Uint thrID,
  const bool bUseTargetWeights = false) const;
void backward(Rvec grad, const Uint samp, const Uint thrID, const int USE_ACT=0) const;
void gradient(const Uint thrID, const int wID = 0) const;
void prepareUpdate();
void applyUpdate();
bool ready2ApplyUpdate();
*/
