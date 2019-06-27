#include <pybind11/pybind11.h>
#include "Core/Communicator.h"

namespace py = pybind11;

PYBIND11_MODULE(smarties, m)
{
  py::class_<smarties::Communicator>(m, "Communicator")

    .def(py::init<int stateDim, int actionDim, int number_of_agents> () )

    .def("sendInitState",
         & Communicator::sendInitState,
         "Send initial state of a new episode of agent # 'agentID'.",
         "state"_a, "agentID"_a = 0)

    .def("sendState",
         & Communicator::sendState,
         "Send normal state and reward for agent # 'agentID'.",
         "state"_a, "reward"_a, "agentID"_a = 0)

    .def("sendTermState",
         & Communicator::sendTermState,
         "Send terminal state and reward for agent # 'agentID'."
         " Note: V(s_terminal) = 0.",
         "state"_a, "reward"_a, "agentID"_a = 0)

    .def("sendLastState",
         & Communicator::sendLastState,
         "Send last state and reward of the episode for agent # 'agentID'."
         " Note that in general: V(s_last) != 0.",
         "state"_a, "reward"_a, "agentID"_a = 0)

    .def("recvAction",
         & Communicator::recvAction,
         "Get an action for agent # 'agentID' given previously sent state.",
         "agentID"_a = 0)

    .def("set_state_action_dims",
         & Communicator::set_state_action_dims,
         "Set dimensionality of state and action for agent # 'agentID'.",
         "dimState"_a, "dimAct"_a, "agentID"_a = 0)

    .def("set_action_scales",
         ( void (Communicator::*) (
            const std::vector<double>, const std::vector<double>,
            const bool, const int)
         ) & Communicator::set_action_scales,
         "Set lower and upper scale of the actions for agent # 'agentID'.",
         "upper_scale"_a, "lower_scale"_a, "areBounds"_a, "agentID"_a = 0)

    .def("set_action_scales",
         ( void (Communicator::*) (
            const std::vector<double>, const std::vector<double>,
            const std::vector<bool>, const int)
         ) & Communicator::set_action_scales,
         "Set lower and upper scale of the actions for agent # 'agentID'.",
         "upper_scale"_a, "lower_scale"_a, "areBounds"_a, "agentID"_a = 0)

    .def("set_action_options",
         ( void (Communicator::*) (const int, const int)
         ) & Communicator::set_action_options,
         "Set number of discrete control options for agent # 'agentID'.",
         "n_options"_a, "agentID"_a = 0)

    .def("set_action_options",
         ( void (Communicator::*) (const int, const std::vector<int>)
         ) & Communicator::set_action_options,
         "Set number of discrete control options for agent # 'agentID'.",
         "n_options"_a, "agentID"_a = 0)

    .def("set_state_observable",
         & Communicator::set_state_observable,
         "Set whether each state var is observed by the agent # 'agentID'.",
         "is_observable"_a, "agentID"_a = 0)

    .def("set_state_scales",
         & Communicator::sendState)

    .def("env_has_distributed_agents",
         & Communicator::sendState)

    .def("agents_define_different_MDP",
         & Communicator::sendState)

    .def("disableDataTrackingForAgents",
         & Communicator::disableDataTrackingForAgents)

    .def("isTraining",
         & Communicator::isTraining)

    .def("terminateTraining",
         & Communicator::terminateTraining)

    .def("desiredNepisodes",
         & Communicator::desiredNepisodes)

    .def("env_has_distributed_agents",
         & Communicator::env_has_distributed_agents);
}
