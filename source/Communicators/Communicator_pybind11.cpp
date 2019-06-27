#include <pybind11/pybind11.h>
#include "Communicators/Communicator.cpp"

namespace py = pybind11;

PYBIND11_MODULE(smarties, m)
{
  py::class_<smarties::Communicator>(m, "Communicator")

    .def(py::init<int, int, int> () )

    .def("sendInitState",
         & smarties::Communicator::sendInitState,
         py::arg("state"), py::arg("agentID") = 0,
         "Send initial state of a new episode for agent # 'agentID'.")

    .def("sendState",
         & smarties::Communicator::sendState,
         py::arg("state"), py::arg("reward"), py::arg("agentID") = 0,
         "Send normal state and reward for agent # 'agentID'.")

    .def("sendTermState",
         & smarties::Communicator::sendTermState,
         py::arg("state"), py::arg("reward"), py::arg("agentID") = 0,
         "Send terminal state and reward for agent # 'agentID'. "
         "NOTE: V(s_terminal) = 0 because episode cannot continue. "
         "For example, agent succeeded in task or is incapacitated.")

    .def("sendLastState",
         & smarties::Communicator::sendLastState,
         py::arg("state"), py::arg("reward"), py::arg("agentID") = 0,
         "Send last state and reward of the episode for agent # 'agentID'. "
         "NOTE: V(s_last) != 0 because it would be possible to continue the "
         "episode. For example, timeout not caused by the agent's policy.")

    .def("recvAction",
         & smarties::Communicator::recvAction,
         py::arg("agentID") = 0,
         "Get an action for agent # 'agentID' given previously sent state.")

    .def("set_state_action_dims",
         & smarties::Communicator::set_state_action_dims,
         py::arg("dimState"), py::arg("dimAct"), py::arg("agentID") = 0,
         "Set dimensionality of state and action for agent # 'agentID'.")

    .def("set_action_scales",
         ( void (smarties::Communicator::*) (
            const std::vector<double>, const std::vector<double>,
            const bool, const int) )
         & smarties::Communicator::set_action_scales,
         py::arg("upper_scale"), py::arg("lower_scale"),
         py::arg("areBounds"), py::arg("agentID") = 0,
         "Set lower and upper scale of the actions for agent # 'agentID'. "
         "Boolean arg specifies if actions are bounded between gien values.")

    .def("set_action_scales",
         ( void (smarties::Communicator::*) (
            const std::vector<double>, const std::vector<double>,
            const std::vector<bool>, const int) )
         & smarties::Communicator::set_action_scales,
         py::arg("upper_scale"), py::arg("lower_scale"),
         py::arg("areBounds"), py::arg("agentID") = 0,
         "Set lower and upper scale of the actions for agent # 'agentID'. "
         "Boolean arg specifies if actions are bounded between gien values.")

    .def("set_action_options",
         ( void (smarties::Communicator::*) (const int, const int) )
         & smarties::Communicator::set_action_options,
         py::arg("n_options"), py::arg("agentID") = 0,
         "Set number of discrete control options for agent # 'agentID'.")

    .def("set_action_options",
         ( void (smarties::Communicator::*) (const std::vector<int>,const int) )
         & smarties::Communicator::set_action_options,
         py::arg("n_options"), py::arg("agentID") = 0,
         "Set number of discrete control options for agent # 'agentID'.")

    .def("set_state_observable",
         & smarties::Communicator::set_state_observable,
         py::arg("is_observable"), py::arg("agentID") = 0,
         "For each state variable, set whether observed by agent # 'agentID'.")

    .def("set_state_scales",
         & smarties::Communicator::set_state_scales,
         py::arg("upper_scale"), py::arg("lower_scale"), py::arg("agentID") = 0,
         "Set upper & lower scaling values for the state of agent # 'agentID'.")

    .def("agents_define_different_MDP",
         & smarties::Communicator::agents_define_different_MDP,
         "Specify that each agent defines a different MPD (state/action/rew).")

    .def("disableDataTrackingForAgents",
         & smarties::Communicator::disableDataTrackingForAgents,
         py::arg("agentStart"), py::arg("agentEnd"),
         "Set agents whose experiences should not be used as training data.")

    .def("isTraining",
         & smarties::Communicator::isTraining,
         "Returns true if smarties is training, false if evaluating a policy.")

    .def("terminateTraining",
         & smarties::Communicator::terminateTraining,
         "Returns true if smarties is requesting application to exit.")

    .def("desiredNepisodes",
         & smarties::Communicator::desiredNepisodes,
         "Returns the number of state/action steps requested by smarties.");
}
