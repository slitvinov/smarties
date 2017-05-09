#include <boost/python.hpp>
#include "Communicator.h"
using namespace boost::python;

BOOST_PYTHON_MODULE(Communicator)
{
    class_<Communicator>("Communicator", init<int, int, int>())
        .def("sendState", &Communicator::sendState)
        .def("recvState", &Communicator::recvState)
    ;
}
