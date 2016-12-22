#pragma once

#include <sstream>
#include <sys/un.h>
#include <vector>

#define _AGENT_STATUS int
#define _AGENT_FIRSTCOMM 1
#define _AGENT_NORMCOMM  0
#define _AGENT_LASTCOMM  2
#define _AGENT_FAILCOMM  -1

class Communicator
{
    const int workerid, nActions, nStates, isServer;
    int CallerSocket, ListenerSocket, sizein, sizeout;
    std::ostringstream o;
    int msgID;

    char SOCK_PATH[256];
    struct sockaddr_un serverAddress;
    struct sockaddr_un clientAddress;
    double *datain, *dataout;

public:
    void setupServer();
    void setupClient(const int iter, const string execpath);
    void closeSocket();

    void dbg(double *x, int *pn);

    void sendState(int& agentId,
                   int& info,
                   std::vector<double>& state,
                   double& reward);
    void recvState(int& agentId,
                   int& info,
                   std::vector<double>& state,
                   double& reward);

    // assumption is that Action always follows a State:
    // no need to specify agent
    void recvAction(std::vector<double> & actions);
    void sendAction(std::vector<double> & actions);

    ~Communicator();
    Communicator(int _sockID, int _statedim, int _actdim, int _isserver);
};
