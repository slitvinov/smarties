//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include <sys/un.h>
#include <cassert>

inline int SOCKET_Irecv(void* const buffer,
                        const unsigned size,
                        const int socketid,
                        unsigned* const request
)
{
  char* const charbuf = (char*) buffer;
  unsigned& received_size = * request;
  const unsigned to_receive = size - received_size;
  char* const recv_buffer = charbuf + received_size;
  assert(received_size <= size);
  if(to_receive == 0) return 0;
  const int bytesrecv = recv(socketid, recv_buffer, to_receive, MSG_DONTWAIT);
  if(bytesrecv >= 0)
  {
    received_size += bytesrecv;
    assert(received_size <= size);
    return 0;
  } else return -1;
}

inline int SOCKET_Isend(const void* const buffer,
                        const unsigned size,
                        const int socketid,
                        unsigned* const request
)
{
  const char* const charbuf = (const char*) buffer;
  unsigned& sent_size = * request;
  const unsigned to_send = size - sent_size;
  const char* const send_buffer = charbuf + sent_size;
  assert(sent_size <= size);
  if(to_send == 0) return 0;
  const int bytessent = recv(socketid, send_buffer, to_send, MSG_DONTWAIT);
  if(bytessent >= 0)
  {
    sent_size += bytessent;
    assert(sent_size <= size);
    return 0;
  } else return -1;
}

inline int SOCKET_Brecv(void* const buffer,
                        const unsigned size,
                        const int socketid)
{
  unsigned bytes_to_receive = size;
  char* pos = (char*) buffer;
  while (bytes_to_receive > 0)
  {
    const int bytesrecv = recv(socketid, pos, bytes_to_receive, 0);
    assert(bytesrecv <= bytes_to_receive);
    if( bytesrecv >= 0 ) {
      bytes_to_receive -= bytesrecv;
      pos += bytesrecv;
    }
    else return -1;
  }
  return 0;
}

inline int SOCKET_Recv(void* const buffer,
                       const unsigned size,
                       const int socketid)
{
  unsigned request = 0;
  while (1)
  {
    const int err = SOCKET_Irecv(buffer, size, socketid, &request);
    if(err) return err;
    if(request >= size) return 0;
    usleep(1); // wait for master without burning a cpu
  }
}

inline int SOCKET_Bsend(const void*const buffer,
                        const unsigned size,
                        const int socketid)
{
  unsigned bytes_to_send = size;
  char* pos = (char*)buffer;
  while ( bytes_to_send > 0 )
  {
    const int bytessent = send(socketid, pos, bytes_to_send, 0);
    assert(bytessent <= bytes_to_send);
    if( bytessent >= 0 ) {
      bytes_to_send -= bytessent;
      pos += bytessent;
    }
    else return -1;
  }
  return 0;
}

inline int SOCKET_Send(const void*const buffer,
                       const unsigned size,
                       const int socketid)
{
  unsigned request = 0;
  while (1)
  {
    const int err = SOCKET_Isend(buffer, size, socketid, &request);
    if(err) return err;
    if(request >= size) return 0;
    usleep(1); // wait for master without burning a cpu
  }
}

inline int SOCKET_clientConnect()
{
  // Specify the socket
  char SOCK_PATH[] = "../smarties_AFUNIX_socket_FD";

  int serverAddr;
  if( ( serverAddr = socket(AF_UNIX, SOCK_STREAM, 0) ) == -1 )
  {
    printf("SOCKET_clientConnect::socket failed"); fflush(0); abort();
  }
  int _TRU = 1;
  if( setsockopt(serverAddr, SOL_SOCKET, SO_REUSEADDR, &_TRU, sizeof(int)) < 0 )
  {
    printf("SOCKET_clientConnect::setsockopt failed\n"); fflush(0); abort();
  }

  // Specify the server
  struct sockaddr_un serverAddress;
  bzero((char *)&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path)+1;

  // Connect to the server
  size_t nAttempts = 0;
  while (connect(serverAddr, (struct sockaddr *)&serverAddress, servlen) < 0)
  {
    if(++nAttempts % 1000 == 0) {
      printf("Application is taking too much time to connect to smarties."
             " If your application needs to change directories (e.g. set up a"
             " dedicated directory for each run) it should do so AFTER"
             " the connection to smarties has been initialzed.\n");
    }
    usleep(1);
  }

  return serverAddr;
}

inline int SOCKET_serverConnect(const unsigned nClients,
                             std::vector<int>& clientSockets)
{
  // Specify the socket
  char SOCK_PATH[] = "smarties_AFUNIX_socket_FD";
  unlink(SOCK_PATH);

  int serverAddr;
  if ( ( serverAddr = socket(AF_UNIX, SOCK_STREAM, 0) ) == -1 ) {
    printf("SOCKET_serverConnect::socket failed"); fflush(0); abort();
  }

  struct sockaddr_un serverAddress;
  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  //this printf is to check that there is no funny business with trailing 0s:
  //printf("%s %s\n",serverAddress.sun_path, SOCK_PATH); fflush(0);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path) +1;

  if ( bind(serverAddr, (struct sockaddr *)&serverAddress, servlen) < 0 ) {
    printf("SOCKET_serverConnect::bind failed"); fflush(0); abort();
  }

  if (listen(serverAddr, nOwnWorkers) == -1) { // liste
    printf("SOCKET_serverConnect::listen failed"); fflush(0); abort();
  }

  clientSockets.resize(nClients, 0);
  for(int i = 0; i<nClients; i++)
  {
    struct sockaddr_un clientAddress;
    unsigned int addr_len = sizeof(clientAddress);
    struct sockaddr*const addrPtr = (struct sockaddr*) &clientAddress;
    if( ( clientSockets[i] = accept(serverAddr, addrPtr, &addr_len) ) == -1 )
    {
      printf("SOCKET_serverConnect::accept failed"); fflush(0); abort();
    }
    printf("server: new connection on socket %d\n", clientSockets[i]);
  }

  return serverAddr;
}
