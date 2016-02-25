#pragma once
#define _XOPEN_SOURCE 700
#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <sys/time.h>
#include <ftw.h>
#include <sys/un.h>
#include <netdb.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>


#ifndef MY_GETTIME
#define MY_GETTIME
double my_gettime();
#endif

void parse(char *line, char **argv);

int execute_cmd(int me, char *largv[], char *dir);

int execute_cmd2(int me, char *largv[], char *dir);

int recv_all(int fd, void *buffer, unsigned int size);

int send_all(int fd, void *buffer, unsigned int size);