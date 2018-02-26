#pragma once
#ifndef UTIL
#define UTIL
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

    /*************************************************************************/
    /**************************   HELPER ROUTINES   **************************/
    /*************************************************************************/

    int recv_all(int fd, void *buffer, unsigned int size)
    {
        int result;
        unsigned int s=size;
        char *pos = (char*)buffer;
        

        do {
            result=recv(fd,pos,s,0);
            if((result!=-1)&&(result>0)) {
                s -= result;
                pos += result;
            }
            else 
                return result; /*-1;*/
        } while (s>0);
        //printf("recver %f\n",*((double*)buffer));
        return size;
    }

    int send_all(int fd, void *buffer, unsigned int size)
    {
        int result;
        unsigned int s=size;
        char *pos = (char*)buffer;
        
        do {
            result=send(fd,pos,s,0);
            if((result!=-1)&&(result>0)) {
                s -= result;
                pos += result;
            }
            else return result; /*-1;*/
        } while (s>0);
        return size;
    }
#endif