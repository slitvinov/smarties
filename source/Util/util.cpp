#include "util.h"

#ifndef MY_GETTIME
#define MY_GETTIME
double my_gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec*1.0E-6;
}
#endif

void parse(char *line, char **argv)
{
    while (*line != '\0') {		/* if not the end of line ....... */
        while (*line == ' ' || *line == '\t' || *line == '\n')
            *line++ = '\0';		/* replace white spaces with 0 */
        *argv++ = line;		/* save the argument position */
        while (*line != '\0' && *line != ' ' &&
               *line != '\t' && *line != '\n')
            line++;	/* skip the argument until ...*/
    }
    *argv = '\0';	/* mark the end of argument list */
}


int execute_cmd(int me, char *largv[], char *dir)
{
    int rf, res = 0;
    
    rf = fork();
    if (rf < 0) {
        printf("spanwer(%d): fork failed!!!!\n", me); fflush(0);
        return 1;
    }
    
    if (rf == 0) {
        if (dir != NULL)
            chdir(dir);	/* move to the specified directory */
        
        res = execvp(*largv, largv);
    }
    waitpid(rf, NULL, 0);
    return res;
}

int execute_cmd2(int me, char *largv[], char *dir)
{
    int rf, res = 0;
    
    rf = fork();
    if (rf < 0) {
        printf("spanwer(%d): fork failed!!!!\n", me); fflush(0);
        return 1;
    }
    
    if (rf == 0) {
        if (dir != NULL)
            chdir(dir);	/* move to the specified directory */
        
#if 1
        int fd = open("outputxxx", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        
        dup2(fd, 1);    /* make stdout go to file */
        dup2(fd, 2);    /* make stderr go to file */
        close(fd);	/* fd no longer needed - the dup'ed handles are sufficient */
#endif
        
        
        res = execvp(*largv, largv);
    }
    waitpid(rf, NULL, 0);
    return res;
}

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
    
    //printf("sender %f\n",*((double*)buffer));
    return size;
}