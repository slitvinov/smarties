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

int parse2(char *line, char **argv)
{
  int argc = 0;

  while (*line != '\0') {         /* if not the end of line ....... */
    while (*line == ' ' || *line == '\t' || *line == '\n')
      *line++ = '\0';         /* replace white spaces with 0 */
    *argv++ = line;         /* save the argument position */

    if (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n')
      argc++;

    while (*line != '\0' && *line != ' ' &&
        *line != '\t' && *line != '\n')
      line++; /* skip the argument until ...*/
  }
  *argv = NULL;//'\0';   /* mark the end of argument list */

  return argc;
}

int cp(const char *from, const char *to)
{
  int fd_to, fd_from;
  char buf[4096];
  ssize_t nread;
  int saved_errno;
  struct stat sb;

  fd_from = open(from, O_RDONLY);
  if (fd_from < 0)
    return -1;

  fstat(fd_from, &sb);
  if (S_ISDIR(sb.st_mode)) {  /* more supported than DT_REG */
    //printf("It is a directory!\n");
    fd_to = -1;
    goto out_error;
  }

  fd_to = open(to, O_WRONLY | O_CREAT | O_EXCL, sb.st_mode);
  if (fd_to < 0)
    goto out_error;

  while (nread = read(fd_from, buf, sizeof buf), nread > 0) {
    char *out_ptr = buf;
    ssize_t nwritten;

    do {
      nwritten = write(fd_to, out_ptr, nread);
      if (nwritten >= 0) {
        nread -= nwritten;
        out_ptr += nwritten;
      }
      else if (errno != EINTR) {
        goto out_error;
      }
    } while (nread > 0);
  }

  if (nread == 0) {
    if (close(fd_to) < 0) {
      fd_to = -1;
      goto out_error;
    }
    else {  // peh: added due to some issues on monte rosa
      fsync(fd_to);
    }
    close(fd_from);

    /* Success! */
    return 0;
  }

  out_error:
  saved_errno = errno;

  close(fd_from);
  if (fd_to >= 0)
    close(fd_to);

  errno = saved_errno;
  return -1;
}

int copy_from_dir(const std::string name)
{
  DIR *dir;
  struct dirent *ent;
  //struct stat sb;

  dir = opendir (name.c_str());
  if (dir != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      //if (ent->d_type == DT_REG) {
      //printf ("%s (%d)\n", ent->d_name, ent->d_type);
      char source[1025], dest[1026];

      sprintf(source, "%s/%s", name.c_str(), ent->d_name);
      sprintf(dest, "./%s", ent->d_name);
      cp(source, dest);
      //}

    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("oops!");
    return 1;
  }
  return 0;
}

void comm_sock(int fd, const bool bsend, double*const data, const int size)
{
  int bytes = bsend ? send_all(fd, data, size) : recv_all(fd, data, size);

  if (bytes <= 0)
  {
    printf("Lost contact with smarties, aborting..\n");
    fflush(0);
    abort();
  }
}

#ifdef MPI_INCLUDED
//these commands only work to send to master, matching command is defined in Scheduler.cpp
//this is messy, expect changes
void recv_MPI(double*const data, const int size, const MPI_Comm comm, unsigned long &wait)
{
  assert(comm != MPI_COMM_NULL);
  MPI_Request request;
  MPI_Irecv(data, size, MPI_BYTE, 0, 0, comm, &request);
  int cnt=0;
  while(true) {
    usleep(wait);
    cnt++;
    int completed=0;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    if (completed) break;
  }
  //avoid wasting cpu for communication if MPI implementation does busy-wait
  //note that non-MPI send and recv usually already have a yield-wait policy
  wait = std::max(static_cast<double>(wait)+std::floor((cnt-10)/10.), 1.1);
}

void send_MPI(double*const data, const int size, const MPI_Comm comm)
{
  assert(comm != MPI_COMM_NULL);
  MPI_Request dummyreq;
  MPI_Isend(data, size, MPI_BYTE, 0, 1, comm, &dummyreq);
  MPI_Request_free(&dummyreq); //Not my problem?
}
#endif
