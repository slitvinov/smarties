//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Panagiotis Hadjidoukas.
//

//#include <chrono>
#include <sys/un.h>
/*************************************************************************/
/**************************   HELPER ROUTINES   **************************/
/*************************************************************************/

int SOCKET_Irecv(void* const buffer,
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

int SOCKET_Isend(const void* const buffer,
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

int SOCKET_Brecv(void* const buffer, const unsigned size, const int socketid)
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

int SOCKET_Recv(void* const buffer, const unsigned size, const int socketid)
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

int SOCKET_Bsend(const void*const buffer,const unsigned size,const int socketid)
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

int SOCKET_Send(const void*const buffer,const unsigned size,const int socketid)
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
    printf("Could not open directory %s\n",name.c_str());
    fflush(0);
    return 1;
  }
  return 0;
}
