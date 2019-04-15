//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Panagiotis Hadjidoukas.
//

#pragma once

#include <cassert>
#include <dirent.h>

#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <limits>

inline MPI_Comm MPICommDup(const MPI_Comm C)
{
  MPI_Comm ret;
  MPI_Comm_dup(C, &ret);
  return ret;
}
inline Uint MPICommSize(const MPI_Comm C)
{
  int size;
  MPI_Comm_size(C, &size);
  return (Uint) size;
}
inline Uint MPICommRank(const MPI_Comm C)
{
  int rank;
  MPI_Comm_rank(C, &rank);
  return (Uint) rank;
}

inline std::vector<std::string> split(const std::string &s, const char delim)
{
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> tokens;
  while (getline(ss, item, delim)) tokens.push_back(item);
  return tokens;
}

inline int redirect_stdout_stderr()
{
  fflush(0);
  char output[256];
  sprintf(output, "output");
  int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  dup2(fd, 1);    // make stdout go to file
  dup2(fd, 2);    // make stderr go to file
  close(fd);      // fd no longer needed
  return fd;
}

inline void redirect_stdout_init(std::pair<int, fpos_t>& currOutputFD,
                                 unsigned iter)
{
  fflush( stdout);
  fgetpos(stdout, & currOutputFD.second);
  currOutputFD.first = dup( fileno( stdout ) );
  char buf[500];
  int wrank = getRank(MPI_COMM_WORLD);
  sprintf(buf, "output_%03d_%05u", wrank, iter);
  freopen(buf, "w", stdout); // put stdout onto buf
}

inline void redirect_stdout_finalize(std::pair<int, fpos_t>& currOutputFD)
{
  dup2( currOutputFD.first, fileno(stdout));
  close(currOutputFD.first);
  clearerr(stdout);
  fsetpos( stdout, & currOutputFD.second);
}

inline int parse2(char *line, char **argv)
{
  int argc = 0;

  while (*line != '\0') // if not the end of line
  {
    // replace white spaces with 0
    while (*line == ' ' || *line == '\t' || *line == '\n') *line++ = '\0';
    *argv++ = line; // save the argument position

    if (*line != '\0' && *line != ' ' && *line != '\t' && *line != '\n')
      argc++;

    while (*line != '\0' && *line != ' ' &&
           *line != '\t' && *line != '\n') line++; // skip the argument
  }
  *argv = NULL;//'\0';   /* mark the end of argument list */

  return argc;
}

inline int cp(const char *from, const char *to)
{
  int saved_errno;
  int fd_to = -1;
  int fd_from = open(from, O_RDONLY);
  if (fd_from < 0) return -1;

  struct stat sb;
  fstat(fd_from, &sb);
  if (S_ISDIR(sb.st_mode)) goto out_error; // It is a directory

  fd_to = open(to, O_WRONLY | O_CREAT | O_EXCL, sb.st_mode);
  if (fd_to < 0) goto out_error; // unable to open

  char buf[4096];
  ssize_t nread;
  while (nread = read(fd_from, buf, sizeof buf), nread > 0)
  {
    char *out_ptr = buf;
    while (nread > 0)
    {
      ssize_t nwritten = write(fd_to, out_ptr, nread);
      if (nwritten >= 0)
      {
        nread -= nwritten;
        out_ptr += nwritten;
      }
      else if (errno != EINTR) goto out_error;
    }
  }

  if (nread == 0)
  {
    if (close(fd_to) < 0) {
      fd_to = -1;
      goto out_error;
    } else {  // peh: added due to some issues on monte rosa
      fsync(fd_to);
    }
    close(fd_from);
    return 0; // Success!
  }

  out_error: saved_errno = errno;

  close(fd_from);
  if (fd_to >= 0) close(fd_to);

  errno = saved_errno;
  return -1;
}

inline int copy_from_dir(const std::string name)
{
  DIR * dir = opendir (name.c_str());
  if (dir != NULL)
  {
    // print all the files and directories within directory
    struct dirent * ent;
    while ((ent = readdir (dir)) != NULL)
    {
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
