#include "Communicator.h"
#include "Communicator_utils.cpp"
#include <iostream>
#include <string>
#include <fstream>
#define COMM_REDIRECT_OUT
//APPLICATION SIDE CONSTRUCTOR
using namespace std;

Communicator::Communicator(const int socket, const int sdim, const int adim)
{
	update_state_action_dims(sdim, adim);
	spawner = socket==0; // if app gets socket prefix 0, then it spawns smarties
	socket_id = socket;
	called_by_app = true;
	launch();
}

void Communicator::update_state_action_dims(const int sdim, const int adim)
{
	defined_spaces = true;
	nStates = sdim;
	nActions = adim;
	// agent number, initial/normal/terminal indicator, state,  reward
	size_state = (3+sdim)*sizeof(double);
	size_action = adim*sizeof(double);
	_dealloc(data_action);
	_dealloc(data_state);
	data_action = _alloc(size_action);
	data_state = _alloc(size_state);
}

#ifdef MPI_INCLUDED
//MPI APPLICATION SIDE CONSTRUCTOR
Communicator::Communicator(const int socket, const int sdim, const int adim, const MPI_Comm app)
{
	comm_inside_app = app;
	update_rank_size();
	update_state_action_dims(sdim, adim);
	spawner = socket==0; // if app gets socket prefix 0, then it spawns smarties
	socket_id = socket;
	called_by_app = true;
	if (rank_inside_app == 0) //only rank 0 of the app talks with smarties
		launch();
}
#endif

void Communicator::sendState(const int iAgent, const _AGENT_STATUS status,
		const std::vector<double> state, const double reward)
{
	if(rank_inside_app>0) return; //only rank 0 of the app sends state
	assert(state.size()==(std::size_t)nStates && data_state not_eq nullptr && iAgent>=0);

	intToDoublePtr(iAgent, data_state+0);
	intToDoublePtr(status, data_state+1);
	for (int j=0; j<nStates; j++) {
		data_state[j+2] = state[j];
		assert(not std::isnan(state[j]) && not std::isinf(state[j]));
	}
	data_state[nStates+2] = reward;
	assert(not std::isnan(reward) && not std::isinf(reward));

	if (logfile != std::string()) {
		if(verbose) printLog(data_state, size_state);
		else  printBuf(data_state, size_state);
	}

#ifdef MPI_INCLUDED
	if (rank_learn_pool>0) send_MPI(data_state, size_state, comm_learn_pool);
	else
#endif
		comm_sock(Socket, true, data_state, size_state);

	if (status == _AGENT_LASTCOMM) {
		//receive continue/abort
		if(rank_learn_pool<1) { //TODO: add continue command to master
			comm_sock(Socket, false, data_action, size_action);
			if(data_action[0]<0) {
				printf("Received end of training signal. Aborting...\n"); fflush(0);
#ifdef MPI_INCLUDED
				if(size_inside_app>0) MPI_Abort(comm_inside_app, 1);
				else
#endif
					abort();
			}
		}
		seq_id++;
		msg_id = 0;
	}
}

void Communicator::recvAction(std::vector<double>& actions)
{
	assert(actions.size() == (std::size_t) nActions);
#ifdef MPI_INCLUDED
	if(rank_inside_app <= 0)
	{
		if (rank_learn_pool>0)
			recv_MPI(data_action, size_action, comm_learn_pool, lag);
		else
#endif
			comm_sock(Socket, false, data_action, size_action);
#ifdef MPI_INCLUDED
		for (int i=1; i<size_inside_app; ++i)
			MPI_Send(data_action, size_action, MPI_BYTE, i, 42, comm_inside_app);
	} else {
		MPI_Recv(data_action, size_action, MPI_BYTE, 0, 42, comm_inside_app, MPI_STATUS_IGNORE);
	}
#endif

	for (int j=0; j<nActions; j++) {
		actions[j] = data_action[j];
		assert(not std::isnan(actions[j]) && not std::isinf(actions[j]));
	}

	if (logfile != std::string() && rank_inside_app <= 0) {
		if(verbose) printLog(data_action, size_action);
		else  printBuf(data_action, size_action);
	}
}

void Communicator::printLog(const double*const buf, const int size)
{
	std::ostringstream o;
	o << "seq_id:" << seq_id << " msg_id:"  << msg_id << " ";
	for (unsigned j=0; j<size/sizeof(double); j++) o << buf[j] << " ";
	const std::string fname = logfile+std::to_string(iter)+".txt";
	FILE * f = fopen(fname.c_str(), "a");
	if (f != NULL)
		fprintf(f, "%s\n", o.str().c_str());
	fclose(f);
}

void Communicator::printBuf(const double*const buf, const int size)
{
	const std::string fname = logfile+std::to_string(iter)+".raw";
	FILE * pFile = fopen (fname.c_str(), "ab");
	fwrite (buf, sizeof(double), size/sizeof(double), pFile);
	fclose (pFile);
}

void Communicator::launch_smarties()
{
#ifdef __Smarties_
	printf("launch_smarties\n"); fflush(0);
	abort();
#else
	//go up til a file runClient is found: shaky
	struct stat buffer;
	while(stat("runClient.sh", &buffer))
	{
		chdir("..");
		char cwd[1024];
		if (getcwd(cwd, sizeof(cwd)) != NULL)
			printf("Current working dir: %s\n", cwd);
		else perror("getcwd() error");
	}

	redirect_stdout_stderr();

	launch_exec("./runClient.sh");
#endif
	abort(); //if app returns: TODO
}

void Communicator::launch()
{
	assert(rank_inside_app<1);
	if (spawner && called_by_app)
	{ //cheap way to ensure multiple sockets can exist on same node
		struct timeval clock;
		gettimeofday(&clock, NULL);
		socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
	}
	sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock", socket_id);

	if (spawner) setupClient();
	else setupServer();
}

void Communicator::launch_exec(const std::string exec)
{
	printf("About to exec %s.... \n",exec.c_str());

system("../launchSim.sh");

    //char *myArray[92];
    //char **myArray;

    ifstream file("./argList.txt");

int count = 0;
string line;
    if(file.is_open()){
        while (getline(file, line))
                count++;
    } else {
                printf("could not open argList, aborting"); abort();
    }
printf("numLines = %d\n", count);

file.clear();
file.seekg(0);
const int vecLength = count+2;

string myArray[vecLength];
    if(file.is_open())
    {
        myArray[0] = exec;
        for(int i = 1; i < 90; ++i)
        {
            file >> myArray[i];
        }
    } else{
        printf("could not open argList, aborting"); abort();
    }
file.close();

myArray[count] = std::to_string(socket_id);
myArray[count+1] = '\0';

        char*argv[vecLength];
        for (int i = 0; i < vecLength; i++)
        {
            argv[i] = const_cast<char*>(myArray[i].c_str());
        }

        argv[vecLength-1] = NULL;

const int res = execvp(exec.c_str(), argv);

	/*const int res = execlp(exec.c_str(),
			exec.c_str(),
			std::to_string(socket_id).c_str(),
			NULL);*/
	//int res = execvp(*largv, largv);
	if (res < 0)
		fprintf(stderr,"Unable to exec file '%s'!\n", exec.c_str());
}

void Communicator::launch_app()
{
#ifdef __Smarties_
	mkdir(("simulation_"+std::to_string(socket_id)+"_"
			+std::to_string(iter)+"/").c_str(),
			S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	const int ret = chdir(("simulation_"+std::to_string(socket_id)+"_"
			+std::to_string(iter)+"/").c_str());

if(ret<0){
	printf("changeDir failed"); abort();
}

	redirect_stdout_stderr();

	launch_exec(execpath);
#else
	printf("launch_app\n");
	fflush(0);
#endif
	abort(); //if app returns: TODO
}

void Communicator::setupClient()
{
	unlink(SOCK_PATH);
	print();
	fflush(0);
	const int rf = fork();

	if (rf == 0) {  //child spawns server
		if (execpath == std::string())
			launch_smarties();
		else
			launch_app();
	} else {  //parent
		Socket = socket(AF_UNIX, SOCK_STREAM, 0);

		int _true = 1;
		if(setsockopt(Socket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0) {
			perror("Sockopt failed\n");
			exit(1);
		}
		printf("Created socket\n");
		fflush(0);
		/* Specify the server */
		bzero((char *)&serverAddress, sizeof(serverAddress));
		serverAddress.sun_family = AF_UNIX;
		strcpy(serverAddress.sun_path, SOCK_PATH);
		const int servlen = sizeof(serverAddress.sun_family)
											+ strlen(serverAddress.sun_path)+1;

		/* Connect to the server */
		while (connect(Socket, (struct sockaddr *)&serverAddress, servlen) < 0)
			usleep(1);

		printf("Connected to server\n");
		fflush(0);
	}
}

void Communicator::setupServer()
{
	if ((ServerSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
	{
		perror("socket");
		exit(1);
	}

	bzero(&serverAddress, sizeof(serverAddress));
	serverAddress.sun_family = AF_UNIX;
	strcpy(serverAddress.sun_path, SOCK_PATH);
	printf("%s %s\n",serverAddress.sun_path,SOCK_PATH);
	fflush(0);
	const int servlen = sizeof(serverAddress.sun_family)
										+ strlen(serverAddress.sun_path) +1;

	if (bind(ServerSocket, (struct sockaddr *)&serverAddress, servlen) < 0)
	{
		perror("bind");
		exit(1);
	}
	/*
	int _true = 1;
	if(setsockopt(ServerSocket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0)
	{
		perror("Sockopt failed\n");
		exit(1);
	}
	 */
	/* listen (only 1)*/
	if (listen(ServerSocket, 1) == -1)
	{
		perror("listen");
		exit(1);
	}

	unsigned int addr_len = sizeof(clientAddress);
	if((Socket=accept(ServerSocket,(struct sockaddr*)&clientAddress,&addr_len))==-1)
	{
		perror("accept");
		return;
	}
	else printf("selectserver: new connection from on socket %d\n", Socket);
	fflush(0);
}

Communicator::~Communicator()
{
	if (rank_learn_pool>0) {
		if (spawner)
			close(Socket);
		else
			close(ServerSocket);
	} //if with forked process paradigm

	if(data_state not_eq nullptr) free(data_state);
	if(data_action not_eq nullptr) free(data_action);
}

#ifdef __Smarties_
Communicator::Communicator(const MPI_Comm scom, const int socket, const bool spawn)
{
	spawner = spawn;
	socket_id = socket;
	comm_learn_pool = scom;
	update_rank_size();
}

void Communicator::getStateActionShape(
	std::vector<std::vector<double>>&action_values,
	std::vector<double>& state_upper_bound,
	std::vector<double>& state_lower_bound,
	std::vector<bool>&bounded)
{
	//unsigned long dummy = 1;
	double* sketchy_ptr;
	sketchy_ptr = _alloc(2*sizeof(double));
	if (rank_learn_pool==0)
		MPI_Recv(sketchy_ptr, 2*sizeof(double), MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
	else {
		comm_sock(Socket, false, sketchy_ptr, 2*sizeof(double));
		if (rank_learn_pool==1)
			MPI_Ssend(sketchy_ptr, 2*sizeof(double), MPI_BYTE, 0, 3, comm_learn_pool);
	}

	nStates  = doublePtrToInt(sketchy_ptr+0);
	nActions = doublePtrToInt(sketchy_ptr+1);

	assert(nStates>=0 && nActions>=0);
	state_upper_bound.resize(nStates);
	state_lower_bound.resize(nStates);
	action_values.resize(nActions);
	bounded.resize(nActions);
	_dealloc(sketchy_ptr);

	sketchy_ptr = _alloc(2*nStates*sizeof(double));
	if (rank_learn_pool==0)
		MPI_Recv(sketchy_ptr, 2*nStates*sizeof(double), MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
	else {
		comm_sock(Socket, false, sketchy_ptr, 2*nStates*sizeof(double));
		if (rank_learn_pool==1)
			MPI_Ssend(sketchy_ptr, 2*nStates*sizeof(double), MPI_BYTE, 0, 3, comm_learn_pool);
	}

	double* sketchier_ptr = sketchy_ptr;
	for(int i=0; i<nStates; i++) {
		state_upper_bound[i] = *sketchier_ptr++;
		state_lower_bound[i] = *sketchier_ptr++;
		//printf("%d %f %f\n",i,state_lower_bound[i], state_upper_bound[i]);
	}


	_dealloc(sketchy_ptr);

	sketchy_ptr = _alloc(2*nActions*sizeof(double));
	if (rank_learn_pool==0)
		MPI_Recv(sketchy_ptr, 2*nActions*sizeof(double), MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
	else {
		comm_sock(Socket, false, sketchy_ptr, 2*nActions*sizeof(double));
		if (rank_learn_pool==1)
			MPI_Ssend(sketchy_ptr, 2*nActions*sizeof(double), MPI_BYTE, 0, 3, comm_learn_pool);
	}

	sketchier_ptr = sketchy_ptr;
	int n_action_vals = 0;
	for(int i=0; i<nActions; i++) {
		n_action_vals += doublePtrToInt(sketchier_ptr);
		action_values[i].resize(doublePtrToInt(sketchier_ptr++));
		bounded[i] = doublePtrToInt(sketchier_ptr++);
	}
	_dealloc(sketchy_ptr);

	sketchy_ptr = _alloc(n_action_vals*sizeof(double));
	if (rank_learn_pool==0)
		MPI_Recv(sketchy_ptr, n_action_vals*sizeof(double), MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
	else {
		comm_sock(Socket, false, sketchy_ptr, n_action_vals*sizeof(double));
		if (rank_learn_pool==1)
			MPI_Ssend(sketchy_ptr, n_action_vals*sizeof(double), MPI_BYTE, 0, 3, comm_learn_pool);
	}

	sketchier_ptr = sketchy_ptr;
	for(int i=0; i<nActions; i++)
		for(unsigned j=0; j<action_values[i].size(); j++)
		{
			action_values[i][j] = *sketchier_ptr++;
			//printf("%d %u %f\n",i,j,action_values[i][j]); fflush(0);
		}

	_dealloc(sketchy_ptr);

	update_state_action_dims(nStates, nActions);
}

extern int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv);

int Communicator::recvStateFromApp()
{
	int bytes = recv_all(Socket, data_state, size_state);

	if (bytes <= 0)
	{
		if (bytes == 0) printf("socket %d hung up\n", Socket);
		else perror("(1) recv");
		close(Socket);

		intToDoublePtr(0, data_state+0);
		intToDoublePtr(_AGENT_FAILCOMM, data_state+1);
		iter++;
	}
	else assert(bytes == size_state);

	if(comm_learn_pool != MPI_COMM_NULL)
		send_MPI(data_state, size_state, comm_learn_pool);

	return bytes <= 0;
}

int Communicator::sendActionToApp()
{
	//printf("I think im sending action %f\n",data_action[0]);
	if(comm_learn_pool != MPI_COMM_NULL)
		recv_MPI(data_action, size_action, comm_learn_pool, lag);

	if(fabs(data_action[0]+256)<2.2e-16) return 1;
	
	send_all(Socket, data_action, size_action);
	return 0;
}

void Communicator::answerTerminateReq(const double answer)
{
	data_action[0] = answer;
 	//printf("I think im givign the goahead %f\n",data_action[0]);
	send_all(Socket, data_action, size_action);
}

void Communicator::restart(std::string fname)
{
	int wrank = getRank(MPI_COMM_WORLD);
	FILE * f = fopen(("comm_"+to_string(wrank)+".status").c_str(), "r");
	if (f == NULL) return;
	{
		long unsigned ret = 0;
		fscanf(f, "sim number: %lu\n", &ret);
		iter = ret;
		printf("sim number: %lu\n", iter);
	}
	{
		long unsigned ret = 0;
		fscanf(f, "message ID: %lu\n", &ret);
		msg_id = ret;
		printf("message ID: %lu\n", msg_id);
	}
	{
		int ret = -1;
		fscanf(f, "socket  ID: %d\n", &ret);
		if(ret>=0) socket_id = ret;
		printf("socket  ID: %d\n", socket_id);
	}
	fclose(f);
}

void Communicator::save() const
{
	int wrank = getRank(MPI_COMM_WORLD);
	FILE * f = fopen(("comm_"+to_string(wrank)+".status").c_str(), "w");
	if (f != NULL)
	{
		fprintf(f, "sim number: %lu\n", iter);
		fprintf(f, "message ID: %lu\n", msg_id);
		fprintf(f, "socket  ID: %d\n", socket_id);
		fclose(f);
	}
	//printf( "sim number: %d\n", env->iter);
}

void Communicator::ext_app_run()
{
	char *largv[256];
	char line[1024];
	int largc = jobs_init(line, largv);

	char initd[256], newd[1024];
	getcwd(initd,256);
	assert(slaveGroup>=0 && rank_inside_app >= 0 && comm_inside_app != MPI_COMM_NULL);
	sprintf(newd,"%s/%s_%d_%lu",initd,"simulation",slaveGroup,iter);
	if(rank_inside_app == 0)
		mkdir(newd, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	MPI_Barrier(comm_inside_app);
	chdir(newd);	// go to the task private directory

	//copy any additional file
	if(!rank_inside_app)
		if (copy_from_dir("../bin") !=0 )  {
			printf("Error in copy from dir\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	MPI_Barrier(comm_inside_app);

	redirect_stdout_init();
	app_main(this, comm_inside_app, largc, largv);
	redirect_stdout_finalize();

	chdir(initd);	// go up one level
	iter++;
}

int Communicator::jobs_init(char *line, char **largv)
{
	FILE * cmdfp = fopen(paramfile.c_str(), "r");

	if (cmdfp == NULL)
		_die("Missing %s\n", paramfile.c_str());
	if(fgets(line, 1024, cmdfp)== NULL)
		_die("Empty %s\n",   paramfile.c_str());
	if (strstr(line,       paramfile.c_str()) == NULL)
		_die("Invalid %s\n", paramfile.c_str());

	fclose(cmdfp);

	return parse2(line, largv);
}

void Communicator::redirect_stdout_init()
{
	fflush(stdout);
	fgetpos(stdout, &pos);
	fd = dup(fileno(stdout));
	char buf[500];
	int wrank = getRank(MPI_COMM_WORLD);
	sprintf(buf, "output_%d_%lu",wrank,iter);
	freopen(buf, "w", stdout);
}

void Communicator::redirect_stdout_finalize()
{
	dup2(fd, fileno(stdout));
	close(fd);
	clearerr(stdout);
	fsetpos(stdout, &pos);        /* for C9X */
}
#endif


void Communicator::redirect_stdout_stderr()
{
#ifdef COMM_REDIRECT_OUT
	fflush(0);
	char output[256];
	sprintf(output, "output");
	fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	dup2(fd, 1);    // make stdout go to file
	dup2(fd, 2);    // make stderr go to file
	close(fd);      // fd no longer needed
#endif
}
