#!/usr/bin/env python
import sys, socket, os, os.path, time
import numpy as np

class Communicator:
    def start_server(self):
        #read from argv identifier for communication:
        sockid = sys.argv[1]
        server_address = "/tmp/smarties_sock"+str(sockid)
        try: os.unlink(server_address)
        except OSError:
            if os.path.exists(server_address): raise

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(server_address)
        server.listen(1)
        self.conn, addr = server.accept()
        #time.sleep(1)
        #conn = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        #conn.connect( server_address )

    def set_action_scales(self, upper, lower, bounded=False):
        actVals = np.zeros([0],dtype=np.float64)
        actOpts = np.zeros([0],dtype=np.float64)
        if bounded: bound=1.0
        else: bound=0.0;
        assert(len(upper) == self.nActions and len(lower) == self.nActions)
        for i in range(self.nActions):
            actOpts = np.append(actOpts, [2.1, bound])
            actVals = np.append(actVals, [lower[i], upper[i]])
        self.action_options = actOpts
        self.action_bounds = actVals

    def set_state_scales(self, upper, lower):
        obsBnds = np.zeros([0],dtype=np.float64)
        assert(len(upper) == self.nStates and len(lower) == self.nStates)
        for i in range(self.nStates):
            obsBnds = np.append(obsBnds, [upper[i], lower[i]])
        self.observation_bounds = obsBnds

    def set_state_observable(self, observable):
        self.obs_in_use = np.zeros([self.nStates],dtype=np.float64)
        assert(len(observable) == self.nStates)
        for i in range(self.nStates): self.obs_in_use[i] = observable[i]

    def __init__(self, state_components, action_components, number_of_agents=1, discrete_actions=False):
        assert(False)
        self.start_server()
        self.sent_stateaction_info = False
        self.discrete_actions = discrete_actions
        self.number_of_agents = number_of_agents
        self.nActions = action_components
        self.nStates =   state_components
        actVals = np.zeros([0],dtype=np.float64)
        actOpts = np.zeros([0],dtype=np.float64)
        obsBnds = np.zeros([0],dtype=np.float64)
        for i in range(action_components):
            #tell smarties unbounded (0) continuous actions
            actOpts = np.append(actOpts, [2.1, 0.])
            #tell smarties non-rescaling bounds -1 : 1
            actVals = np.append(actVals, [-1, 1])
        for i in range(state_components):
            #tell smarties non-rescaling bounds -1 : 1.
            obsBnds = np.append(obsBnds, [1, -1])
        self.obs_in_use = np.ones(state_components, dtype=np.float64)
        self.observation_bounds = obsBnds
        self.action_options = actOpts
        self.action_bounds = actVals
        self.seq_id, self.frame_id = 0, 0

    def __del__(self):
        self.conn.close()

    def send_stateaction_info(self):
        if(not self.sent_stateaction_info):
            sizes_ary = np.array([self.nStates+.1, self.nActions+.1, self.discrete_actions+.1, self.number_of_agents+.1],np.float64)
            print(sizes_ary); sys.stdout.flush()
            self.conn.send(sizes_ary.tobytes())
            self.conn.send(self.obs_in_use.tobytes())
            self.conn.send(self.observation_bounds.tobytes())
            self.conn.send(self.action_options.tobytes())
            self.conn.send(self.action_bounds.tobytes())
            self.bRender = np.frombuffer(self.conn.recv(8), dtype=np.float64)
            self.bRender = int(round(self.bRender[0]))
            #print(bRender); sys.stdout.flush()
            self.sent_stateaction_info = True

    def send_state(self, observation, reward=0, terminal=False, initial=False, agent_id = 0):
        if initial: self.seq_id, self.frame_id = self.seq_id+1, 0
        self.frame_id = self.frame_id + 1
        self.send_stateaction_info()
        assert(agent_id<self.number_of_agents)
        state = np.zeros(self.nStates+3, dtype=np.float64)
        state[0] = agent_id
        if terminal:  state[1] = 2.1
        elif initial: state[1] = 1.1
        else:         state[1] = 0.1
        if hasattr(observation, 'shape'):
            assert( self.nStates == observation.size )
            state[2:self.nStates+2] = observation.ravel()
        else:
            assert( self.nStates == 1 )
            state[2] = observation
        state[self.nStates+2] = reward
        for i in range(self.nStates+2): assert(not np.isnan(state[i]))
        #print(state); sys.stdout.flush()
        self.conn.send(state.tobytes())
        #if self.bRender==1 and self.gym is not None: self.gym.render()
        #if self.bRender==2 and self.gym is not None:
        #    seq_id, frame_id = 0, 0
        #    fname = 'state_seq%04d_frame%07d' % (seq_id, frame_id)
        #    plt.imshow(self.gym.render(mode='rgb_array'))
        #    plt.savefig(fname, dpi=100)
        if(terminal): self.recv_action()

    def recv_action(self):
        buf = np.frombuffer(self.conn.recv(self.nActions*8), dtype=np.float64)
        for i in range(self.nActions): assert(not np.isnan(buf[i]))
        if abs(buf[0]+256)<2.2e-16: quit()
        return buf
