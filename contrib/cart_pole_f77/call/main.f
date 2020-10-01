      program main
      implicit none
      external smarties_setStateActionDims
      integer*8 :: smarties_comm
      integer :: STATE_SIZE
      integer :: NUM_ACTIONS
      integer :: AGENT_ID
      state_size = 6
      num_actions = 1
      agent_id = 0
      call smarties_setStateActionDims(smarties_comm, STATE_SIZE,
     + NUM_ACTIONS, AGENT_ID)
      END program main
