!
!  cart_pole.f90
!  cart-pole
!
!  Created by Jacopo Canton on 31/01/19
!  Copyright (c) 2019 Jacopo Canton. All rights reserved.
!
!  This file contains the cart pole test case for Smarties in Fortran.
!   - The first module, `class_cartPole`, defines the environment and the
!   simulation parameters.
!   - The second module, `fortran_smarties`, contains the interface to the C++
!   code as well as the main function `fortran_app_main` called by Smarties.
!
!==============================================================================


!==============================================================================

module fortran_smarties
  ! Module interfacing with Smarties

  implicit none

  include 'mpif.h'

  interface
    ! Interfaces to the C++ side which send and receive data from Smarties.
    ! There should be no need to edit these unless Smarties is changed.
    subroutine rlcomm_update_state_action_dims(rlcomm, sdim, adim) &
        bind(c, name='rlcomm_update_state_action_dims')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: rlcomm
      integer(c_int),  intent(in), value :: sdim
      integer(c_int),  intent(in), value :: adim
    end subroutine rlcomm_update_state_action_dims

    subroutine rlcomm_set_action_scales(rlcomm, upper_action_bound, lower_action_bound, array_len, bounded) &
        bind(c, name='rlcomm_set_action_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: rlcomm
      type(c_ptr),     intent(in), value :: upper_action_bound, lower_action_bound
      integer(c_int),  intent(in), value :: array_len
      logical(c_bool), intent(in), value :: bounded
    end subroutine rlcomm_set_action_scales

    subroutine rlcomm_set_state_scales(rlcomm, upper_state_bound, lower_state_bound, array_len) &
        bind(c, name='rlcomm_set_state_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: rlcomm
      type(c_ptr),     intent(in), value :: upper_state_bound, lower_state_bound
      integer(c_int),  intent(in), value :: array_len
    end subroutine rlcomm_set_state_scales

    subroutine rlcomm_set_state_observable(rlcomm, b_observable, array_len) &
        bind(c, name='rlcomm_set_state_observable')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: b_observable
      integer(c_int), intent(in), value :: array_len
    end subroutine rlcomm_set_state_observable

    subroutine rlcomm_sendInitState(rlcomm, state, array_len) &
        bind(c, name='rlcomm_sendInitState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: array_len
    end subroutine rlcomm_sendInitState

    subroutine rlcomm_sendTermState(rlcomm, state, array_len, reward) &
        bind(c, name='rlcomm_sendTermState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: array_len
      real(c_double), intent(in), value :: reward
    end subroutine rlcomm_sendTermState

    subroutine rlcomm_sendState(rlcomm, state, array_len, reward) &
        bind(c, name='rlcomm_sendState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: rlcomm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: array_len
      real(c_double), intent(in), value :: reward
    end subroutine rlcomm_sendState

    subroutine rlcomm_recvAction(rlcomm, action, array_len) &
        bind(c, name='rlcomm_recvAction')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr), intent(in), value :: rlcomm
      type(c_ptr), intent(in), value :: action
      integer(c_int), intent(in), value :: array_len
    end subroutine rlcomm_recvAction

  end interface

end module fortran_smarties
