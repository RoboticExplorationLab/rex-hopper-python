"""
Copyright (C) 2020 Benjamin Bokser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np


class Gait:
    def __init__(self, controller, leg, t_p, phi_switch, hconst, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.t_p = t_p
        self.phi_switch = phi_switch
        self.dt = dt
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.init_angle = np.array([self.init_alpha, self.init_beta, self.init_gamma])
        self.controller = controller
        self.leg = leg
        self.x_last = None
        self.target = None
        self.hconst = hconst
        # TODO: should come from robotrunner.py
        self.r_save = np.array([0, 0, -self.hconst])
        self.target = np.hstack(np.append(np.array([0, 0, -self.hconst]), self.init_angle))

    def u(self, state, prev_state, r_in, r_d, delp, b_orient, fr_mpc, skip):

        if state == 'swing':
            # set target position
            self.target = np.hstack(np.append(np.array([0, 0, 0.5]), self.init_angle))
            # calculate wbc control signal
            u = -self.controller.wb_control(leg=self.leg, target=self.target, b_orient=b_orient, force=None)

        elif state == 'stance' or state == 'early':

            if prev_state != state and prev_state != 'early':
                # if contact has just been made, save that contact point as the new target to stay at
                # (stop following through with trajectory)
                self.r_save = r_in
            self.r_save = self.r_save - delp
            self.target = np.hstack(np.append(self.r_save, self.init_angle))
            self.target[2] = -self.hconst  # maintain height estimate at constant to keep ctrl simple

            if delp[2] <= 0 and self.leg.position()[2] >= -0.3 and skip is False:
                force = fr_mpc
            else:
                force = None

            u = -self.controller.wb_control(leg=self.leg, target=self.target, b_orient=b_orient, force=force)

        elif state == 'late':
            # calculate wbc control signal
            u = -self.controller.wb_control(leg=self.leg, target=self.target, b_orient=b_orient, force=None)

        else:
            u = None

        return u
