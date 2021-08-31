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
import simulationbridge
import leg
import wbc
import statemachine
import gait

import time
import sys

import transforms3d
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=np.nan)


def contact_check(c, c_s, c_prev, steps, con_c):
    if c_prev != c:
        con_c = steps  # timestep at contact change
        c_s = c  # saved contact value
    if con_c - steps <= 300:  # TODO: reset con_c
        c = c_s

    return c, c_s, con_c


class Runner:

    def __init__(self, dt=1e-3):

        self.dt = dt
        self.u = np.zeros(2)

        # height constant
        self.hconst = 0.3

        self.leg = leg.Leg(dt=dt)
        controller_class = wbc
        self.controller = controller_class.Control(dt=dt)
        self.simulator = simulationbridge.Sim(dt=dt)
        self.state = statemachine.Char()

        # gait scheduler values
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -self.hconst, self.init_alpha, self.init_beta, self.init_gamma])
        self.target = self.target_init[:]
        self.sh = 1  # estimated contact state
        self.dist_force = np.array([0, 0, 0])
        self.t_p = 1.4  # gait period, seconds 0.5
        self.phi_switch = 0.15  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.gait = gait.Gait(controller=self.controller, leg=self.leg, t_p=self.t_p, phi_switch=self.phi_switch,
                              hconst=self.hconst, dt=dt)

        # self.target = None
        self.r = np.array([0, 0, -self.hconst])  # initial footstep planning position

        # footstep planner values
        self.omega_d = np.array([0, 0, 0])  # desired angular acceleration for footstep planner
        self.pdot_des = np.array([0, 0, 0])  # desired body velocity in world coords
        self.force_control_test = False
        self.qvis_animate = False
        self.plot = True
        self.cycle = True
        self.closedform_invkin = False

    def run(self):

        steps = 0
        t = 0  # time
        p = np.array([0, 0, 0])  # initialize body position

        t0 = t  # starting time

        prev_state = str("init")

        time.sleep(self.dt)

        ct = 0
        s = 0
        c_prev = 0
        con_c = 0
        c_s = 0

        total = 3000  # number of timesteps to plot
        if self.plot:
            fig, axs = plt.subplots(1, 2, sharey=False)
            value1 = np.zeros((total, 3))
            value2 = np.zeros((total, 3))
        else:
            value1 = None
            value2 = None

        while 1:
            steps += 1
            t = t + self.dt
            # t_diff = time.clock() - t_prev
            # t_prev = time.clock()
            skip = False
            # run simulator to get encoder and IMU feedback
            # put an if statement here once we have hardware bridge too
            q, b_orient, c = self.simulator.sim_run(u=self.u)
            q[1] *= -1

            # enter encoder values into leg kinematics/dynamics
            self.leg.update_state(q_in=q)

            s_prev = s
            # gait scheduler
            s = self.gait_scheduler(t, t0)

            go, ct = self.gait_check(s, s_prev=s_prev, ct=ct, t=t)

            # Like using limit switches
            # if contact has just been made, freeze contact detection to True for 300 timesteps
            # protects against vibration/bouncing-related bugs
            c, c_s, con_c = contact_check(c, c_s, c_prev, steps, con_c)
            sh = int(c)

            c_prev = c

            # forward kinematics
            pos = np.dot(b_orient, self.leg.position())  # [:, -1])  TODO: Check

            pdot = np.array(self.simulator.v)  # base linear velocity in global Cartesian coordinates
            p = p + pdot * self.dt  # body position in world coordinates

            theta = np.array(transforms3d.euler.mat2euler(b_orient, axes='sxyz'))

            phi = np.array(transforms3d.euler.mat2euler(b_orient, axes='szyx'))[0]
            c_phi = np.cos(phi)
            s_phi = np.sin(phi)
            # rotation matrix Rz(phi)
            rz_phi = np.zeros((3, 3))
            rz_phi[0, 0] = c_phi
            rz_phi[0, 1] = s_phi
            rz_phi[1, 0] = -s_phi
            rz_phi[1, 1] = c_phi
            rz_phi[2, 2] = 1

            state = self.state.FSM.execute(s=s, sh=sh, go=go, pdot=pdot, leg_pos=self.leg.position())

            # TODO: Bring back footstep planner method to use this
            # if state is not 'stance' and prev_state is 'stance':
            #     self.r = self.footstep(rz_phi=rz_phi, pdot=pdot, pdot_des=self.pdot_des)

            omega = np.array(self.simulator.omega_xyz)

            x_in = np.hstack([theta, p, omega, pdot]).T  # array of the states for MPC
            x_ref = np.hstack([np.zeros(3), np.zeros(3), self.omega_d, self.pdot_des]).T  # reference pose (desired)
            # x_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T

            mpc_force = np.zeros(3)
            mpc_force[2] = 218  # TODO: Fix this

            delp = pdot*self.dt
            # calculate wbc control signal
            if self.cycle is True:
                self.u = self.gait.u(state=state, prev_state=prev_state, r_in=pos, r_d=self.r, delp=delp,
                                            b_orient=b_orient, fr_mpc=mpc_force, skip=skip)

            elif self.closedform_invkin is True:
                # TODO: could use an integral term due to friction
                self.u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * 2 + self.leg.dq * 0.15

            else:
                self.u = -self.controller.wb_control(leg=self.leg, target=self.target, b_orient=b_orient, force=None)

            prev_state = state

            if self.plot and steps <= total-1:
                value1[steps-1, :] = self.u[0]
                value2[steps-1, :] = self.u[1]

                if steps == total-1:
                    axs[0].plot(range(total-1), value1[:-1, 0], color='blue')
                    # axs[0, 1].plot(range(total-1), value1[:-1, 1], color='blue')
                    # axs[0, 2].plot(range(total-1), value1[:-1, 2], color='blue')
                    axs[1].plot(range(total-1), value2[:-1, 0], color='blue')
                    # axs[1, 1].plot(range(total-1), value2[:-1, 1], color='blue')
                    # axs[1, 2].plot(range(total-1), value2[:-1, 2], color='blue')
                    plt.show()

            # print(t, state)
            print(self.leg.position())
            # print("kin = ", self.leg.inv_kinematics(xyz=self.target[0:3]) * 180/np.pi)
            # print("encoder = ", self.leg.q * 180/np.pi)
            # sys.stdout.write("\033[F")  # back to previous line
            # sys.stdout.write("\033[K")  # clear line

    def gait_scheduler(self, t, t0):
        # Schedules gait, obviously
        # TODO: Add variable period
        phi = np.mod((t - t0) / self.t_p, 1)

        if phi > self.phi_switch:
            s = 0  # scheduled swing
        else:
            s = 1  # scheduled stance

        return s

    def gait_check(self, s, s_prev, ct, t):
        # To ensure that after state has been changed from stance to swing, it cannot change to "early" immediately...
        # Generates "go" variable as True only when criteria for time passed since gait change is met
        if s_prev != s:
            ct = t  # record time of gait change
        if ct - t >= self.t_p * (1 - self.phi_switch) * 0.5:
            go = True
        else:
            go = False

        return go, ct
