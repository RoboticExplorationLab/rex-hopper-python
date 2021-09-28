"""
Copyright (C) 2020 Benjamin Bokser
"""
import simulationbridge
import leg_serial
import leg_parallel
import leg_belt
import wbc
import statemachine
import gait

import time
# import sys

import transforms3d
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, linewidth=np.nan)

def spring(q, l):
    """
    adds linear extension spring b/t joints 1 and 3 of parallel mechanism
    approximated by applying torques to joints 0 and 2
    """
    init_q = [-30 * np.pi / 180, -150 * np.pi / 180]
    if q is None:
        q0 = init_q[0]
        q1 = init_q[1]
    else:
        q0 = q[0]
        q1 = q[1]
    k = 1500  # spring constant, N/m
    L0 = l[0]  # .15
    L2 = l[2]  # .3
    gamma = abs(q1 - q0)
    rmin = 0.204
    r = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(gamma))  # length of spring
    if r < rmin:
        print("error: incorrect spring params")
    T = k * (r - rmin)  # spring tension force
    alpha = np.arccos((-L0 ** 2 + L2 ** 2 + r ** 2) / (2 * L2 * r))
    beta = np.arccos((-L2 ** 2 + L0 ** 2 + r ** 2) / (2 * L0 * r))
    tau_s0 = -T * np.sin(beta) * L0
    tau_s1 = T * np.sin(alpha) * L2
    tau_s = np.array([tau_s0, tau_s1])

    return tau_s

def contact_check(c, c_s, c_prev, steps, con_c):
    if c_prev != c:
        con_c = steps  # timestep at contact change
        c_s = c  # saved contact value
    if con_c - steps <= 300:  # TODO: reset con_c
        c = c_s

    return c, c_s, con_c


class Runner:

    def __init__(self, dt=1e-3, model='serial', ctrl_type='wbc_cycle', plot=False, fixed=False, spring=False,
                 record=False):

        self.dt = dt
        self.u = np.zeros(2)

        # height constant
        self.hconst = 0.3
        self.model = model
        self.ctrl_type = ctrl_type
        self.plot = plot
        self.spring = spring
        if model == 'design':
            L0 = .1
            L1 = .3
            L2 = .3
            L3 = .1
            L4 = .2
            L5 = 0.0205
            self.L = np.array([L0, L1, L2, L3, L4, L5])
            self.leg = leg_parallel.Leg(dt=dt, l=self.L, model=model)
            self.k_kin = 20
            self.k_d = self.k_kin * 0.02
            self.t_p = 0.9  # gait period, seconds 0.5
            self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
            self.dir_s = 1  # spring "direction" (accounts for swapped leg config)
            print("WARNING: Parallel model only works with closed form inv kin, do not attempt wbc (WIP)")
        elif model == 'serial':
            self.leg = leg_serial.Leg(dt=dt)
            self.k_kin = 70 # np.array([70, 70])
            self.k_d = self.k_kin * 0.02
            self.t_p = 1.4  # gait period, seconds 0.5
            self.phi_switch = 0.15  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        elif model == 'parallel':
            L0 = .15
            L1 = .3
            L2 = .3
            L3 = .15
            L4 = .15
            L5 = 0
            self.L = np.array([L0, L1, L2, L3, L4, L5])
            self.leg = leg_parallel.Leg(dt=dt, l=self.L, model=model)
            self.k_kin = 70
            self.k_d = self.k_kin * 0.02
            self.t_p = 1.4  # gait period, seconds 0.5
            self.phi_switch = 0.15  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
            self.dir_s = -1  # spring "direction" (accounts for swapped leg config)
            print("WARNING: Parallel model only works with closed form inv kin, do not attempt wbc (WIP)")
        elif model == 'belt':
            self.leg = leg_belt.Leg(dt=dt)
            self.k_kin = 15 # 210
            self.k_d = self.k_kin * 0.02
            self.t_p = 1.4  # gait period, seconds 0.5
            self.phi_switch = 0.15  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
            print("WARNING: Belt model only works with closed form inv kin, do not attempt wbc (WIP)")

        controller_class = wbc
        self.controller = controller_class.Control(dt=dt)
        self.simulator = simulationbridge.Sim(dt=dt, model=model, fixed=fixed, record=record)
        self.state = statemachine.Char()

        # gait scheduler values
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -self.hconst, self.init_alpha, self.init_beta, self.init_gamma])
        self.target = self.target_init[:]
        self.sh = 1  # estimated contact state
        self.dist_force = np.array([0, 0, 0])

        self.gait = gait.Gait(controller=self.controller, leg=self.leg, t_p=self.t_p, phi_switch=self.phi_switch,
                              hconst=self.hconst, dt=dt)

        # self.target = None
        self.r = np.array([0, 0, -self.hconst])  # initial footstep planning position

        # footstep planner values
        self.omega_d = np.array([0, 0, 0])  # desired angular acceleration for footstep planner
        self.pdot_des = np.array([0, 0, 0])  # desired body velocity in world coords


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

        sh_prev = 0
        t_l = 0
        t_f = 0

        total = 4000  # number of timesteps to plot
        if self.plot:
            value1 = np.zeros((total, 3))
            value2 = np.zeros((total, 3))
            value3 = np.zeros((total, 3))
            value4 = np.zeros((total, 3))
            value5 = np.zeros((total, 3))
            value6 = np.zeros((total, 3))
            if self.model == 'serial' or self.model == 'parallel' or self.model == 'design':
                fig, axs = plt.subplots(2, 3, sharey=False, sharex=True)
                axs[0, 0].set_title('q0 torque')
                plt.xlabel("Timesteps")
                axs[0, 0].set_ylabel("q0 torque (Nm)")
                axs[0, 1].set_title('q1 torque')
                axs[0, 1].set_ylabel("q1 torque (Nm)")
                axs[0, 2].set_title('base z position')
                axs[0, 2].set_ylabel("z position (m)")
                axs[1, 0].set_title('angular velocity q0_dot')
                axs[1, 0].set_ylabel("angular velocity, rpm")
                axs[1, 1].set_title('angular velocity q1_dot')
                axs[1, 1].set_ylabel("angular velocity, rpm")
                axs[1, 2].set_title('Magnitude of Reaction Force on q0')
                axs[1, 2].set_ylabel("Reaction Force Fz, N")
            elif self.model == 'belt':
                fig, axs = plt.subplots(1, 3, sharey=False, sharex=True)
                axs[0].set_title('q0 torque')
                plt.xlabel("Timesteps")
                axs[0].set_ylabel("q0 torque (Nm)")
                axs[1].set_title('angular velocity q0_dot')
                axs[1].set_ylabel("angular velocity, rpm")
                axs[2].set_title('base z position')
                axs[2].set_ylabel("z position (m)")
        else:
            value1 = None
            value2 = None
            value3 = None
            value4 = None
            value5 = None
            value6 = None

        while 1:
            steps += 1
            t = t + self.dt
            # t_diff = time.clock() - t_prev
            # t_prev = time.clock()
            skip = False
            # run simulator to get encoder and IMU feedback
            # put an if statement here once we have hardware bridge too
            if self.spring:
                tau_s = spring(self.leg.q, self.L)*self.dir_s
            else:
                tau_s = np.zeros(2)
            q, b_orient, c, torque, q_dot, f = self.simulator.sim_run(u=self.u, tau_s=tau_s)

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

            if sh == 0 and sh_prev == 1:
                t_f = t  # time of flight
            if sh == 1 and sh_prev == 0:
                t_l = t  # time of landing
                t_ft = t_l - t_f  # last flight time
                print(t_ft)

            sh_prev = sh
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
            mpc_force[2] = 218  # 218  # just a static number for now (removed mpc)

            delp = pdot*self.dt
            # calculate wbc control signal
            if self.ctrl_type == 'wbc_cycle':
                self.u = self.gait.u(state=state, prev_state=prev_state, r_in=pos, r_d=self.r, delp=delp,
                                            b_orient=b_orient, fr_mpc=mpc_force, skip=skip)

            elif self.ctrl_type == 'simple_invkin':
                time.sleep(self.dt/2)  # closed form inv kin runs much faster than full wbc, slow it down
                # self.target[2] = -0.5
                if state == 'Return':
                    # set target position
                    self.target = np.array([0, 0, -0.5])

                elif state == 'HeelStrike':
                    self.target[2] = -self.hconst

                elif state == 'Crouch':
                    # self.target = np.array([-0.1, 0, -self.hconst])
                    self.target[2] = -self.hconst  # go to crouch

                elif state == 'Leap':
                    self.target = np.array([0, 0, -0.55])

                self.u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * self.k_kin \
                         + self.leg.dq * self.k_d

            elif self.ctrl_type == 'static_invkin':
                time.sleep(self.dt / 2)  # closed form inv kin runs much faster than full wbc, slow it down
                self.u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * self.k_kin \
                         + self.leg.dq * self.k_d
                # self.u = -self.controller.wb_control(leg=self.leg, target=self.target, b_orient=b_orient, force=None)

            prev_state = state

            p_base_z = self.simulator.base_pos[0][2]  # base vertical position in world coords

            if self.plot == True and steps <= total-1:
                if self.model == 'serial':
                    value1[steps-1, :] = torque[0] # self.u[0]
                    value2[steps-1, :] = torque[1] # self.u[1]
                    value3[steps-1, :] = p_base_z
                    value4[steps - 1, :] = q_dot[0]*60/(2*np.pi)  # conversion to RPM
                    value5[steps - 1, :] = q_dot[1]*60/(2*np.pi)
                    value6[steps - 1, :] = f[0]
                    if steps == total - 1:
                        axs[0, 0].plot(range(total - 1), value1[:-1, 0], color='blue')
                        axs[0, 1].plot(range(total - 1), value2[:-1, 0], color='blue')
                        axs[0, 2].plot(range(total - 1), value3[:-1, 0], color='blue')
                        axs[1, 0].plot(range(total - 1), value4[:-1, 0], color='blue')
                        axs[1, 1].plot(range(total - 1), value5[:-1, 0], color='blue')
                        axs[1, 2].plot(range(total - 1), value6[:-1, 0], color='blue')
                        plt.show()

                elif self.model == 'parallel' or self.model == 'design':
                    value1[steps - 1, :] = torque[0]  # self.u[0]
                    value2[steps - 1, :] = torque[2]  # self.u[1]
                    value3[steps - 1, :] = p_base_z
                    value4[steps - 1, :] = q_dot[0]*60/(2*np.pi)
                    value5[steps - 1, :] = q_dot[1]*60/(2*np.pi)
                    value6[steps - 1, :] = f[0]
                    if steps == total - 1:
                        axs[0, 0].plot(range(total - 1), value1[:-1, 0], color='blue')
                        axs[0, 1].plot(range(total - 1), value2[:-1, 0], color='blue')
                        axs[0, 2].plot(range(total - 1), value3[:-1, 0], color='blue')
                        axs[1, 0].plot(range(total - 1), value4[:-1, 0], color='blue')
                        axs[1, 1].plot(range(total - 1), value5[:-1, 0], color='blue')
                        axs[1, 2].plot(range(total - 1), value6[:-1, 0], color='blue')
                        plt.show()

                elif self.model == 'belt':
                    value1[steps - 1, :] = torque[0]
                    value2[steps - 1, :] = q_dot[0]*60/(2*np.pi)
                    value3[steps - 1, :] = p_base_z
                    if steps == total - 1:
                        axs[0].plot(range(total - 1), value1[:-1, 0], color='blue')
                        axs[1].plot(range(total - 1), value2[:-1, 0], color='blue')
                        axs[2].plot(range(total - 1), value3[:-1, 0], color='blue')
                        plt.show()
            # print(tau_s)
            print(t)
            # print(s, sh, state)
            # print(p_base_z)
            # print(self.leg.position())
            # print(self.target)
            # print("kin = ", self.leg.inv_kinematics(xyz=self.target) * 180/np.pi)
            # print("enc = ", self.leg.q * 180/np.pi)
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
        # To ensure that after state has been changed, it cannot change to again immediately...
        # Generates "go" variable as True only when criteria for time passed since gait change is met
        if s_prev != s:
            ct = t  # record time of state change
        if ct - t >= self.t_p * (1 - self.phi_switch) * 0.5:
            go = True
        else:
            go = False

        return go, ct
