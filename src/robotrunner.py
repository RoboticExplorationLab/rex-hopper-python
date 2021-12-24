"""
Copyright (C) 2020 Benjamin Bokser
"""
import simulationbridge
import statemachine
import gait
import rw
import plots

import time
# import sys

import transforms3d
import numpy as np

np.set_printoptions(suppress=True, linewidth=np.nan)


def contact_check(c, c_s, c_prev, steps, con_c):
    if c_prev != c:
        con_c = steps  # timestep at contact change
        c_s = c  # saved contact value
    if c_prev != c and con_c - steps <= 10:  # TODO: reset con_c
        c = c_s
    return c, c_s, con_c


def gait_check(s, s_prev, ct, t):
    # To ensure that after state has been changed, it cannot change to again immediately...
    # Generates "go" variable as True only when criteria for time passed since gait change is met
    if s_prev != s:
        ct = t  # record time of state change
    if ct - t >= 0.25:
        go = True
    else:
        go = False

    return go, ct


class Runner:

    def __init__(self, model, dt=1e-3, ctrl_type='simple_invkin', plot=False, fixed=False, spring=False,
                 record=False, scale=1, gravoff=False, direct=False, total_run=10000, gain=4):

        self.dt = dt
        self.u = np.zeros(2)
        self.u_rw = np.zeros(3)
        self.total_run = total_run
        # height constant

        self.model = model
        self.ctrl_type = ctrl_type
        self.plot = plot
        self.spring = spring

        controller_class = model["controllerclass"]
        leg_class = model["legclass"]
        self.L = np.array(model["linklengths"])
        self.leg = leg_class.Leg(dt=dt, model=model)
        self.k_g = model["k_g"]
        self.k_gd = model["k_gd"]
        self.k_a = model["k_a"]
        self.k_ad = model["k_ad"]
        self.dir_s = model["springpolarity"]
        self.hconst = model["hconst"]  # 0.3
        self.fixed = fixed
        self.controller = controller_class.Control(dt=dt, gain=gain)
        self.simulator = simulationbridge.Sim(dt=dt, model=model, fixed=fixed, spring=spring, record=record,
                                              scale=scale, gravoff=gravoff, direct=direct)
        self.state = statemachine.Char()

        # gait scheduler values
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -self.hconst, self.init_alpha, self.init_beta, self.init_gamma])
        self.target = self.target_init[:]
        self.sh = 1  # estimated contact state
        self.dist_force = np.array([0, 0, 0])

        self.gait = gait.Gait(controller=self.controller, leg=self.leg, target=self.target, hconst=self.hconst,
                              use_qp=False, dt=dt)

        # self.target = None
        self.r = np.array([0, 0, -self.hconst])  # initial footstep planning position

        # footstep planner values
        self.omega_d = np.array([0, 0, 0])  # desired angular acceleration for footstep planner
        self.pdot_des = np.array([0, 0, 0])  # desired body velocity in world coords

    def run(self):
        steps = 0
        t = 0  # time
        p = np.array([0, 0, 0])  # initialize body position
        skip = False
        prev_state = str("init")

        ct = 0
        s = 0
        c_prev = 0
        con_c = 0
        c_s = 0
        sh_prev = 0

        t_f = 0
        ft_saved = np.zeros(self.total_run)
        i_ft = 0  # flight timer counter

        total = self.total_run  # number of timesteps to plot

        tau0hist = np.zeros((total, 1))
        tau2hist = np.zeros(total)
        phist = np.zeros(total)
        thetahist = np.zeros((total, 3))
        rw1hist = np.zeros(total)
        rw2hist = np.zeros(total)
        rwzhist = np.zeros(total)
        err_sum = np.zeros(3)
        err_prev = np.zeros(3)
        thetar = np.zeros(3)
        while steps < self.total_run:
            steps += 1
            t = t + self.dt

            # run simulator to get encoder and IMU feedback
            q, q_dot, qrw, qrw_dot, Q_base, c, torque, f = self.simulator.sim_run(u=self.u, u_rw=self.u_rw)
            b_orient = transforms3d.quaternions.quat2mat(Q_base)
            # enter encoder values into leg kinematics/dynamics
            self.leg.update_state(q_in=q)

            s_prev = s
            # prevents stuck in stance bug
            go, ct = gait_check(s, s_prev=s_prev, ct=ct, t=t)

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
                if t_ft > 0.1:  # ignore flight times of less than 0.1 second (these are foot bounces)
                    print(t_ft)
                    ft_saved[i_ft] = t_ft  # save flight time to vector
                    i_ft += 1

            sh_prev = sh
            c_prev = c

            # forward kinematics
            pos = np.dot(b_orient, self.leg.position())  # [:, -1])  TODO: Check
            pdot = np.array(self.simulator.v)  # base linear velocity in global Cartesian coordinates
            p = p + pdot * self.dt  # body position in world coordinates

            theta = np.array(transforms3d.euler.mat2euler(b_orient, axes='sxyz'))
            # theta[0] -= np.pi
            # theta = transforms3d.quaternions.quat2axangle(Q_base)  # ax-angle representation

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

            omega = np.array(self.simulator.omega_xyz)

            # x_in = np.hstack([theta, p, omega, pdot]).T  # array of the states for MPC
            x_ref = np.hstack([np.zeros(3), np.zeros(3), self.omega_d, self.pdot_des]).T  # reference pose (desired)

            mpc_force = np.zeros(3)
            mpc_force[2] = 218  # TODO: Add MPC back

            delp = pdot*self.dt
            # calculate wbc control signal
            if self.ctrl_type == 'wbc_cycle':
                self.u = self.gait.u(state=state, prev_state=prev_state, r_in=pos, r_d=self.r, delp=delp,
                                            b_orient=b_orient, fr_mpc=mpc_force, skip=skip)

            elif self.ctrl_type == 'simple_invkin':
                self.u = self.gait.u_invkin(state=state, k_g=self.k_g, k_gd=self.k_gd, k_a = self.k_a, k_ad=self.k_ad)

            elif self.ctrl_type == 'static_invkin':
                time.sleep(self.dt)  # closed form inv kin runs much faster than full wbc, slow it down
                if self.fixed == True:
                    k = self.k_a
                    kd = self.k_ad
                else:
                    k = self.k_g
                    kd = self.k_gd

                if self.model["model"] == 'design' or self.model["model"] == 'design_rw':
                    q02 = np.zeros(2)
                    q02[0] = self.leg.q[0]
                    q02[1] = self.leg.q[2]
                    dq02 = np.zeros(2)
                    dq02[0] = self.leg.dq[0]
                    dq02[1] = self.leg.dq[2]
                    self.u = (q02 - self.leg.inv_kinematics(xyz=self.target[0:3]*5/3)) * k + dq02 * kd
                else:
                    self.u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3]*5/3)) * k + self.leg.dq * kd
                # self.u = -self.controller.wb_control(leg=self.leg, target=self.target, b_orient=b_orient, force=None)

            if self.model["model"] == 'design_rw':
                self.u_rw, err_sum, err_prev, thetar = rw.rw_control(self.dt, Q_base, err_sum, err_prev)

            prev_state = state

            p_base_z = self.simulator.base_pos[0][2]  # base vertical position in world coords

            tau0hist[steps - 1] = torque[0]  # self.u[0]
            tau2hist[steps - 1] = torque[2]  # self.u[1]
            phist[steps - 1] = p_base_z
            thetahist[steps - 1, :] = thetar
            rw1hist[steps - 1] = torque[4]
            rw2hist[steps - 1] = torque[5]
            rwzhist[steps - 1] = torque[6]

            # print("pos = ", self.leg.position())
            # print("kin = ", self.leg.inv_kinematics(xyz=self.target) * 180/np.pi)
            # print("enc = ", self.leg.q * 180/np.pi)
            # sys.stdout.write("\033[F")  # back to previous line
            # sys.stdout.write("\033[K")  # clear line

        plots.thetaplot(total, thetahist[:, 0], thetahist[:, 1], thetahist[:, 2], rw1hist, rw2hist, rwzhist)

        return ft_saved