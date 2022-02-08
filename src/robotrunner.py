"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import simulationbridge
import statemachine
import gait
import plots

import time
# import sys

import transforms3d
import numpy as np

np.set_printoptions(suppress=True, linewidth=np.nan)


def contact_check(c, c_s, c_prev, steps, con_c):
    # if contact has just been made, freeze contact detection to True for 300 timesteps
    # protects against vibration/bouncing-related bugs
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
                 record=False, scale=1, gravoff=False, direct=False, recalc=False, total_run=10000, gain=5000):

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
        self.leg = leg_class.Leg(dt=dt, model=model, recalc=recalc)
        self.k_g = model["k_g"]
        self.k_gd = model["k_gd"]
        self.k_a = model["k_a"]
        self.k_ad = model["k_ad"]
        self.hconst = model["hconst"]  # 0.3
        self.fixed = fixed
        self.controller = controller_class.Control(leg=self.leg, dt=dt, gain=gain)
        self.simulator = simulationbridge.Sim(dt=dt, model=model, fixed=fixed, spring=spring, record=record,
                                              scale=scale, gravoff=gravoff, direct=direct)
        self.state = statemachine.Char()

        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -self.hconst, self.init_alpha, self.init_beta, self.init_gamma])
        self.target = self.target_init[:]
        self.sh = 1  # estimated contact state

        use_qp = False  # TODO: Change
        self.gait = gait.Gait(model=model, controller=self.controller, leg=self.leg, target=self.target, hconst=self.hconst,
                              use_qp=use_qp, dt=dt)

        # self.r = np.array([0, 0, -self.hconst])  # initial footstep planning position

        self.omega_d = np.array([0, 0, 0])  # desired angular acceleration for footstep planner
        self.p_ref = np.array([2, 0, 0])  # desired body pos in world coords

    def run(self):
        total = self.total_run  # number of timesteps to plot
        p_ref = self.p_ref
        steps = 0
        t = 0  # time
        p = np.array([0, 0, 0])  # initialize body position

        state_prev = str("init")

        ct = 0
        s = 0
        c_prev = 0
        con_c = 0
        c_s = 0
        sh_prev = 0

        t_f = 0
        ft_saved = np.zeros(total)
        i_ft = 0  # flight timer counter

        force_f = None
        # force_f = np.zeros((3, 1))
        # force_f[2] = -120

        tau0hist = np.zeros((total, 1))
        tau2hist = np.zeros(total)
        phist = np.zeros((total, 3))
        thetahist = np.zeros((total, 3))
        setphist = np.zeros((total, 3))
        rw1hist = np.zeros(total)
        rw2hist = np.zeros(total)
        rwzhist = np.zeros(total)
        w1hist = np.zeros(total)
        w2hist = np.zeros(total)
        w3hist = np.zeros(total)
        thetar = np.zeros(3)
        setp = np.zeros(3)
        x_des_hist = np.zeros((total, 3))
        fhist = np.zeros((total, 3))
        fthist = np.zeros(total)
        q0ahist = np.zeros(total)
        q2ahist = np.zeros(total)
        rw1ahist = np.zeros(total)
        rw2ahist = np.zeros(total)
        rwzahist = np.zeros(total)
        q0vhist = np.zeros((total, 2))
        q2vhist = np.zeros((total, 2))
        rw1vhist = np.zeros((total, 2))
        rw2vhist = np.zeros((total, 2))
        rwzvhist = np.zeros((total, 2))

        while steps < self.total_run:
            steps += 1
            t = t + self.dt

            # run simulator to get encoder and IMU feedback
            # TODO: More realistic contact detection
            q, q_dot, qrw, qrw_dot, Q_base, c, torque, f = self.simulator.sim_run(u=self.u, u_rw=self.u_rw)

            # enter encoder values into leg kinematics/dynamics
            self.leg.update_state(q_in=q)

            s_prev = s
            # prevents stuck in stance bug
            go, ct = gait_check(s, s_prev=s_prev, ct=ct, t=t)

            # Like using limit switches
            c, c_s, con_c = contact_check(c, c_s, c_prev, steps, con_c)
            sh = int(c)

            # flight time checker
            if sh == 0 and sh_prev == 1:
                t_f = t  # time of flight
            if sh == 1 and sh_prev == 0:
                t_l = t  # time of landing
                t_ft = t_l - t_f  # last flight time
                if t_ft > 0.1:  # ignore flight times of less than 0.1 second (these are foot bounces)
                    print("flight time = ", t_ft)
                    ft_saved[i_ft] = t_ft  # save flight time to vector
                    i_ft += 1

            # TODO: Actual state estimator... getting it straight from sim is cheating
            pdot = np.array(self.simulator.v)  # base linear velocity in global Cartesian coordinates
            # p = p + pdot * self.dt  # body position in world coordinates
            p = np.array(self.simulator.p)
            # delp = pdot * self.dt

            state = self.state.FSM.execute(s=s, sh=sh, go=go, pdot=pdot, leg_pos=self.leg.position())

            # calculate wbc control signal
            if self.ctrl_type == 'wbc_raibert':
                self.u, self.u_rw, thetar, setp = self.gait.u_raibert(state=state, state_prev=state_prev, Q_base=Q_base,
                                                                      p=p, p_ref=p_ref, pdot=pdot, qrw_dot=qrw_dot,
                                                                      fr=force_f)

            elif self.ctrl_type == 'wbc_vert':
                self.u, self.u_rw, thetar, setp = self.gait.u_wbc_vert(state=state, Q_base=Q_base, qrw_dot=qrw_dot,
                                                                       fr=force_f)

            elif self.ctrl_type == 'wbc_static':
                self.u, self.u_rw, thetar, setp = self.gait.u_wbc_static(Q_base=Q_base, qrw_dot=qrw_dot, fr=force_f)

            elif self.ctrl_type == 'invkin_vert':
                time.sleep(self.dt)
                self.u, self.u_rw, thetar, setp = self.gait.u_invkin_vert(state=state, Q_base=Q_base, qrw_dot=qrw_dot,
                                                                          k_g=self.k_g, k_gd=self.k_gd,
                                                                          k_a=self.k_a, k_ad=self.k_ad)

            elif self.ctrl_type == 'invkin_static':
                time.sleep(self.dt)  # closed form inv kin runs much faster than full wbc, slow it down
                if self.fixed == True:
                    k = self.k_a
                    kd = self.k_ad
                else:
                    k = self.k_g
                    kd = self.k_gd

                self.u, self.u_rw, thetar, setp = self.gait.u_invkin_static(Q_base=Q_base, qrw_dot=qrw_dot, k=k, kd=kd)
                # self.u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3] * 5 / 3)) * k + self.leg.dq * kd

            if self.model["model"] == 'design_rw':
                rw1hist[steps - 1] = torque[4]
                rw2hist[steps - 1] = torque[5]
                rwzhist[steps - 1] = torque[6]
                w1hist[steps - 1] = qrw_dot[0]
                w2hist[steps - 1] = qrw_dot[1]
                w3hist[steps - 1] = qrw_dot[2]
                setphist[steps - 1, :] = setp
                thetahist[steps - 1, :] = thetar
                tau0hist[steps - 1] = torque[0]  # self.u[0]
                tau2hist[steps - 1] = torque[2]  # self.u[1]
                x_des_hist[steps - 1, :] = self.gait.x_des
                fhist[steps - 1, :] = f[1, :]
                fthist[steps - 1] = ft_saved[i_ft]
                q0ahist[steps - 1] = self.simulator.actuator_q0.i_actual
                q2ahist[steps - 1] = self.simulator.actuator_q2.i_actual
                rw1ahist[steps - 1] = self.simulator.actuator_rw1.i_actual
                rw2ahist[steps - 1] = self.simulator.actuator_rw2.i_actual
                rwzahist[steps - 1] = self.simulator.actuator_rwz.i_actual
                q0vhist[steps - 1, :] = self.simulator.actuator_q0.v_actual
                q2vhist[steps - 1, :] = self.simulator.actuator_q2.v_actual
                rw1vhist[steps - 1, :] = self.simulator.actuator_rw1.v_actual
                rw2vhist[steps - 1, :] = self.simulator.actuator_rw2.v_actual
                rwzvhist[steps - 1, :] = self.simulator.actuator_rwz.v_actual

            p_base = self.simulator.base_pos[0]  # base position in world coords

            phist[steps - 1, :] = p_base

            state_prev = state
            sh_prev = sh
            c_prev = c

        if self.plot == True:
            plots.rwplot(total, thetahist[:, 0], thetahist[:, 1], thetahist[:, 2],
                         rw1hist, rw2hist, rwzhist,
                         w1hist, w2hist, w3hist,
                         setphist[:, 0], setphist[:, 1], setphist[:, 2])
            plots.posplot(p_ref=p_ref, phist=phist, xfhist=x_des_hist)
            plots.tauplot(total, tau0hist, tau2hist, pzhist=phist[:, 2], fxhist=fhist[:, 0],
                          fzhist=fhist[:, 2], fthist=fthist)
            plots.electrplot(total, q0ahist, q2ahist, rw1ahist, rw2ahist, rwzahist,
                             q0vhist, q2vhist, rw1vhist, rw2vhist, rwzvhist)
            plots.electrtotalplot(total, q0ahist, q2ahist, rw1ahist, rw2ahist, rwzahist,
                                  q0vhist, q2vhist, rw1vhist, rw2vhist, rwzvhist, dt=self.dt)
        return ft_saved
