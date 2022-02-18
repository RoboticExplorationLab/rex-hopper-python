"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import simulationbridge
import statemachine
import gait
import plots
import moment_ctrl

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
        self.u = np.zeros(model["n_a"])
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
        self.moment = moment_ctrl.MomentCtrl(model=model, dt=dt)

        self.state = statemachine.Char()

        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.target_init = np.array([0, 0, -self.hconst, self.init_alpha, self.init_beta, self.init_gamma])
        self.target = self.target_init[:]
        self.sh = 1  # estimated contact state

        use_qp = False  # TODO: Change
        self.gait = gait.Gait(model=model, moment=self.moment, controller=self.controller, leg=self.leg,
                              target=self.target, hconst=self.hconst, use_qp=use_qp, dt=dt)

        # self.r = np.array([0, 0, -self.hconst])  # initial footstep planning position

        self.omega_d = np.array([0, 0, 0])  # desired angular acceleration for footstep planner
        self.p_ref = np.array([2, 0, 0])  # desired body pos in world coords
        self.n_a = model["n_a"]

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        p_ref = self.p_ref
        steps = -1
        t = 0  # time
        # p = np.array([0, 0, 0])  # initialize body position

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
        n_a = self.n_a
        tauhist = np.zeros((total, n_a))
        dqhist = np.zeros((total, n_a))
        ahist = np.zeros((total, n_a))
        vhist = np.zeros((total, n_a))

        phist = np.zeros((total, 3))
        thetahist = np.zeros((total, 3))
        setphist = np.zeros((total, 3))
        fhist = np.zeros((total, 3))
        fthist = np.zeros(total)
        x_des_hist = np.zeros((total, 3))
        thetar = np.zeros(3)
        setp = np.zeros(3)

        while steps < self.total_run:
            steps += 1
            t = t + self.dt

            # run simulator to get encoder and IMU feedback
            # TODO: More realistic contact detection

            q, dq, Q_base, c, tau, f, i, v = self.simulator.sim_run(u=self.u)
            # enter encoder values into leg kinematics/dynamics
            self.leg.update_state(q_in=q[0:4])  # TODO: should not take unactuated q from simulator
            self.moment.update_state(q_in=q[4:], qdot_in=dq[2:])

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
                self.u, thetar, setp = self.gait.u_raibert(state=state, state_prev=state_prev, Q_base=Q_base,
                                                           p=p, p_ref=p_ref, pdot=pdot, fr=force_f)

            elif self.ctrl_type == 'wbc_vert':
                self.u, thetar, setp = self.gait.u_wbc_vert(state=state, Q_base=Q_base, fr=force_f)

            elif self.ctrl_type == 'wbc_static':
                self.u, thetar, setp = self.gait.u_wbc_static(Q_base=Q_base, fr=force_f)

            elif self.ctrl_type == 'invkin_vert':
                time.sleep(self.dt)
                self.u, thetar, setp = self.gait.u_invkin_vert(state=state, Q_base=Q_base,
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

                self.u, thetar, setp = self.gait.u_invkin_static(Q_base=Q_base, k=k, kd=kd)
                # self.u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3] * 5 / 3)) * k + self.leg.dq * kd

            x_des_hist[steps, :] = self.gait.x_des
            fhist[steps, :] = f[1, :]
            fthist[steps] = ft_saved[i_ft]
            setphist[steps, :] = setp
            thetahist[steps, :] = thetar
            tauhist[steps, :] = tau
            dqhist[steps, :] = dq
            ahist[steps, :] = i
            vhist[steps, :] = v
            phist[steps, :] = self.simulator.base_pos[0]  # base position in world coords

            state_prev = state
            sh_prev = sh
            c_prev = c

        if self.plot == True:
            plots.thetaplot(total, thetahist, setphist)
            plots.tauplot(total, n_a, tauhist)
            plots.dqplot(total, n_a, dqhist)
            # plots.fplot(total, phist, fhist, fthist)
            plots.posplot(p_ref=p_ref, phist=phist, xfhist=x_des_hist)
            # plots.currentplot(total, n_a, ahist)
            # plots.voltageplot(total, n_a, vhist)
            # plots.electrtotalplot(total, ahist, vhist, dt=self.dt)

        return ft_saved
