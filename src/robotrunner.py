"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import simulationbridge
import statemachine
import gait
import plots
import moment_ctrl
import mpc
# import time
# import sys
import copy
import numpy as np
import itertools
np.set_printoptions(suppress=True, linewidth=np.nan)


def ft_check(sh, sh_prev, t, t_f, ft_saved, i_ft):
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
    return t_f, ft_saved, i_ft


def contact_check(c, c_s, c_prev, k, con_c):
    # if contact has just been made, freeze contact detection to True for 300 timesteps
    # protects against vibration/bouncing-related bugs
    if c_prev != c:
        con_c = k  # timestep at contact change
        c_s = c  # saved contact value
    if c_prev != c and con_c - k <= 10:  # TODO: reset con_c
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

    def __init__(self, model, dt=1e-3, ctrl_type='ik_vert', plot=False, fixed=False, spring=False,
                 record=False, scale=1, gravoff=False, direct=False, recalc=False, total_run=10000, gain=5000):

        self.dt = dt
        self.u = np.zeros(model["n_a"])
        self.total_run = total_run
        self.model = model
        self.ctrl_type = ctrl_type
        self.plot = plot
        self.spring = spring
        controller_class = model["controllerclass"]
        leg_class = model["legclass"]
        self.L = np.array(model["linklengths"])
        self.n_a = model["n_a"]
        self.leg = leg_class.Leg(dt=dt, model=model, recalc=recalc)
        # print("total mass = ", self.leg.m_total)
        self.hconst = model["hconst"]  # 0.3  # height constant
        self.fixed = fixed
        self.controller = controller_class.Control(leg=self.leg, dt=dt, gain=gain)
        self.mu = 0.3  # friction
        self.g = 9.81
        X_0 = np.array([0, 0, 0.7*scale, 0, 0, 0, self.g])  # initial conditions
        self.simulator = simulationbridge.Sim(X_0=X_0, model=model, dt=dt, fixed=fixed, spring=spring, record=record,
                                              scale=scale, gravoff=gravoff, direct=direct)
        self.moment = moment_ctrl.MomentCtrl(model=model, dt=dt)
        self.state = statemachine.Char()
        self.target_init = np.array([0, 0, -self.hconst, 0, 0, 0])
        self.target = self.target_init[:]
        self.t_p = 1  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        t_st = self.t_p * self.phi_switch  # time spent in stance
        self.gait = gait.Gait(model=model, moment=self.moment, controller=self.controller, leg=self.leg,
                              target=self.target, hconst=self.hconst, t_st=t_st, use_qp=False, gain=gain, dt=dt)
        self.N = 10  # mpc horizon
        self.horz_len = self.t_p * self.phi_switch * self.N / self.dt  # horizon length (timesteps)
        if self.ctrl_type == 'mpc':
            self.mpc_t = self.t_p * self.phi_switch  # mpc sampling time (s)
            self.gaitfn = self.gait.u_mpc
            self.mpc = mpc.Mpc(t=self.mpc_t, N=self.N, m=self.leg.m_total, g=self.g, mu=self.mu)
            self.mpc_factor = self.mpc_t / self.dt  # mpc sampling time (timesteps)
        elif self.ctrl_type == 'wbc_raibert':
            self.gaitfn = self.gait.u_raibert
        elif self.ctrl_type == 'wbc_vert':
            self.gaitfn = self.gait.u_wbc_vert
        elif self.ctrl_type == 'wbc_static':
            self.gaitfn = self.gait.u_wbc_static
        elif self.ctrl_type == 'ik_vert':
            self.gaitfn = self.gait.u_ik_vert
        elif self.ctrl_type == 'ik_static':
            self.gaitfn = self.gait.u_ik_static

        self.X_f = np.hstack([2, 2, 0.5, 0, 0, 0, self.g]).T  # desired final state in world frame

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        state_prev = str("init")
        ct, s, c_prev, con_c, c_s, sh_prev = 0, 0, 0, 0, 0, 0
        t_f = 0
        ft_saved = np.zeros(total)
        i_ft = 0  # flight timer counter

        mpc_factor = None
        mpc_counter = None
        if self.ctrl_type == 'mpc':
            mpc_factor = self.mpc_factor  # repeat mpc every x seconds
            mpc_counter = copy.copy(mpc_factor)

        force_f = np.zeros((3, 1))
        n_a = self.n_a
        tauhist = np.zeros((total, n_a))
        dqhist = np.zeros((total, n_a))
        ahist = np.zeros((total, n_a))
        vhist = np.zeros((total, n_a))
        phist = np.zeros((total, 3))
        thetahist = np.zeros((total, 3))
        setphist = np.zeros((total, 3))
        fhist = np.zeros((total, 3))
        x_des_hist = np.zeros((total, 3))
        fthist = np.zeros(total)

        t = 0  # time
        for k in range(0, total):
            t = t + self.dt

            # run simulator to get encoder and IMU feedback
            qa, dqa, Q_base, c, tau, f, i, v = self.simulator.sim_run(u=self.u)  # TODO: More realistic contact detect
            self.leg.update_state(q_in=qa[0:2])  # enter encoder values into leg kinematics/dynamics
            self.moment.update_state(q_in=qa[2:], dq_in=dqa[2:])

            s_prev = s
            c, c_s, con_c = contact_check(c, c_s, c_prev, k, con_c)  # Like using limit switches
            sh = copy.copy(c)

            t_f, ft_saved, i_ft = ft_check(sh, sh_prev, t, t_f, ft_saved, i_ft)  # flight time checker

            # TODO: Actual state estimator... getting it straight from sim is cheating
            pdot = np.array(self.simulator.v)  # base linear velocity in global Cartesian coordinates
            p = np.array(self.simulator.p)  # p = p + pdot * self.dt  # body position in world coordinates
            # delp = pdot * self.dt

            go, ct = gait_check(s, s_prev=s_prev, ct=ct, t=t)  # prevents stuck in stance bug
            state = self.state.FSM.execute(s=s, sh=sh, go=go, pdot=pdot, leg_pos=self.leg.position())
            # pf = utils.Z(Q_base, self.leg.position()[:, -1])  # position of the foot in world frame
            # calculate control signal
            X_in = np.hstack([p, pdot, self.g]).T  # array of the states for MPC
            X_ref = self.path_plan(X_in=X_in)
            if self.ctrl_type == 'mpc':
                if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    X_refN = X_ref[::int(self.mpc_factor)]
                    force_f, sm = self.mpc.mpcontrol(X_in=X_in, X_ref=X_refN, s=s)
                mpc_counter += 1

            self.u, thetar, setp = self.gaitfn(state=state, state_prev=state_prev, X_in=X_in, X_ref=X_ref[100, :],
                                               Q_base=Q_base, fr=np.reshape(force_f[:, 0], (3, 1)))

            x_des_hist[k, :] = self.gait.x_des
            fhist[k, :] = f[1, :]
            fthist[k] = ft_saved[i_ft]
            setphist[k, :] = setp
            thetahist[k, :] = thetar
            tauhist[k, :] = tau
            dqhist[k, :] = dqa
            ahist[k, :] = i
            vhist[k, :] = v
            phist[k, :] = self.simulator.base_pos[0]  # base position in world coords

            state_prev = state
            sh_prev = sh
            c_prev = c

        if self.plot == True:
            plots.thetaplot(total, thetahist, setphist)
            plots.tauplot(total, n_a, tauhist)
            plots.dqplot(total, n_a, dqhist)
            # plots.fplot(total, phist, fhist, fthist)
            plots.posplot(p_ref=self.X_f[0:3], phist=phist, xfhist=x_des_hist)
            # plots.currentplot(total, n_a, ahist)
            # plots.voltageplot(total, n_a, vhist)
            # plots.electrtotalplot(total, ahist, vhist, dt=self.dt)

        return ft_saved

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        if phi > self.phi_switch:
            s = 0  # scheduled swing
        else:
            s = 1  # scheduled stance
        return s

    def path_plan(self, X_in):
        # Path planner--generate reference trajectory
        dt = self.dt
        size_mpc = int(self.horz_len)  # length of MPC horizon in s TODO: Perhaps N should vary wrt time?
        # timesteps given to get to target, either mpc length or based on distance (whichever is smaller)
        t_ref = int(np.minimum(size_mpc, np.linalg.norm(self.X_f[0:2] - X_in[0:2]) * 1000))
        X_ref = np.linspace(start=X_in, stop=self.X_f, num=t_ref)  # interpolate positions
        # interpolate velocities
        X_ref[:-1, 3] = [(X_ref[i + 1, 0] - X_ref[i, 0]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]
        X_ref[:-1, 4] = [(X_ref[i + 1, 1] - X_ref[i, 1]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]
        X_ref[:-1, 5] = [(X_ref[i + 1, 2] - X_ref[i, 2]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]

        if (size_mpc - t_ref) == 0:
            pass
        elif t_ref == 0:
            X_ref = np.array(list(itertools.repeat(self.X_f, int(size_mpc))))
        else:
            X_ref = np.vstack((X_ref, list(itertools.repeat(self.X_f, int(size_mpc - t_ref)))))

        return X_ref