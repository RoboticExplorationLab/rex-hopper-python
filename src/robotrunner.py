"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import simulationbridge
import statemachine
import statemachine_s
import gait
import plots
import moment_ctrl
import mpc
import utils
import spring
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

    def __init__(self, model, dt=1e-3, ctrl_type='ik_vert', plot=False, fixed=False, spr=False,
                 record=False, scale=1, gravoff=False, direct=False, recalc=False, total_run=10000, gain=5000):

        self.dt = dt
        self.u = np.zeros(model["n_a"])
        self.total_run = total_run
        self.model = model
        self.ctrl_type = ctrl_type
        self.plot = plot
        self.spr = spr
        self.fixed = fixed
        self.L = np.array(model["linklengths"])
        self.n_a = model["n_a"]
        self.hconst = model["hconst"]  # 0.3  # height constant
        self.g = 9.807
        self.mu = 0.3  # friction

        # [theta, p, omega, pdot]
        self.X_0 = np.array([0, 0, 0, 0, 0, 0.7 * scale, 0, 0, 0, 0, 0, 0]).T  # initial conditions
        self.X_f = np.array([0, 0, 0, 2, 2, 0.5 * scale, 0, 0, 0, 0, 0, 0]).T  # desired final state in world frame

        self.spring = spring.Spring(model)
        leg_class = model["legclass"]
        self.leg = leg_class.Leg(dt=dt, model=model, g=self.g, recalc=recalc)
        controller_class = model["controllerclass"]
        self.controller = controller_class.Control(leg=self.leg, spring=self.spring, m=self.leg.m_total,
                                                   spr=spr, dt=dt, gain=gain)
        self.simulator = simulationbridge.Sim(X_0=self.X_0, model=model, spring=self.spring, dt=dt, g=self.g,
                                              fixed=fixed, spr=spr, record=record, scale=scale,
                                              gravoff=gravoff, direct=direct)
        self.moment = moment_ctrl.MomentCtrl(model=model, dt=dt)
        self.target_init = np.array([0, 0, -self.hconst, 0, 0, 0])
        self.target = self.target_init[:]
        self.t_p = 0.8  # gait period, seconds  # TODO: Modify
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        t_st = self.t_p * self.phi_switch  # time spent in stance
        self.gait = gait.Gait(model=model, moment=self.moment, controller=self.controller, leg=self.leg,
                              target=self.target, hconst=self.hconst, t_st=t_st, X_f=self.X_f,
                              use_qp=True, gain=gain, dt=dt)
        self.N = 20  # mpc prediction horizon length (mpc steps)  # TODO: Modify
        self.mpc_dt = 0.05  # mpc sampling time (s)
        self.mpc_factor = int(self.mpc_dt / self.dt)  # mpc sampling time (timesteps), repeat mpc every x timesteps
        self.N_k = self.N * self.mpc_factor  # mpc prediction horizon length (timesteps)
        self.mpc = mpc.Mpc(model=model, t=self.mpc_dt, N=self.N, m=self.leg.m_total, g=self.g, mu=self.mu)

        if self.ctrl_type == 'mpc':
            self.state = statemachine_s.Char()
            self.gaitfn = self.gait.u_mpc
        elif self.ctrl_type == 'wbc_raibert':
            self.gaitfn = self.gait.u_raibert
            self.state = statemachine.Char()
        elif self.ctrl_type == 'wbc_vert':
            self.gaitfn = self.gait.u_wbc_vert
            self.state = statemachine.Char()
        elif self.ctrl_type == 'wbc_static':
            self.gaitfn = self.gait.u_wbc_static
            self.state = statemachine.Char()
        elif self.ctrl_type == 'ik_vert':
            self.gaitfn = self.gait.u_ik_vert
            self.state = statemachine.Char()
        elif self.ctrl_type == 'ik_static':
            self.gaitfn = self.gait.u_ik_static
            self.state = statemachine.Char()

    def run(self):
        total = self.total_run + 1  # number of timesteps to plot
        state_prev = str("init")
        ct, s, c_prev, con_c, c_s, sh_prev = 0, 0, 0, 0, 0, 0
        t_f = 0
        ft_saved = np.zeros(total)
        i_ft = 0  # flight timer counter
        X_pred = None
        U_pred = None
        mpc_factor = self.mpc_factor  # repeat mpc every x seconds
        mpc_counter = copy.copy(mpc_factor)

        force_f = np.zeros((3, 1))
        n_a = self.n_a
        tauhist = np.zeros((total, n_a))
        dqhist = np.zeros((total, n_a))
        ahist = np.zeros((total, n_a))
        vhist = np.zeros((total, n_a))
        phist = np.zeros((total, 3))
        pfhist = np.zeros((total, 3))
        thetahist = np.zeros((total, 3))
        setphist = np.zeros((total, 3))
        grfhist = np.zeros((total, 3))
        fhist = np.zeros((total, 3))
        pfdes = np.zeros((total, 3))
        fthist = np.zeros(total)
        s_hist = np.zeros((total, 2))

        t = 0  # time
        t0 = None
        first_contact = 0
        s_prev = 0
        for k in range(0, total):
            t = t + self.dt

            # run simulator to get encoder and IMU feedback
            qa, dqa, Q_base, c, tau, f_sens, tau_sens, i, v, grf = self.simulator.sim_run(u=self.u)
            self.leg.update_state(q_in=qa[0:2], Q_base=Q_base)  # enter encoder & IMU values into leg k/dynamics
            self.moment.update_state(q_in=qa[2:], dq_in=dqa[2:])

            c, c_s, con_c = contact_check(c, c_s, c_prev, k, con_c)  # Like using limit switches
            sh = copy.copy(c)
            if sh == 1 and first_contact == 0:
                t0 = t  # starting time begins when robot first makes contact
                first_contact = 1  # ensure this doesn't trigger again
            elif sh == 1 and first_contact == 1:
                s = self.gait_scheduler(t, t0)
            else:
                s = 0

            t_f, ft_saved, i_ft = ft_check(sh, sh_prev, t, t_f, ft_saved, i_ft)  # flight time checker

            # TODO: Actual state estimator... getting it straight from sim is cheating
            theta = np.array(utils.quat2euler(Q_base))
            p = np.array(self.simulator.p)  # p = p + pdot * self.dt  # body position in world coordinates
            omega = np.array(self.simulator.omega)
            pdot = np.array(self.simulator.v)  # base linear velocity in global Cartesian coordinates

            go, ct = gait_check(s, s_prev=s_prev, ct=ct, t=t)  # prevents stuck in stance bug
            state = self.state.FSM.execute(s=s, sh=sh, go=go, pdot=pdot, leg_pos=self.leg.position())
            # pf = utils.Z(Q_base, self.leg.position()[:, -1])  # position of the foot in world frame

            X_in = np.hstack([theta, p, omega, pdot]).T  # array of the states for MPC
            X_ref = self.path_plan(X_in=X_in)

            if self.ctrl_type == 'mpc' and first_contact == 1:
                if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    C = self.gait_map(t, t0)
                    X_refN = X_ref[::int(mpc_factor)]
                    U_pred, X_pred = self.mpc.mpcontrol(X_in=X_in, X_ref=X_refN, Q_base=Q_base, C=C)

                mpc_counter += 1

            self.u, thetar, setp = self.gaitfn(state=state, state_prev=state_prev, X_in=X_in, X_ref=X_ref[100, :],
                                               X_pred=X_pred, U_pred=U_pred, Q_base=Q_base, grf=grf, s=s)

            grfhist[k, :] = grf.flatten()  # ground reaction force
            fhist[k, :] = force_f[0, :]
            fthist[k] = ft_saved[i_ft]
            setphist[k, :] = setp
            thetahist[k, :] = thetar
            tauhist[k, :] = tau
            dqhist[k, :] = dqa
            ahist[k, :] = i
            vhist[k, :] = v
            phist[k, :] = self.simulator.base_pos[0]  # base position in world coords
            pfdes[k, :] = self.gait.x_des  # desired footstep positions
            pfhist[k, :] = self.simulator.base_pos[0] + utils.Z(Q_base, self.leg.position()).flatten()  # foot pos
            s_hist[k, :] = [s, sh]
            state_prev = state
            s_prev = s
            sh_prev = sh
            c_prev = c
            # time.sleep(0.1)

        if self.plot == True:
            plots.thetaplot(total, thetahist, setphist)
            # plots.tauplot(total, n_a, tauhist)
            # plots.dqplot(total, n_a, dqhist)
            plots.fplot(total, phist=phist, fhist=fhist, shist=s_hist)
            plots.grfplot(total, phist, grfhist, fthist)
            plots.posplot_3d(p_ref=self.X_f[0:3], phist=phist, pfdes=pfdes)
            # plots.posplot(p_ref=self.X_f[0:3], phist=phist, pfdes=pfdes)
            # plots.currentplot(total, n_a, ahist)
            # plots.voltageplot(total, n_a, vhist)
            plots.etotalplot(total, ahist, vhist, dt=self.dt)

        return ft_saved

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        return phi < self.phi_switch  # TODO: make sure this works

    def gait_map(self, ts, t0):
        # generate vector of scheduled contact states over the mpc's prediction horizon
        C = np.zeros(self.N + 1)
        for k in range(0, (self.N + 1)):
            C[k] = self.gait_scheduler(t=ts, t0=t0)
            ts += self.mpc_dt
        return C

    def path_plan(self, X_in):
        # Path planner--generate reference trajectory
        dt = self.dt
        size_mpc = int(self.N_k)  # length of MPC horizon in timesteps TODO: Perhaps N should vary wrt time?
        # timesteps given to get to target, either mpc length or based on distance (whichever is smaller)
        t_ref = int(np.minimum(size_mpc, np.linalg.norm(self.X_f[3:5] - X_in[3:5]) * 1000))
        X_ref = np.linspace(start=X_in, stop=self.X_f, num=t_ref)  # interpolate positions
        # interpolate linear velocities
        X_ref[:-1, 9] = [(X_ref[i + 1, 3] - X_ref[i, 3]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]
        X_ref[:-1, 10] = [(X_ref[i + 1, 4] - X_ref[i, 4]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]
        X_ref[:-1, 11] = [(X_ref[i + 1, 5] - X_ref[i, 5]) / dt for i in range(0, np.shape(X_ref)[0] - 1)]

        if (size_mpc - t_ref) == 0:
            pass
        elif t_ref == 0:
            X_ref = np.array(list(itertools.repeat(self.X_f, int(size_mpc))))
        else:
            X_ref = np.vstack((X_ref, list(itertools.repeat(self.X_f, int(size_mpc - t_ref)))))

        return X_ref


