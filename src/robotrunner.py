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

from copy import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, model, dt=1e-3, ctrl_type='ik_vert', plot=False, fixed=False, spr=False,
                 record=False, scale=1, gravoff=False, direct=False, recalc=False, t_run=10000, gain=5000):

        self.g = 9.807  # gravitational acceleration, m/s2
        self.dt = dt
        self.t_run = t_run
        self.ctrl_type = ctrl_type
        self.plot = plot
        self.spr = spr
        self.fixed = fixed

        # model parameters
        self.model = model
        self.u = np.zeros(model["n_a"])
        self.L = np.array(model["linklengths"])
        self.n_a = model["n_a"]
        self.hconst = model["hconst"]  # height constant
        controller_class = model["controllerclass"]
        leg_class = model["legclass"]
        self.J = model["inertia"]
        self.rh = model["rh"]
        self.Jinv = np.linalg.inv(self.J)
        self.mu = model["mu"]  # friction

        self.spline = True

        # simulator uses SE(3) states! (X). mpc uses euler-angle based states! (x). Pay attn to X vs x !!!
        self.n_X = 13
        self.n_U = 6
        self.X_0 = np.array([0, 0, 0.6 * scale, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # initial conditions
        self.X_f = np.array([2, 0, 0.6 * scale, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # desired final state in world frame

        self.leg = leg_class.Leg(dt=dt, model=model, g=self.g, recalc=recalc)
        self.m = self.leg.m_total

        self.target_init = np.array([0, 0, -self.hconst])
        self.target = self.target_init[:]

        self.t_p = 0.8  # 0.8 gait period, seconds
        self.phi_switch = 0.5  # 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.N = 40  # mpc prediction horizon length (mpc steps)
        self.mpc_dt = 0.02  # mpc sampling time (s), needs to be a factor of N
        self.mpc_factor = int(self.mpc_dt / self.dt)  # mpc sampling time (timesteps), repeat mpc every x timesteps
        self.N_time = self.N * self.mpc_dt  # mpc horizon time
        self.N_k = int(self.N * self.mpc_factor)  # total mpc prediction horizon length (low-level timesteps)
        self.t_start = 0.5 * self.t_p * self.phi_switch  # start halfway through stance phase
        self.t_st = self.t_p * self.phi_switch  # time spent stance

        # class initializations
        self.spring = spring.Spring(model)
        self.controller = controller_class.Control(leg=self.leg, spring=self.spring, m=self.m,
                                                   spr=spr, dt=dt, gain=gain)
        self.simulator = simulationbridge.Sim(X_0=self.X_0, model=model, spring=self.spring, dt=dt, g=self.g,
                                              fixed=fixed, spr=spr, record=record, scale=scale,
                                              gravoff=gravoff, direct=direct)
        self.moment = moment_ctrl.MomentCtrl(model=model, dt=dt)
        self.mpc = mpc.Mpc(t=self.mpc_dt, N=self.N, m=self.m, g=self.g, mu=self.mu, Jinv=self.Jinv, rh=self.rh)
        self.gait = gait.Gait(model=model, moment=self.moment, controller=self.controller, leg=self.leg,
                              target=self.target, hconst=self.hconst, t_st=self.t_st, X_f=self.X_f,
                              use_qp=False, gain=gain, dt=dt)

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

        self.t_f = 0
        self.ft_saved = 0
        self.k_c = -100
        self.c_s = 0
        self.go = True

    def run(self):
        n_a = self.n_a
        mpc_factor = self.mpc_factor  # repeat mpc every x seconds
        mpc_counter = copy(mpc_factor)
        t_run = self.t_run + 1  # number of timesteps to plot
        t = self.t_start  # time
        t0 = 0

        X_traj = np.tile(self.X_0, (t_run, 1))  # initial conditions
        U = np.zeros(self.n_U)
        U_hist = np.tile(U, (t_run, 1))  # initial conditions

        x_ref, pf_ref = self.ref_traj_init(x_in=utils.convert(X_traj[0, :]), xf=utils.convert(self.X_f))
        '''
        if self.plot == True:
            plots.posplot_animate(p_ref=self.X_f[0:3], p_hist=X_traj[::mpc_factor, 0:3],
                                  ref_traj=x_ref[::mpc_factor, 0:3], pf_ref=pf_ref[::mpc_factor, :])'''
        init = True
        first_contact = 0
        state_prev = str("init")
        s, sh_prev = 0, 0
        c_prev = False

        tauhist = np.zeros((t_run, n_a))
        dqhist = np.zeros((t_run, n_a))
        ahist = np.zeros((t_run, n_a))
        vhist = np.zeros((t_run, n_a))
        pfhist = np.zeros((t_run, 3))
        thetahist = np.zeros((t_run, 3))
        setphist = np.zeros((t_run, 3))
        grfhist = np.zeros((t_run, 3))
        pfdes = np.zeros((t_run, 3))
        fthist = np.zeros(t_run)
        s_hist = np.zeros((t_run, 2))

        for k in range(0, t_run):
            t += self.dt

            X, qa, dqa, c, tau, f_sens, tau_sens, i, v, grf = self.simulator.sim_run(u=self.u)  # run sim
            Q_base = X[3:7]
            pdot = X[7:10]
            X_traj[k, :] = X  # update state from simulator

            self.leg.update_state(q_in=qa[0:2], Q_base=Q_base)  # enter encoder & IMU values into leg k/dynamics
            self.moment.update_state(q_in=qa[2:], dq_in=dqa[2:])

            sh = self.contact_check(c, c_prev, k)  # run contact check to make sure vibration doesn't affect sh
            self.ft_check(sh, sh_prev, t)  # flight time recorder

            if sh == 1 and first_contact == 0:
                t0 = t  # starting time begins when robot first makes contact
                first_contact = 1  # ensure this doesn't trigger again
            elif sh == 1 and first_contact == 1:
                s = self.gait_scheduler(t, t0)
            else:
                s = 0

            state = self.state.FSM.execute(s=s, sh=sh, go=self.go, pdot=pdot, leg_pos=self.leg.position())

            pf = X_traj[k, 0:3] + utils.Z(Q_base, self.leg.position())  # position of the foot in world frame

            if self.ctrl_type == 'mpc' and first_contact == 1:
                if mpc_counter == mpc_factor:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    C = self.gait_map(self.N, self.mpc_dt, t, t0)
                    x_refk = self.ref_traj_grab(x_ref=x_ref, k=k)
                    pf_refk = self.ref_traj_grab(x_ref=pf_ref, k=k)
                    x_in = utils.convert(X_traj[k, :])  # convert to mpc states
                    U = self.mpc.mpcontrol(x_in=x_in, x_ref_in=x_refk, pf=pf_refk, C=C, init=init)
                    init = False  # after the first mpc run, change init to false

                mpc_counter += 1
                U_hist[k, :] = U[0, :]  # * s  # take first timestep

            self.u, thetar, setp = self.gaitfn(state=state, state_prev=state_prev,
                                               X_in=X_traj[k, :], p_ref=x_ref[100, 0:3], U_in=U, grf=grf, s=s)

            grfhist[k, :] = grf.flatten()  # ground reaction force
            fthist[k] = self.ft_saved
            setphist[k, :] = setp
            thetahist[k, :] = thetar
            tauhist[k, :] = tau
            dqhist[k, :] = dqa
            ahist[k, :] = i
            vhist[k, :] = v
            pfdes[k, :] = self.gait.x_des  # desired footstep positions
            pfhist[k, :] = pf  # foot pos in world frame
            s_hist[k, :] = [s, sh]
            state_prev = state
            sh_prev = sh
            c_prev = c

        if self.plot == True:
            plots.thetaplot(t_run, thetahist, setphist)
            plots.tauplot(self.model, t_run, n_a, tauhist)
            plots.dqplot(self.model, t_run, n_a, dqhist)
            plots.fplot(t_run, phist=X_traj[:, 0:3], fhist=U_hist[:, 0:3], shist=s_hist)
            plots.grfplot(t_run, X_traj[:, 0:3], grfhist, fthist)
            plots.posplot_3d(p_ref=self.X_f[0:3], phist=X_traj[:, 0:3], pfdes=pfdes)
            # plots.posplot(p_ref=self.X_f[0:3], phist=phist, pfdes=pfdes)
            # plots.currentplot(t_run, n_a, ahist)
            # plots.voltageplot(t_run, n_a, vhist)
            plots.etotalplot(t_run, ahist, vhist, dt=self.dt)

        return fthist

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        return phi < self.phi_switch  # TODO: make sure this works

    def gait_map(self, N, dt, ts, t0):
        # generate vector of scheduled contact states over the mpc's prediction horizon
        C = np.zeros(N)
        for k in range(0, N):
            C[k] = self.gait_scheduler(t=ts, t0=t0)
            ts += dt
        return C

    def ref_traj_init(self, x_in, xf):
        # Path planner--generate low-level reference trajectory for the entire run
        N_k = self.N_k  # total MPC horizon in low-level timesteps
        t_run = self.t_run
        dt = self.dt
        t_sit = 0  # timesteps spent "sitting" at goal
        t_traj = int(t_run - t_sit)  # timesteps for trajectory not including sit time
        t_ref = t_run + N_k  # timesteps for reference (extra for MPC)
        x_ref = np.linspace(start=x_in, stop=xf, num=t_traj)  # interpolate positions

        if self.spline is True:
            spline_t = np.array([0, t_traj * 0.3, t_traj])
            spline_y = np.array([x_in[1], xf[1] * 0.7, xf[1]])
            csy = CubicSpline(spline_t, spline_y)
            spline_psi = np.array([0, np.sin(45 * np.pi / 180) * 0.7, np.sin(45 * np.pi / 180)])
            cspsi = CubicSpline(spline_t, spline_psi)
            for k in range(t_traj):
                x_ref[k, 1] = csy(k)  # create evenly spaced sample points of desired trajectory
                x_ref[k, 5] = cspsi(k)  # create evenly spaced sample points of desired trajectory
                # interpolate angular velocity
            x_ref[:-1, 11] = [(x_ref[i + 1, 11] - x_ref[i, 11]) / dt for i in range(t_run - 1)]

        x_ref = np.vstack((x_ref, np.tile(xf, (N_k + t_sit, 1))))  # sit at the goal
        period = self.t_p  # *1.2  # * self.mpc_dt / 2
        amp = self.t_p / 4  # amplitude
        phi = np.pi * 3 / 2  # np.pi*3/2  # phase offset
        # make height sine wave
        x_ref[:, 2] = [x_in[2] + amp + amp * np.sin(2 * np.pi / period * (i * dt) + phi) for i in range(t_ref)]
        x_ref[:-1, 6:9] = [(x_ref[i + 1, 0:3] - x_ref[i, 0:3]) / dt for i in range(t_ref - 1)]  # interpolate linear vel

        C = self.gait_map(t_ref, dt, self.t_start, 0)  # low-level contact map for the entire run
        idx_pf = find_peaks(-x_ref[:, 2])[0]  # indexes of footstep positions
        idx_pf = np.hstack((0, idx_pf))  # add initial footstep idx based on first timestep
        idx_pf = np.hstack((idx_pf, t_ref - 1))  # add final footstep idx based on last timestep
        n_idx = np.shape(idx_pf)[0]

        pf_ref = np.zeros((t_ref, 3))
        kf = 0
        for k in range(1, t_ref):
            if C[k - 1] == 0 and C[k] == 1 and kf < n_idx:
                kf += 1
            pf_ref[k, 0:2] = x_ref[idx_pf[kf], 0:2]
        # np.set_printoptions(threshold=sys.maxsize)
        return x_ref, pf_ref

    def ref_traj_grab(self, x_ref, k):
        # Grab appropriate timesteps of pre-planned trajectory for mpc
        return x_ref[k:(k + self.N_k):self.mpc_factor, :]  # change to mpc-level timesteps

    def ft_check(self, sh, sh_prev, t):
        # flight time recorder
        if sh == 0 and sh_prev == 1:
            self.t_f = t  # time of flight
        if sh == 1 and sh_prev == 0:
            t_ft = t - self.t_f  # last flight time
            if t_ft > 0.1:  # ignore flight times of less than 0.1 second (these are foot bounces)
                print("flight time = ", t_ft)
                self.ft_saved = t_ft  # save flight time
        return None

    def contact_check(self, c, c_prev, k):
        # if contact has just been made, freeze contact detection to True for x timesteps
        # or if contact has just been lost, freeze contact detection to False for x timesteps
        # protects against vibration/bouncing-related bugs

        if c_prev != c and self.go == True:
            self.k_c = k  # timestep at contact change
            self.c_s = c  # saved contact value

        if k - self.k_c <= 10:  # freeze contact value
            c = self.c_s
            self.go = False
        else:
            self.go = True

        return c
