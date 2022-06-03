"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import simulationbridge
import statemachine
import statemachine_m
import gait
import plots
import moment_ctrl
import mpc_srb
import utils
import spring

import sys
from copy import copy
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

np.set_printoptions(suppress=True, linewidth=np.nan)


class Runner:

    def __init__(self, model, dt=1e-3, ctrl_type='ik_vert', plot=False, fixed=False, spr=False,
                 record=False, scale=1, gravoff=False, direct=False, recalc=False, N_run=10000, gain=5000):

        self.g = 9.807  # gravitational acceleration, m/s2
        self.dt = dt
        self.N_run = N_run
        self.ctrl_type = ctrl_type
        self.plot = plot
        self.spr = spr
        self.fixed = fixed

        # model parameters
        self.model = model
        self.n_a = model["n_a"]
        self.L = np.array(model["linklengths"])
        self.hconst = model["hconst"]  # height constant
        self.J = model["inertia"]
        self.rh = model["rh"]
        self.mu = model["mu"]  # friction
        self.a_kt = model["a_kt"]
        self.S = model["S"]
        controller_class = model["controllerclass"]
        leg_class = model["legclass"]
        self.Jinv = np.linalg.inv(self.J)
        self.u = np.zeros(self.n_a)

        # simulator uses SE(3) states! (X). mpc uses euler-angle based states! (x). Pay attn to X vs x !!!
        self.n_X = 13
        self.n_U = 6
        self.h = 0.3 * scale  # default extended height
        self.dist = 1.2 * (N_run * dt)  # 1.2 make travel distance dependent on runtime
        self.X_0 = np.array([0,         0, self.h, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # initial conditions
        self.X_f = np.array([self.dist, 0, self.h, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T  # des final state in world frame

        self.leg = leg_class.Leg(dt=dt, model=model, g=self.g, recalc=recalc)
        self.m = self.leg.m_total
        self.tau_max1 = np.array([-50, 50]).T
        self.tau_max2 = np.array([50, 50]).T

        self.target_init = np.array([0, 0, -self.hconst])
        self.target = self.target_init[:]

        # mpc and planning-related constants
        self.t_p = 0.8  # gait period, seconds
        self.phi_switch = 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.t_c = self.t_p * self.phi_switch  # time (seconds) spent in contact
        self.t_fl = self.t_p * (1 - self.phi_switch)  # time (seconds) spent in flight
        self.N = int(160)  # mpc prediction horizon length (mpc steps)
        self.dt_mpc = 0.01  # 0.01 mpc sampling time (s), needs to be a factor of N
        self.N_dt = int(self.dt_mpc / self.dt)  # mpc sampling time (low-level timesteps), repeat mpc every x timesteps
        self.N_k = int(self.N * self.N_dt)  # total mpc prediction horizon length (in low-level timesteps)
        self.N_c = int(self.t_c / self.dt)  # number of low-level timesteps spent in contact
        self.N_f = int(self.t_fl / self.dt)  # number of low-level timesteps spent in flight
        self.t_horizon = self.N * self.dt_mpc  # time (seconds) of mpc horizon
        self.t_start = 0. * self.t_p * self.phi_switch  # start halfway through stance phase

        self.n_idx = None
        self.pf_list = None
        self.z_ref = None
        # class initializations
        self.spring = spring.Spring(model=model, spr=spr)
        self.controller = controller_class.Control(leg=self.leg, m=self.m, dt=dt, gain=gain)
        self.simulator = simulationbridge.Sim(X_0=self.X_0, model=model, spring=self.spring, dt=dt, g=self.g,
                                              fixed=fixed, record=record, scale=scale,
                                              gravoff=gravoff, direct=direct)
        self.moment = moment_ctrl.MomentCtrl(model=model, dt=dt)
        self.mpc = mpc_srb.Mpc(t=self.dt_mpc, N=self.N, m=self.m, g=self.g, mu=self.mu, Jinv=self.Jinv, rh=self.rh)
        self.gait = gait.Gait(model=model, moment=self.moment, controller=self.controller, leg=self.leg,
                              target=self.target, hconst=self.hconst, t_st=self.t_c, X_f=self.X_f, gain=gain, dt=dt)
        if self.ctrl_type == 'mpc':
            self.state = statemachine_m.Char()
        else:
            self.state = statemachine.Char()
        if self.ctrl_type == 'wbc_raibert':
            self.gaitfn = self.gait.u_raibert
        elif self.ctrl_type == 'wbc_vert':
            self.gaitfn = self.gait.u_wbc_vert
        elif self.ctrl_type == 'wbc_static':
            self.gaitfn = self.gait.u_wbc_static
        elif self.ctrl_type == 'ik_vert':
            self.gaitfn = self.gait.u_ik_vert
        elif self.ctrl_type == 'ik_static':
            self.gaitfn = self.gait.u_ik_static

        self.t_f = 0
        self.ft_saved = 0
        self.k_c = -100
        self.c_s = 0
        self.go = True
        self.ref_curve = False
        self.spline_k = None
        self.spline_i = None
        self.N_ref = None
        self.kf_list = None

    def run(self):
        n_a = self.n_a
        N_dt = self.N_dt  # repeat mpc every x seconds
        mpc_counter = copy(N_dt)
        N_run = self.N_run + 1  # number of timesteps to plot
        t = self.t_start  # time
        t0 = 0

        X_traj = np.tile(self.X_0, (N_run, 1))  # initial conditions
        U = np.zeros(self.n_U)
        U_hist = np.tile(U, (N_run, 1))  # initial conditions

        x_ref, pf_ref, C = self.ref_traj_init(x_in=utils.convert(X_traj[0, :]), xf=utils.convert(self.X_f))
        pf_ref0 = copy(pf_ref)  # original unmodified pf_ref
        x_ref0 = copy(x_ref)  # original unmodified x_ref
        '''
        if self.plot == True:
            plots.posplot_animate(p_hist=X_traj[::N_dt*10, 0:3], pf_hist=np.zeros((N_run, 3))[::N_dt*10, :],
                                  ref_traj=x_ref[::N_dt*10, 0:3], pf_ref=pf_ref[::N_dt*10, :],
                                  ref_traj0=x_ref0[::N_dt*10, 0:3], dist=self.dist)'''
        init = True
        state_prev = str("init")
        s, sh_prev = 1, 1
        c_prev = False
        u_hist = np.zeros((N_run, n_a))  # gait torque command output
        tau_hist = np.zeros((N_run, n_a))  # torque commands after actuator
        dq_hist = np.zeros((N_run, n_a))
        a_hist = np.zeros((N_run, n_a))
        v_hist = np.zeros((N_run, n_a))
        pf_hist = np.zeros((N_run, 3))
        theta_hist = np.zeros((N_run, 3))
        setp_hist = np.zeros((N_run, 3))
        grf_hist = np.zeros((N_run, 3))
        f_hist = np.zeros((N_run, 3))
        pf_des = np.zeros((N_run, 3))
        ft_hist = np.zeros(N_run)
        s_hist = np.zeros((N_run, 2))

        '''qa = np.zeros(n_a)  # init values 
        dqa = np.zeros(n_a)
        c = False
        tau = np.zeros(n_a)
        i = np.zeros(n_a)
        v = np.zeros(n_a)
        grf = np.zeros(3)
        kt = 0.8
        kr = 0.1'''

        for k in range(0, N_run):
            t += self.dt
            X_traj[k, :], qa, dqa, c, tau, i, v, grf = self.simulator.sim_run(u=self.u)  # run sim
            Q_base = X_traj[k, 3:7]
            pdot = X_traj[k, 7:10]

            self.leg.update_state(q_in=qa[0:2], Q_base=Q_base)  # enter encoder & IMU values into leg k/dynamics
            self.moment.update_state(q_in=qa[2:], dq_in=dqa[2:])

            sh = self.contact_check(c, c_prev, k)  # run contact check to make sure vibration doesn't affect sh
            self.ft_check(sh, sh_prev, t)  # flight time recorder

            pfb = self.leg.position()  # position of the foot in body frame
            pf = X_traj[k, 0:3] + utils.Z(Q_base, pfb)  # position of the foot in world frame

            s = C[k]  # self.gait_scheduler(t, t0)

            state = self.state.FSM.execute(s=s, sh=sh, go=self.go, pdot=pdot, leg_pos=pfb)

            k_f = self.kf_list[k]

            if self.ctrl_type == 'mpc':
                if sh_prev == 0 and sh == 1 and k > 10:  # if contact has just been made...
                    C = self.contact_update(C, k)  # update C to reflect new timing
                    self.pf_list[k_f, 0:2] = pf[0:2]  # update footstep list
                    pf_ref = self.footstep_update(pf_ref=pf_ref)  # update current footstep position

                if state == 'Leap' and state_prev == 'HeelStrike':
                    # update traj
                    x_ref = self.traj_update(X_in=X_traj[k, :], x_ref=x_ref, pf_ref=pf_ref, k=k, k_f=k_f, k_c=0)
                '''
                if state == 'Return' and state_prev == 'Leap':
                    pdot_ref = x_ref[k, 6:9]
                    pf_next = pdot * kt + (pdot - pdot_ref) * kr + X_traj[k, 0:3]  # The so-called.
                    self.pf_list[k_f, 0:2] = pf_next[0:2]  # update footstep list
                    pf_ref = self.footstep_update(pf_ref=pf_ref)  # update next footstep position
                    print(self.pf_list[:, 0:2], "k_f = ", k_f)
                    # update traj
                    # x_ref = self.traj_update(X_in=X_traj[k, :], x_ref=x_ref, pf_ref=pf_ref, k=k, k_f=k_f, k_c=-1)
                '''
                if mpc_counter >= N_dt:  # check if it's time to restart the mpc
                    mpc_counter = 0  # restart the mpc counter
                    Ck = C[k:(k + self.N_k):self.N_dt]  # 1D ref_traj_grab
                    x_refk = self.ref_traj_grab(ref=x_ref, k=k)
                    pf_refk = self.ref_traj_grab(ref=pf_ref, k=k)
                    x_in = utils.convert(X_traj[k, :])  # convert to mpc states
                    U = self.mpc.mpcontrol(x_in=x_in, x_ref_in=x_refk, pf_ref=pf_refk,
                                           C=Ck, f_max=self.f_max(), init=init)
                    init = False  # after the first mpc run, change init to false

                mpc_counter += 1
                U_hist[k, :] = U  # take first timestep
                self.u = self.gait.u_mpc(sh=sh, X_in=X_traj[k, :], U_in=U, pf_refk=pf_ref[k, :])

            else:
                # state = self.state.FSM.execute(s=s, sh=sh, go=self.go, pdot=pdot, leg_pos=pfb)
                self.u, theta_hist[k, :], setp_hist[k, :] = \
                    self.gaitfn(state=state, state_prev=state_prev, X_in=X_traj[k, :], x_ref=x_ref[k+100, :])

            ft_hist[k] = self.ft_saved
            grf_hist[k, :] = grf  # ground reaction force in world frame
            f_hist[k, :] = utils.Z(Q_base, U[0:3])  # body frame -> world frame output force
            u_hist[k, :] = -self.u * self.a_kt
            tau_hist[k, :] = tau
            dq_hist[k, :] = dqa
            a_hist[k, :] = i
            v_hist[k, :] = v
            pf_des[k, :] = self.gait.x_des  # desired footstep positions
            pf_hist[k, :] = pf  # foot pos in world frame
            s_hist[k, :] = [s, sh]
            state_prev = state
            sh_prev = sh
            c_prev = c
            #if k >= 2000:
            #    break

        if self.plot == True:
            # plots.thetaplot(N_run, theta_hist, setp_hist, tau_hist, dq_hist)
            # plots.tauplot(self.model, N_run, n_a, tau_hist, u_hist)
            # plots.dqplot(self.model, N_run, n_a, dq_hist)
            plots.f_plot(N_run, f_hist=f_hist, grf_hist=grf_hist, s_hist=s_hist)
            plots.posplot_3d(p_hist=X_traj[::N_dt, 0:3], pf_hist=pf_hist[::N_dt, :],
                             ref_traj=x_ref[::N_dt, 0:3], pf_ref=pf_ref[::N_dt, :],
                             pf_ref0=pf_ref0[::N_dt, :], dist=self.dist)
            plots.posplot_animate(p_hist=X_traj[::N_dt, 0:3], pf_hist=pf_hist[::N_dt, :],
                                  ref_traj=x_ref[::N_dt, 0:3], pf_ref=pf_ref[::N_dt, :],
                                  ref_traj0=x_ref0[::N_dt, 0:3], dist=self.dist)
            # plots.currentplot(N_run, n_a, a_hist)
            # plots.voltageplot(N_run, n_a, v_hist)
            # plots.etotalplot(N_run, a_hist, v_hist, dt=self.dt)

        return ft_hist

    def gait_scheduler(self, t, t0):
        phi = np.mod((t - t0) / self.t_p, 1)
        return int(phi < self.phi_switch)

    def contact_map(self, N, dt, ts, t0):
        # generate vector of scheduled contact states over the mpc's prediction horizon
        C = np.zeros(N)
        for k in range(0, N):
            C[k] = self.gait_scheduler(t=ts, t0=t0)
            ts += dt
        return C

    def ref_traj_init(self, x_in, xf):
        # Path planner--generate low-level reference trajectory for the entire run
        N_k = self.N_k  # total MPC horizon in low-level timesteps
        N_run = self.N_run
        dt = self.dt
        t_sit = 0  # timesteps spent "sitting" at goal
        t_traj = int(N_run - t_sit)  # timesteps for trajectory not including sit time
        N_ref = N_run + N_k  # timesteps for reference (extra for MPC)
        x_ref = np.linspace(start=x_in, stop=xf, num=t_traj)  # interpolate positions
        C = self.contact_map(N_ref, dt, self.t_start, 0)  # low-level contact map for the entire run

        if self.ref_curve is True:
            spline_t = np.array([0, t_traj * 0.3, t_traj])
            spline_y = np.array([x_in[1], xf[1] * 0.7, xf[1]])
            csy = CubicSpline(spline_t, spline_y)
            spline_psi = np.array([0, np.sin(45 * np.pi / 180) * 0.7, np.sin(45 * np.pi / 180)])
            cspsi = CubicSpline(spline_t, spline_psi)
            for k in range(t_traj):
                x_ref[k, 1] = csy(k)  # create evenly spaced sample points of desired trajectory
                x_ref[k, 5] = cspsi(k)  # create evenly spaced sample points of desired trajectory
            # interpolate angular velocity
            x_ref[:-1, 11] = [(x_ref[i + 1, 11] - x_ref[i, 11]) / dt for i in range(N_run - 1)]

        x_ref = np.vstack((x_ref, np.tile(xf, (N_k + t_sit, 1))))  # sit at the goal
        period = self.t_p  # *1.2  # * self.dt_mpc / 2
        amp = self.t_p / 4  # amplitude
        phi = np.pi * 3 / 2  # np.pi*3/2  # phase offset
        # make height sine wave
        sine_wave = np.array([x_in[2] + amp + amp * np.sin(2 * np.pi / period * (i * dt) + phi) for i in range(N_ref)])
        peaks = find_peaks(sine_wave)[0]
        troughs = find_peaks(-sine_wave)[0]
        spline_k = np.sort(np.hstack((peaks, troughs)))  # independent variable
        spline_k = np.hstack((0, spline_k))  # add initial footstep idx based on first timestep
        spline_k = np.hstack((spline_k, N_ref - 1))  # add final footstep idx based on last timestep
        n_k = np.shape(spline_k)[0]
        spline_i = np.zeros((n_k, 3))
        spline_i[:, 0:2] = x_ref[spline_k, 0:2]
        spline_i[:, 2] = sine_wave[spline_k]  # dependent variable
        ref_spline = CubicSpline(spline_k, spline_i, bc_type='clamped')  # generate cubic spline
        x_ref[:, 0:3] = [ref_spline(k) for k in range(N_ref)]  # create z-spline

        x_ref[:-1, 6:9] = [(x_ref[i + 1, 0:3] - x_ref[i, 0:3]) / dt for i in range(N_ref - 1)]  # interpolate linear vel

        idx = troughs - 115  # indices of footstep positions
        idx = np.hstack((0, idx))  # add initial footstep idx based on first timestep
        idx = np.hstack((idx, N_ref - 1))  # add final footstep idx based on last timestep

        self.n_idx = np.shape(idx)[0]
        pf_ref = np.zeros((N_ref, 3))
        self.pf_list = np.zeros((self.n_idx, 3))

        k_f = 0
        for k in range(1, N_ref):  # generate pf_list
            if C[k - 1] == 1 and C[k] == 0 and k_f < self.n_idx:
                k_f += 1
                self.pf_list[k_f, 0:2] = x_ref[idx[k_f], 0:2]  # store reference footsteps here

        self.N_ref = N_ref
        self.spline_k = spline_k
        self.spline_i = spline_i
        self.kf_list = np.zeros(N_ref, dtype=int)
        self.k_f_update(C)  # update kf_list
        pf_ref = self.footstep_update(pf_ref)
        return x_ref, pf_ref, C

    def k_f_update(self, C):
        """ update kf_list with new contact map """
        k_f = int(0)
        for k in range(1, self.N_ref):
            if C[k - 1] == 1 and C[k] == 0 and k_f < (self.n_idx - 1):  # when flight starts
                k_f += 1
            self.kf_list[k] = int(k_f)

    def contact_update(self, C, k):
        """ shift contact map. Use if contact has been made early or was previously late """
        N = self.N_ref  # np.shape(C)[0]  # size of contact map
        C[k:] = self.contact_map(N=(N-k), dt=self.dt, ts=0, t0=0)  # just rewrite it assuming contact starts now
        self.k_f_update(C)  # update kf_list
        return C

    def footstep_update(self, pf_ref):
        """ rewrite pf_ref based on actual current footstep location """
        N = self.N_ref  # size of contact map
        for k in range(0, N):  # regen pf_ref
            k_f = self.kf_list[k]
            pf_ref[k, :] = self.pf_list[k_f, :]  # add reference footsteps to appropriate timesteps
        return pf_ref

    def ref_traj_grab(self, ref, k):  # Grab appropriate timesteps of pre-planned trajectory for mpc
        return ref[k:(k + self.N_k):self.N_dt, :]  # change to mpc-level timesteps

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
        """if contact has just been made, freeze contact detection to True for x timesteps
        or if contact has just been lost, freeze contact detection to False for x timesteps
        protects against vibration/bouncing-related bugs """

        if c_prev != c and self.go == True:
            self.k_c = k  # timestep at contact change
            self.c_s = c  # saved contact value

        if k - self.k_c <= 10:  # freeze contact value
            c = self.c_s
            self.go = False
        else:
            self.go = True

        return c

    def f_max(self):
        jac_inv = np.linalg.pinv(self.leg.gen_jacA())
        f_max1 = np.absolute(self.tau_max1 @ jac_inv)  # max output force at default pose
        f_max2 = np.absolute(self.tau_max2 @ jac_inv)  # max output force at default pose
        return np.maximum(f_max1, f_max2)

    def traj_update(self, X_in, x_ref, pf_ref, k, k_f, k_c):
        # create new ref traj online
        N_k = self.N_k
        N_ref = self.N_ref
        k_s = int(k_f * 2) + k_c  # get spline position index for the current timestep
        spline_k = self.spline_k
        np.set_printoptions(threshold=sys.maxsize)
        self.spline_i[k_s, 1] = X_in[1]  # update spline with new CoM position
        # print("spline_i = ", self.spline_i, "x_in", X_in[0:3], "k_s = ", k_s)
        ref_spline = CubicSpline(spline_k, self.spline_i, bc_type='clamped')  # generate cubic spline
        x_ref[:, 0:3] = [ref_spline(k) for k in range(N_ref)]  # create z-spline
        # interpolate linear vel
        x_ref[k:(k+N_k), 6:9] = [(x_ref[k + i + 1, 0:3] - x_ref[k + i, 0:3]) / self.dt for i in range(N_k)]
        '''
        for i in range(k, k+N_k):  # roll compensation
            pf_b = pf_ref[i, 0:3] - x_ref[i, 0:3]    # find vector b/t x_ref and pf_ref
            # this only works if robot is facing exactly forward
            x_ref[i, 3] = np.pi/2 + utils.wrap_to_pi(np.arctan2(pf_b[2], pf_b[0]))  # get xz plane angle TODO: Check
            # print(x_ref[i, 3])'''

        # interpolate angular velocity
        x_ref[k:(k+N_k), 9] = [(x_ref[k + i + 1, 9] - x_ref[k + i, 9]) / self.dt for i in range(N_k)]

        return x_ref
