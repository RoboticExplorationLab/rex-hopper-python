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
import qp_point
import utils
import spring

# import sys
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
        self.dist = 1 * (N_run * dt)  # make travel distance dependent on runtime
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
        self.phi_switch = 0.5  # 0.5  # switching phase, must be between 0 and 1. Percentage of gait spent in contact.
        self.t_st = self.t_p * self.phi_switch  # time (seconds) spent in contact
        self.N = int(40)  # mpc prediction horizon length (mpc steps)
        self.dt_mpc = 0.01  # mpc sampling time (s), needs to be a factor of N
        self.N_dt = int(self.dt_mpc / self.dt)  # mpc sampling time (low-level timesteps), repeat mpc every x timesteps
        self.N_k = int(self.N * self.N_dt)  # total mpc prediction horizon length (in low-level timesteps)
        self.N_c = int(self.t_st / self.dt)  # number of low-level timesteps spent in contact
        self.t_horizon = self.N * self.dt_mpc  # time (seconds) of mpc horizon
        self.t_start = 0.5 * self.t_p * self.phi_switch  # start halfway through stance phase

        self.dt_qp = self.t_p * self.phi_switch / 2  # point mass qp sampling time
        # make this cover 4x as many seconds to avoid being too short
        self.Np = int(self.t_horizon * 2 / self.dt_qp)  # point mass traj opt prediction horizon length (in qp steps)
        self.Np_dt = int(self.dt_mpc / self.dt)  # qp sampling time (in low-level timesteps)
        self.Np_k = int(self.Np * self.Np_dt)  # point mass prediction horizon length (in low-level timesteps)

        self.kf_ref = None
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

        # point mass dynamics
        self.Ap = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        self.Bp = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [1 / self.m, 0, 0],
                           [0, 1 / self.m, 0],
                           [0, 0, 1 / self.m]])
        self.Gp = np.array([[0, 0, 0, 0, 0, -self.g]]).T

        self.qp_point = qp_point.Qp(t=self.dt_qp, A=self.Ap, B=self.Bp, G=self.Gp,
                                    N=self.Np, m=self.m, mu=self.mu, g=self.g)
        self.gait = gait.Gait(model=model, moment=self.moment, controller=self.controller, leg=self.leg,
                              target=self.target, hconst=self.hconst, t_st=self.t_st, X_f=self.X_f, gain=gain, dt=dt)

        if self.ctrl_type == 'mpc':
            self.state = statemachine_m.Char()
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
        self.ref_curve = False
        self.ref_spline = None

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
        if self.plot == True:
            plots.posplot_animate(p_hist=X_traj[::N_dt*10, 0:3], pf_hist=np.zeros((N_run, 3))[::N_dt*10, :],
                                  ref_traj=x_ref0[::N_dt*10, 0:3], pf_ref=pf_ref[::N_dt*10, :], dist=self.dist)
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
        grf = np.zeros(3)'''

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

            if self.ctrl_type == 'mpc':
                state = self.state.FSM.execute(s=s, sh=sh)
                # if (state_prev == "Flight" and state == "Early") or (state_prev == "Late" and state == "Contact"):
                if sh_prev == 0 and sh == 1 and k > 10:  # if contact has just been made...
                    C = self.contact_update(C, k)  # update C to reflect new timing
                    # generate new ref traj
                    x_ref[k:, :], pf_ref = self.traj_opt(k=k, X_in=X_traj[k, :], x_ref0=x_ref0,
                                                         C=C, pf_ref=pf_ref, pf=pf)

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
                state = self.state.FSM.execute(s=s, sh=sh, go=self.go, pdot=pdot, leg_pos=pfb)
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
            # if k >= 1000:
            #    break

        if self.plot == True:
            # plots.thetaplot(N_run, theta_hist, setp_hist, tau_hist, dq_hist)
            plots.tauplot(self.model, N_run, n_a, tau_hist, u_hist)
            # plots.dqplot(self.model, N_run, n_a, dq_hist)
            plots.f_plot(N_run, f_hist=f_hist, grf_hist=grf_hist, s_hist=s_hist)
            plots.posplot_3d(p_hist=X_traj[::N_dt, 0:3], pf_hist=pf_hist[::N_dt, :],
                             ref_traj=x_ref[::N_dt, 0:3], pf_ref=pf_ref[::N_dt, :],
                             pf_ref0=pf_ref0[::N_dt, :], dist=self.dist)
            plots.posplot_animate(p_hist=X_traj[::N_dt, 0:3], pf_hist=pf_hist[::N_dt, :],
                                  ref_traj=x_ref[::N_dt, 0:3], pf_ref=pf_ref[::N_dt, :], dist=self.dist)
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
        n_idx = np.shape(idx)[0]
        pf_ref = np.zeros((N_ref, 3))
        kf = 0
        pf_list = np.zeros((n_idx, 3))

        self.kf_ref = np.zeros(N_ref)
        for k in range(1, N_ref):
            if C[k - 1] == 1 and C[k] == 0 and kf < n_idx:
                kf += 1
                pf_list[kf, 0:2] = x_ref[idx[kf], 0:2]  # store reference footsteps here
            self.kf_ref[k] = kf
            pf_ref[k, 0:2] = x_ref[idx[kf], 0:2]  # add reference footsteps to appropriate timesteps
        self.n_idx = n_idx
        self.pf_list = pf_list
        # np.set_printoptions(threshold=sys.maxsize)
        return x_ref, pf_ref, C

    def contact_update(self, C, k):
        """ shift contact map and rewrite corresponding pf_ref
        Use if contact has been made early or was previously late """
        N = np.shape(C)[0]  # size of contact map
        C[k:] = self.contact_map(N=(N-k), dt=self.dt, ts=0, t0=0)  # just rewrite it assuming contact starts now
        return C

    def footstep_update(self, C, pf_ref, pf, k):
        """ shift contact map and rewrite corresponding pf_ref
                Use if contact has been made early or was previously late """
        N = np.shape(C)[0]  # size of contact map
        kf = int(self.kf_ref[k])  # get footstep index for the current timestep
        self.pf_list[kf, 0:2] = pf[0:2]  # update footstep list
        for i in range(k, N):  # regen pf_ref
            if C[i - 1] == 1 and C[i] == 0 and kf < (self.n_idx - 1):
                kf += 1
            # self.kf_ref[k] = kf  # update footstep indices
            pf_ref[i, :] = self.pf_list[kf, :]  # add reference footsteps to appropriate timesteps
        # print("pf_list new = ", self.pf_list)
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

    def traj_opt(self, k, X_in, x_ref0, C, pf_ref, pf):
        # use point mass mpc to create new ref traj online
        Np_traj = self.Np_k  # + 1
        xp_traj = np.zeros((Np_traj, 6))
        x_ref = np.zeros((Np_traj, 12))
        f_pred_hist = np.zeros((self.Np+1, 3))
        p_pred_hist = np.zeros((self.Np+1, 3))

        p_in = X_in[0:3]
        dp_in = X_in[7:10]
        p_ref = x_ref0[:, 0:3]
        dp_ref = x_ref0[:, 6:9]

        xp_in = np.hstack((p_in, dp_in))
        xp_ref = np.hstack((p_ref, dp_ref))

        xp_refk = xp_ref[k:(k + Np_traj):self.Np_dt, :]
        Cpk = self.Ck_grab(C, k, self.Np, Np_traj, self.Np_dt)  # C[k:(k + Np_traj):self.Np_dt]  # 1D ref_traj_grab
        u_pred, x_pred = self.qp_point.qpcontrol(x_in=xp_in, x_ref=xp_refk, Ck=Cpk)

        f_hist = np.zeros((Np_traj, 3))
        j = self.Np_dt

        for i in range(0, self.Np):
            f_hist[int(i * j):int(i * j + j), :] = np.tile(u_pred[i, :], (j, 1))

        kf = int(self.kf_ref[k])  # get footstep index for the current timestep
        for i in range(1, self.Np - 1):  # update footstep list
            if Cpk[i - 1] == 0 and Cpk[i] == 1:
                p_pred_hist[i + 1, :] = x_pred[i + 1, 0:3]
                f_pred_hist[i + 1, :] = u_pred[i, 0:3] + u_pred[i + 1, 0:3]
                self.pf_list[kf, :] = utils.projection(p_pred_hist[i + 1, :], f_pred_hist[i + 1, :])
                kf += 1

        for k in range(0, Np_traj):
            xp_traj[k + 1, :] = self.rk4(xk=xp_traj[k, :], uk=f_hist[k, :])

        x_ref[:, 0:3] = xp_traj[:, 0:3]
        x_ref[:, 6:9] = xp_traj[:, 3:6]

        pf_ref = self.footstep_update(C, pf_ref, pf, k)  # update pf_ref to reflect new timing & ref traj

        # roll compensation
        for k in range(0, Np_traj):
            # find vector b/t x_ref and pf_ref
            pf_b = x_ref[k, 0:3] - pf_ref[k, 0:3]
            # get xz plane angle
            x_ref[k, 0] = np.arctan2(pf_b[2], pf_b[0])  # TODO: Check

        return x_ref, pf_ref

    def point_dynamics_ct(self, X, U):
        # CT dynamics X -> dX
        A = self.Ap
        B = self.Bp
        G = self.Gp
        X_next = A @ X + B @ U + G
        return X_next

    def rk4(self, xk, uk):
        # RK4 integrator solves for new X
        dynamics = self.point_dynamics_ct
        h = self.dt
        f1 = dynamics(xk, uk)
        f2 = dynamics(xk + 0.5 * h * f1, uk)
        f3 = dynamics(xk + 0.5 * h * f2, uk)
        f4 = dynamics(xk + h * f3, uk)
        return xk + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

    def Ck_grab(self, C, k, N, N_k, N_dt):  # Grab appropriate timesteps of pre-planned trajectory for mpc
        C_ref = C[k:(k + N_k)]
        Ck = np.array([np.median(C_ref[(i * N_dt):(i * N_dt + N_dt)]) for i in range(N)])
        return Ck  # change to mpc-level timesteps
