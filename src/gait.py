"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import numpy as np
import utils
import pid
import time


def raibert_x(kr, kt, pdot, pdot_ref):
    # footstep placement wrt base CoM in world frame for neutral motion
    x_f0 = pdot * kt * np.abs(pdot[2]) / 2   # takes vertical leaping speed into account
    x_f = x_f0 + kr * (pdot - pdot_ref)  # footstep placement in world frame with desired acceleration
    return x_f  # 3x1


class Gait:
    def __init__(self, model, moment, controller, leg, target, hconst, t_st, X_f,
                 use_qp=False, gain=5000, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.dt = dt
        self.controller = controller
        self.moment = moment
        self.model = model
        self.leg = leg
        self.t_st = t_st  # time spent in stance
        self.hconst = hconst
        self.target = target
        self.n_a = model["n_a"]
        self.k_wbc = gain  # wbc gain
        self.k_k = model["k_k"][0]
        self.kd_k = model["k_k"][1]
        if use_qp is True:
            self.controlf = self.controller.wb_qp_control
        else:
            self.controlf = self.controller.wb_control
        self.x_des = np.array([0, 0, 0])
        # self.pid_pdot = pid.PID1(kp=0.025, ki=0.05, kd=0.02)  # kp=0.6, ki=0.15, kd=0.02
        kp = [0.02, 0.02,  0]
        ki = [0.2,  0.2,  0]
        kd = [0.01,  0.01,  0]
        self.pid_pdot = pid.PIDn(kp=kp, ki=ki, kd=kd)
        self.z_ref = 0
        self.X_f = X_f

    def u_mpc(self, state, state_prev, X_in, X_ref, Q_base, fr):
        # mpc-based hopping
        p = X_in[0:3]
        pdot = X_in[3:6]
        p_ref = X_ref[0:3]
        pdot_ref = X_ref[3:6]
        k_wbc = self.k_wbc
        u = np.zeros(self.n_a)
        force = np.zeros((3, 1))
        hconst = self.hconst
        self.target[0] = 0  # -0.08  # adjustment for balance due to bad mockup design
        if state == 'Return':
            if state_prev == 'Leap':  # find new footstep position based on desired speed and current speed
                x_fb = np.zeros(3)
                kr = 0.4 / (np.linalg.norm(pdot_ref) + 2)  # 0.4 "speed cancellation" constant
                x_fb[0:2] = 0.5 * self.t_st * pdot[0:2] + kr * (pdot[0:2] - pdot_ref[0:2])
                self.x_des = x_fb + p  # world frame desired footstep position
                self.x_des[2] = 0  # enforce footstep location is on ground plane
            if pdot[2] >= 0:  # recognize that robot is still rising
                self.target[2] = -hconst  # pull leg up to prevent stubbing
            else:
                self.target[2] = -hconst * 6 / 3  # brace for impact
            self.controller.update_gains(k_wbc * 0.03, k_wbc * 0.006)
        elif state == 'HeelStrike':
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst * 4.5 / 3
        elif state == 'Leap':
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst * 6 / 3
            if fr is not None:
                force = fr
        else:
            raise NameError('INVALID STATE')
        Q_ref = utils.vec_to_quat2(self.x_des - p)
        Q_ref = utils.Q_inv(Q_ref)  # TODO: Shouldn't be necessary, caused by some other mistake
        u[0:2] = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u[2:], thetar, setp = self.moment.ctrl(Q_ref, Q_base)
        return u, thetar, setp

    def u_raibert(self, state, state_prev, X_in, X_ref, Q_base, fr):
        # continuous raibert hopping
        p = X_in[0:3]
        pdot = X_in[3:6]
        p_ref = X_ref[0:3]
        k_wbc = self.k_wbc
        u = np.zeros(self.n_a)
        force = np.zeros((3, 1))
        pdot_ref = -self.pid_pdot.pid_control(inp=p, setp=p_ref)  # pdot_ref = np.array([0, 0.2, 0])
        # pdot_ref = X_ref[3:6]/1000
        self.target[0] = -0.02  # -0.08  # adjustment for balance due to bad mockup design
        v_ref = X_ref - X_in

        if np.linalg.norm(self.X_f[0:2] - X_in[0:2]) >= 1:
            self.z_ref = np.arctan2(v_ref[1], v_ref[0])  # desired yaw

        k_b = (np.clip(np.linalg.norm(self.X_f[0:2] - X_in[0:2]), 0.5, 1) + 2)/3  # "Braking" gain based on dist
        hconst = self.hconst * k_b
        kr = .15 / k_b  # "speed cancellation" constant
        kt = 0.4  # gain representing leap period accounting for vertical jump velocity at toe-off

        if state == 'Return':
            if state_prev == 'Leap':  # find new footstep position based on desired speed and current speed
                self.x_des = raibert_x(kr, kt, pdot, pdot_ref) + p  # world frame desired footstep position
                self.x_des[2] = 0  # enforce footstep location is on ground plane
            if pdot[2] >= 0:  # recognize that robot is still rising
                self.target[2] = -hconst  # pull leg up to prevent stubbing
            else:
                self.target[2] = -hconst * 6 / 3  # brace for impact
            # self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.controller.update_gains(k_wbc * 0.03, k_wbc * 0.006)
        elif state == 'HeelStrike':
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst * 4.5 / 3
        elif state == 'Leap':
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst * 6 / 3
            if fr is not None:
                force = fr
        else:
            raise NameError('INVALID STATE')

        Q_ref = utils.vec_to_quat2(self.x_des - p)
        Q_ref = utils.Q_inv(Q_ref)  # TODO: Shouldn't be necessary, caused by some other mistake
        u[0:2] = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u[2:], thetar, setp = self.moment.ctrl(Q_ref, Q_base, self.z_ref)
        return u, thetar, setp

    def u_wbc_vert(self, state, state_prev, X_in, X_ref, Q_base, fr):
        k_wbc = self.k_wbc
        u = np.zeros(self.n_a)
        force = np.zeros((3, 1))
        Q_ref = utils.euler2quat([0, 0, 0])  # 2.5 * np.pi / 180
        hconst = self.hconst
        self.target[0] = -0.02
        if state == 'Return':
            # self.controller.update_gains(k_wbc * 0.03, k_wbc * 0.006)
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst * 5 / 3
        elif state == 'HeelStrike':
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst
        elif state == 'Leap':
            self.controller.update_gains(k_wbc, k_wbc * 0.02)
            self.target[2] = -hconst * 5.5 / 3
            if fr is not None:
                force = fr
        else:
            raise NameError('INVALID STATE')
        u[0:2] = -self.controlf(target=self.target, Q_base=Q_base, force=force)
        u[2:], thetar, setp = self.moment.ctrl(Q_ref, Q_base, z_ref=0)
        return u, thetar, setp

    def u_wbc_static(self, state, state_prev, X_in, X_ref, Q_base, fr):
        u = np.zeros(self.n_a)
        force = np.zeros((3, 1))
        Q_ref = utils.euler2quat([0, 0, 0])  # 2.5 * np.pi / 180
        self.target[0] = 0
        self.target[2] = -self.hconst * 5.5 / 3
        self.controller.update_gains(self.k_wbc, self.k_wbc * 0.02)
        if fr is not None:
            force = fr
        u[0:2] = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u[2:], thetar, setp = self.moment.ctrl(Q_ref, Q_base)
        return u, thetar, setp

    def u_ik_vert(self, state, state_prev, X_in, X_ref, Q_base, fr):
        time.sleep(self.dt)  # closed form inv kin runs much faster than full wbc, slow it down
        u = np.zeros(self.n_a)
        Q_ref = utils.euler2quat([0, 0, 0])  # 2.5 * np.pi / 180
        hconst = self.hconst
        k = self.k_k
        kd = self.kd_k
        if state == 'Return':
            self.target[2] = -hconst * 5 / 3
            k = self.k_k/45
            kd = self.kd_k/45
        elif state == 'HeelStrike':
            self.target[2] = -hconst
            k = self.k_k
            kd = self.kd_k
        elif state == 'Leap':
            self.target[2] = -hconst * 5.5 / 3
            k = self.k_k
            kd = self.kd_k
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        u[0:2] = (qa - self.leg.inv_kinematics(xyz=self.target[0:3])) * k + dqa * kd
        u[2:], thetar, setp = self.moment.ctrl(Q_ref, Q_base)
        return u, thetar, setp

    def u_ik_static(self, state, state_prev, X_in, X_ref, Q_base, fr):
        k = self.k_k
        kd = self.kd_k
        u = np.zeros(self.n_a)
        target = self.target
        target[2] = -self.hconst * 5 / 3
        Q_ref = utils.euler2quat([0, 0, 0])
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        # u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * k_kin + self.leg.dq * k_d
        u[0:2] = (qa - self.leg.inv_kinematics(xyz=target[0:3])) * k + dqa * kd
        u[2:], thetar, setp = self.moment.ctrl(Q_ref, Q_base)
        return u, thetar, setp
