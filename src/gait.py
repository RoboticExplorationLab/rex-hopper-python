"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import numpy as np
import rw
import transforms3d
import utils


def raibert_x(k, Ts, pdot, pdot_ref):
    x_f0 = pdot[0:2]*Ts/2  # forward placement of foot wrt base CoM in world frame for neutral motion
    x_f = x_f0 + k*(pdot[0:2] - pdot_ref[0:2])  # forward placement in world frame with desired acceleration
    # just x and y
    return x_f


class Gait:
    def __init__(self, controller, leg, target, hconst, use_qp=False, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.dt = dt
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.init_angle = np.array([self.init_alpha, self.init_beta, self.init_gamma])
        self.controller = controller
        self.leg = leg
        self.x_last = None
        self.hconst = hconst
        self.target = target  # np.hstack(np.append(np.array([0, 0, -self.hconst]), self.init_angle))
        # self.r_save = np.array([0, 0, -self.hconst])
        use_qp = False  # TODO: REMOVE
        if use_qp is True:
            self.controlf = self.controller.wb_qp_control
        else:
            self.controlf = self.controller.wb_control

        self.err_sum = np.zeros(3)
        self.err_prev = np.zeros(3)
        self.x_des = np.array([0, 0, 0])

    def u_raibert(self, state, state_prev, p, pdot, pdot_ref, Q_base, theta_prev, fr, skip):
        # raibert hopping
        # Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        Q_ref = utils.vec_to_quat2(self.x_des - p)
        hconst = self.hconst
        self.target[0] = -0.05  # adjustment for balance due to bad mockup design
        if state == 'Return':
            if state_prev != state:
                # find new footstep position based on desired speed and current speed
                kr = 0.05
                Ts = 0.2
                x_fb = np.zeros(3)
                x_fb[0:2] = raibert_x(kr, Ts, pdot, pdot_ref)  # desired footstep relative to current body CoM
                self.x_des = x_fb + p  # world frame desired footstep position
                self.x_des[2] = 0  # enforce footstep location is on ground plane
            self.controller.update_gains(1000, 1000 * 0.08)
            self.target[2] = -hconst*5/3
            fr = np.zeros((3, 1))
        elif state == 'HeelStrike':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst
            fr = np.zeros((3, 1))
        elif state == 'Leap':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst*5.5/3
            if skip is True:
                fr = np.zeros((3, 1))
        else:
            raise NameError('INVALID STATE')

        u = -self.controlf(target=self.target, Q_base=transforms3d.euler.euler2quat(0, 0, 0), force=fr)
        u_rw, self.err_sum, thetar, setp = rw.rw_control_m(self.dt, Q_ref, Q_base, self.err_sum, theta_prev)
        return u, u_rw, thetar, setp

    def u_wbc_vert(self, state, Q_base, fr, skip):
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        self.target[0] = -0.05
        if state == 'Return':
            self.controller.update_gains(1000, 1000 * 0.08)
            self.target[2] = -hconst*5/3
            fr = np.zeros((3, 1))
        elif state == 'HeelStrike':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst
            fr = np.zeros((3, 1))
        elif state == 'Leap':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst*5.5/3
            if skip is True:
                fr = np.zeros((3, 1))
        else:
            raise NameError('INVALID STATE')

        u = -self.controlf(target=self.target, Q_base=Q_base, force=fr)
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)
        return u, u_rw, thetar, setp

    def u_wbc_static(self, Q_base, fr, skip):
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        self.target[0] = -0.05
        if skip is True:
            fr = np.zeros((3, 1))
        u = -self.controlf(target=self.target, Q_base=Q_base, force=fr)
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)
        return u, u_rw, thetar, setp

    def u_invkin_vert(self, state, Q_base, k_g, k_gd, k_a, k_ad):
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        k = k_g
        kd = k_gd
        if state == 'Return':
            self.target = np.array([0, 0, -hconst*5/3])
            k = k_a
            kd = k_ad
        elif state == 'HeelStrike':
            self.target[2] = -hconst
            k = k_g
            kd = k_gd
        elif state == 'Leap':
            self.target = np.array([0, 0, -hconst*5.5/3])
            k = k_g
            kd = k_gd
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        u = (qa - self.leg.inv_kinematics(xyz=self.target[0:3])) * k + dqa * kd
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)
        return u, u_rw, thetar, setp

    def u_invkin_static(self, Q_base, k, kd):
        target = self.target
        target[2] = -self.hconst*5/3
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        # u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * k_kin + self.leg.dq * k_d
        u = (qa - self.leg.inv_kinematics(xyz=target[0:3])) * k + dqa * kd
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)
        return u, u_rw, thetar, setp