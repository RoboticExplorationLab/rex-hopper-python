"""
Copyright (C) 2020 Benjamin Bokser
"""
import numpy as np
import rw
import transforms3d

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
        if use_qp is True:
            self.controlf = self.controller.wb_qp_control
        else:
            self.controlf = self.controller.wb_control

        self.err_sum = np.zeros(3)
        self.err_prev = np.zeros(3)

    def u_raibert(self, state, p, pdot, Q_base, fr, skip):
        # raibert hopping
        b_orient = transforms3d.quaternions.quat2mat(Q_base)
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        # print(p)
        self.target[0] = -0.05
        if state == 'Return':
            #self.controller.update_gains(1, 1 * 0.08)
            self.target[2] = -hconst*5/3
            fr = np.zeros((3, 1))
        elif state == 'HeelStrike':
            #self.controller.update_gains(45, 45*0.02)
            self.target[2] = -hconst
            fr = np.zeros((3, 1))
        elif state == 'Leap':
            #self.controller.update_gains(60, 60 * 0.02)
            self.target[2] = -hconst*5.5/3
            if skip is True:
                fr = np.zeros((3, 1))
        else:
            raise NameError('INVALID STATE')

        u = -self.controlf(leg=self.leg, target=self.target, b_orient=b_orient, force=fr)
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)

        return u, u_rw, thetar, setp

    def u_wbc_vert(self, state, Q_base, fr, skip):
        b_orient = transforms3d.quaternions.quat2mat(Q_base)
        hconst = self.hconst
        if state == 'Return':
            self.target[2] = -hconst*5/3
            fr = 0
        elif state == 'HeelStrike':
            self.target[2] = -self.hconst
            fr = 0
        elif state == 'Leap':
            self.target[2] = -hconst*5.5/3
            if skip is True:
                fr = 0
        else:
            raise NameError('INVALID STATE')

        u = -self.controlf(leg=self.leg, target=self.target, b_orient=b_orient, force=fr)

        return u

    def u_invkin(self, state, k_g, k_gd, k_a, k_ad):
        # self.target[2] = -0.5
        hconst = self.hconst
        k = k_g
        kd = k_gd
        if state == 'Return':
            # set target position
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
        # u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * k_kin + self.leg.dq * k_d
        u = (qa - self.leg.inv_kinematics(xyz=self.target[0:3])) * k + dqa * kd

        return u