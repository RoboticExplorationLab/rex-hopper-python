"""
Copyright (C) 2020 Benjamin Bokser
"""
import numpy as np


class Gait:
    def __init__(self, controller, leg, target, hconst, use_qp=True, dt=1e-3, **kwargs):

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
        self.r_save = np.array([0, 0, -self.hconst])
        if use_qp is True:
            self.controlf = self.controller.wb_qp_control
        else:
            self.controlf = self.controller.wb_control

    def u(self, state, prev_state, r_in, r_d, delp, b_orient, fr_mpc, skip):
        hconst = self.hconst
        if state == 'Return':
            # set target position
            self.target = np.hstack(np.append(np.array([0, 0, -hconst*5/3]), self.init_angle))
            # calculate wbc control signal
            u = -self.controlf(leg=self.leg, target=self.target, b_orient=b_orient, force=0)

        elif state == 'HeelStrike':
            self.target[2] = -self.hconst
            u = -self.controlf(leg=self.leg, target=self.target, b_orient=b_orient, force=0)

        elif state == 'Leap':
            # calculate wbc control signal
            if skip is True:
                fr_mpc = 0

            self.target = np.hstack(np.append(np.array([0, 0, -hconst*5.5/3]), self.init_angle))
            u = -self.controlf(leg=self.leg, target=self.target, b_orient=b_orient, force=fr_mpc)

        else:
            raise NameError('INVALID STATE')

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