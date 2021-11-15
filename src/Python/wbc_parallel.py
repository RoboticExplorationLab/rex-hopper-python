"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import numpy as np
import transforms3d

import cqp
# import qp


class Control:

    def __init__(self, dt=1e-3, null_control=False, **kwargs):
        # self.qp = qp.Qp()
        self.cqp = cqp.Cqp()
        self.dt = dt
        self.null_control = null_control

        self.kp = np.zeros((3, 3))
        np.fill_diagonal(self.kp, 4000)

        self.kv = np.array(self.kp)*0.02

        self.kn = np.zeros((2, 2))
        np.fill_diagonal(self.kn, 100)

        self.kf = 1

        self.B = np.zeros((4, 2))  # actuator selection matrix
        self.B[0, 0] = 1  # q0
        self.B[2, 1] = 1  # q2

    def wb_control(self, leg, target, b_orient, force=0, x_dd_des=None):
        target = np.array(target).reshape(-1, 1)
        b_orient = np.array(b_orient)
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        x = np.dot(b_orient, leg.position())

        # calculate operational space velocity vector
        vel = np.dot(b_orient, (np.transpose(np.dot(Ja, dqa)))[0:3]).reshape(-1, 1)

        # calculate linear acceleration term based on PD control
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]
        x_dd_des[:3] = (np.dot(self.kp, (target[0:3] - x)) + np.dot(self.kv, -vel)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))
        # M = leg.gen_M()
        Mx = leg.gen_Mx()
        fx = Mx @ x_dd_des[0:3]
        tau = Ja.T @ fx

        C = leg.gen_C().flatten()
        G = leg.gen_G().flatten()
        B = self.B

        u = tau + ((- G - C).reshape(-1, 1).T @ B).T

        return u

    def wb_qp_control(self, leg, target, b_orient, force=0, x_dd_des=None):
        target = np.array(target).reshape(-1, 1)
        b_orient = np.array(b_orient)
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        x = np.dot(b_orient, leg.position())

        # calculate operational space velocity vector
        vel = np.dot(b_orient, (np.transpose(np.dot(Ja, dqa)))[0:3]).reshape(-1, 1)

        # calculate linear acceleration term based on PD control
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]
        x_dd_des[:3] = (np.dot(self.kp, (target[0:3] - x)) + np.dot(self.kv, -vel)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        # r_dd_des = np.array(x_dd_des[0:3])
        r_dd_des = np.array([x_dd_des[0], x_dd_des[2]]).flatten()
        x_ref = np.array([0, 0, 0, 0, 0, 0])
        x_in = np.array([leg.d2q[0], leg.d2q[1], leg.d2q[2], leg.d2q[3], 0., 0.])
        u = -self.cqp.qpcontrol(leg, r_dd_des, x_in, x_ref)
        return u


'''
    def null_signal(self):
        leg_des_angle = np.array([-30, -150])
        prop_val = ((leg_des_angle - leg.q) + np.pi) % (np.pi * 2) - np.pi
        q_des = (np.dot(self.kn, prop_val))
        #        + np.dot(self.knd, -leg.dq.reshape(-1, )))

        Fq_null = np.dot(self.Mq, q_des)

        # calculate the null space filter
        Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(self.Mq)))
        null_filter = np.eye(len(leg.q)) - np.dot(JEE.T, Jdyn_inv)
        null_signal = np.dot(null_filter, Fq_null).reshape(-1, )

        return null_signal
    '''
