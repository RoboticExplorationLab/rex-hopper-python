"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import utils

import numpy as np
import transforms3d

import cqp
# import qp


class Control:

    def __init__(self, leg, dt=1e-3, gain=5000, null_control=False, **kwargs):
        # self.qp = qp.Qp()
        self.cqp = cqp.Cqp()
        self.dt = dt
        self.null_control = null_control
        self.leg = leg
        self.kp = np.zeros((3, 3))
        # np.fill_diagonal(self.kp, gain*120)
        self.kd = np.zeros((3, 3))  # np.array(self.kp)*0.02

        self.update_gains(gain, gain*0.02)

        self.B = np.zeros((4, 2))  # actuator selection matrix
        self.B[0, 0] = 1  # q0
        self.B[2, 1] = 1  # q2

    def update_gains(self, kp, kd):
        # Use this to update wbc PID gains in real time
        m = 2  # modifier
        self.kp = np.zeros((3, 3))
        np.fill_diagonal(self.kp, [kp*m, kp*m, kp])
        self.kd = np.zeros((3, 3))
        np.fill_diagonal(self.kd, [kd*m, kd*m, kd])

    def wb_control(self, target, Q_base, force=np.zeros((3, 1))):
        leg = self.leg
        target = np.array(target).reshape(-1, 1)
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        x = utils.Z(Q_base, leg.position())  # rotate leg position from body frame to world frame

        # calculate operational space velocity vector
        vel = utils.Z(Q_base, (np.transpose(np.dot(Ja, dqa)))[0:3]).reshape(-1, 1)

        # calculate linear acceleration term based on PD control
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]
        x_dd_des[:3] = (np.dot(self.kp, (target[0:3] - x)) + np.dot(self.kd, -vel)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        Mx = leg.gen_Mx()
        fx = utils.Z(utils.Q_inv(Q_base), Mx @ x_dd_des[0:3] + force)  # rotate back into body frame for jacobian
        # fx = Mx @ x_dd_des[0:3] + force
        tau = Ja.T @ fx

        M = leg.gen_M()
        C = leg.gen_C()
        G = leg.gen_G()
        B = self.B
        u = tau  # + ((- G - C).T @ B).T
        # u = ((- G - C).T @ B).T
        '''
        qdd_new = np.linalg.solve(M, (B @ u - C - G))
        qdd_n = np.array([qdd_new[0], qdd_new[2]])
        Ja = leg.gen_jacA()
        da = leg.gen_da()  # .flatten()
        # print(np.shape(Ja), np.shape(qdd_n), np.shape(da))
        print("rdd_new in task space = ", Ja @ qdd_n + da)
        '''
        return u

    def wb_qp_control(self, target, Q_base, force=np.zeros((3, 1))):
        leg = self.leg
        target = np.array(target).reshape(-1, 1)
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        x = utils.Z(Q_base, leg.position())

        # calculate operational space velocity vector
        vel = utils.Z(Q_base, (np.transpose(np.dot(Ja, dqa)))[0:3]).reshape(-1, 1)

        # calculate linear acceleration term based on PD control
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]
        x_dd_des[:3] = (np.dot(self.kp, (target[0:3] - x)) + np.dot(self.kd, -vel)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        # 3D version
        r_dd_des = np.array(x_dd_des[0:3])
        # r_dd_des = np.array([[0, 0, -1]]).T
        # print("r_dd_des = ", r_dd_des)
        x_ref = np.array([0, 0, 0, 0, 0, 0])
        x_in = np.array([leg.d2q[0], leg.d2q[1], leg.d2q[2], leg.d2q[3], 0., 0.])
        u = self.cqp.qpcontrol(leg, r_dd_des, x_in, x_ref)

        return u
