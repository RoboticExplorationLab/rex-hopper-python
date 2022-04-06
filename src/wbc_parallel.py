"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import numpy as np

import utils
import cqp
# import qp


class Control:

    def __init__(self, leg, m, dt=1e-3, gain=5000, null_control=False, **kwargs):
        # self.qp = qp.Qp()
        self.cqp = cqp.Cqp(leg=leg)
        self.m = m
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
        # Use this to update wbc PD gains in real time
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
        tau = Ja.T @ fx
        u = tau.flatten()

        return u

    def wb_qp_control(self, target, Q_base, force=np.zeros((3, 1))):
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
        r_dd_des = utils.Z(utils.Q_inv(Q_base), np.array(x_dd_des[0:3]) + force/self.m)  # rotate back into body frame
        u = self.cqp.qpcontrol(r_dd_des)

        return u
