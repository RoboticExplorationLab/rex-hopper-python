"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import numpy as np
import utils
import cqp


class Control:

    def __init__(self, leg, m, dt=1e-3, gain=5000, **kwargs):
        """
        Take in BODY FRAME positions or forces for the end effector and translate to joint torques
        """
        self.cqp = cqp.Cqp(leg=leg)
        self.m = m
        self.dt = dt
        self.leg = leg
        self.Q = utils.Q_inv(np.array([1, 0, 0, 0]))  # base quaternion
        self.B = np.zeros((4, 2))  # actuator selection matrix
        self.B[0, 0] = 1  # q0
        self.B[2, 1] = 1  # q2

        self.kp = np.zeros((3, 3))
        self.kd = np.zeros((3, 3))
        self.update_gains(gain, gain * 0.02)

    def update_gains(self, kp, kd):
        # Use this to update wbc PD gains in real time
        m = 2  # modifier
        self.kp = np.zeros((3, 3))
        np.fill_diagonal(self.kp, [kp*m, kp*m, kp])
        self.kd = np.zeros((3, 3))
        np.fill_diagonal(self.kd, [kd*m, kd*m, kd])

    def wb_pos_control(self, target):
        leg = self.leg
        # target = utils.Z(self.Q, target)  # rotate the target from world to body frame
        x = leg.position()
        Ja = leg.gen_jacA()  # 3x2
        vel = leg.velocity()
        x_dd_des = np.dot(self.kp, (target - x)) + np.dot(self.kd, -vel)  # .reshape((-1, 1))
        Mx = leg.gen_Mx()
        fx = Mx @ x_dd_des
        tau = Ja.T @ fx
        u = tau.flatten()
        return -u

    def wb_f_control(self, force):
        leg = self.leg
        Ja = leg.gen_jacA()  # 3x2
        # force = utils.Z(self.Q, force)  # rotate the force from world to body frame
        r_dd_des = force / self.m
        Mx = leg.gen_Mx()
        fx = Mx @ r_dd_des
        u = (Ja.T @ fx).flatten()
        return -u

    def qp_pos_control(self, target):
        leg = self.leg
        # target = utils.Z(self.Q, target)  # rotate the target from world to body frame
        x = leg.position()
        vel = leg.velocity()
        r_dd_des = np.dot(self.kp, (target - x)) + np.dot(self.kd, -vel)
        u = self.cqp.qpcontrol(r_dd_des)
        return -u

    def qp_f_control(self, force):
        # force = utils.Z(self.Q, force)  # rotate the force from world to body frame
        r_dd_des = force / self.m
        u = self.cqp.qpcontrol(r_dd_des)
        return -u

    def invkin_pos_control(self, target, kp, kd):
        # target = utils.Z(self.Q, target)  # rotate the target from world to body frame
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        u = (qa - self.leg.inv_kinematics(xyz=target[0:3])) * kp + dqa * kd
        return u