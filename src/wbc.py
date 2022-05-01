"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import numpy as np
import utils
import cqp


class Control:

    def __init__(self, leg, spring, m, spr, dt=1e-3, gain=5000, null_control=False, **kwargs):
        self.cqp = cqp.Cqp(leg=leg)
        self.m = m
        self.dt = dt
        self.null_control = null_control
        self.leg = leg
        self.kp = np.zeros((3, 3))
        self.kd = np.zeros((3, 3))
        self.update_gains(gain, gain*0.02)

        self.B = np.zeros((4, 2))  # actuator selection matrix
        self.B[0, 0] = 1  # q0
        self.B[2, 1] = 1  # q2
        self.spr = spr
        self.spring_fn = spring.spring_fn

    def update_gains(self, kp, kd):
        # Use this to update wbc PD gains in real time
        m = 2  # modifier
        self.kp = np.zeros((3, 3))
        np.fill_diagonal(self.kp, [kp*m, kp*m, kp])
        self.kd = np.zeros((3, 3))
        np.fill_diagonal(self.kd, [kd*m, kd*m, kd])

    def wb_control(self, target, Q_base, force=np.zeros((3, 1))):
        leg = self.leg
        target = utils.Z(utils.Q_inv(Q_base), target[0:3]).reshape(-1, 1)  # rotate the target from world to body frame
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        x = leg.position()
        vel = np.dot(Ja, dqa).T[0:3].reshape(-1, 1)  # calculate operational space velocity vector
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma] # calculate linear acceleration term based on PD control
        x_dd_des[:3] = (np.dot(self.kp, (target - x)) + np.dot(self.kd, -vel)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))
        Mx = leg.gen_Mx()
        fx = Mx @ x_dd_des[0:3] + force
        tau = Ja.T @ fx
        u = tau.flatten()
        tau_s = self.spring_fn(leg.q) if self.spr else np.zeros(2)
        u -= tau_s  # spring counter-torque
        return u

    def wb_qp_control(self, target, Q_base, force=np.zeros((3, 1))):
        leg = self.leg
        target = utils.Z(utils.Q_inv(Q_base), target[0:3]).reshape(-1, 1)  # rotate the target from world to body frame
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        x = leg.position()
        vel = np.dot(Ja, dqa).T[0:3].reshape(-1, 1)  # calculate operational space velocity vector
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma] # calculate linear acceleration term based on PD control
        x_dd_des[:3] = (np.dot(self.kp, (target - x)) + np.dot(self.kd, -vel)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))
        r_dd_des = np.array(x_dd_des[0:3]) + force / self.m
        u = self.cqp.qpcontrol(r_dd_des)
        tau_s = self.spring_fn(leg.q) if self.spr else np.zeros(2)
        u -= tau_s  # spring counter-torque
        return u
    