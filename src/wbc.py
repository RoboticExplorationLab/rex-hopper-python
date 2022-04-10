"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import numpy as np
import utils
import cqp


class Control:

    def __init__(self, leg, model, m, spring, dt=1e-3, gain=5000, null_control=False, **kwargs):
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

        # --- spring params --- #
        self.spring = spring
        self.ks = model["ks"]  # spring constant, N/m
        self.dir_s = model["springpolarity"]
        init_q = model["init_q"]
        self.init_q = [init_q[0], init_q[2]]
        self.L = model["linklengths"]
        L0 = self.L[0]  # .15
        L2 = self.L[2]  # .3
        self.r0 = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(2.5 * np.pi / 180))  # 0.17
        # --- #

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
        tau_s = self.spring_fn(leg.q) if self.spring else np.zeros(2)
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
        tau_s = self.spring_fn(leg.q) if self.spring else np.zeros(2)
        u -= tau_s  # spring counter-torque
        return u

    def spring_fn(self, q):
        """
        linear extension spring b/t joints 1 and 3 of parallel mechanism
        approximated by applying torques to joints 0 and 2
        """
        init_q = self.init_q
        k = self.ks
        L0 = self.L[0]
        L2 = self.L[2]
        r0 = self.r0
        if q is None:
            q0 = init_q[0]
            q2 = init_q[2]
        else:
            q0 = q[0] + init_q[0]
            q2 = q[2] + init_q[1]
        gamma = abs(q2 - q0)
        r = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(gamma))  # length of spring
        # if r < r0:
        #     print("error: incorrect spring params, r = ", r, " and r0 = ", r0, "\n gamma = ", gamma)
        T = k * (r - r0)  # spring tension force
        alpha = np.arccos((-L0 ** 2 + L2 ** 2 + r ** 2) / (2 * L2 * r))
        beta = np.arccos((-L2 ** 2 + L0 ** 2 + r ** 2) / (2 * L0 * r))
        tau_s0 = -T * np.sin(beta) * L0
        tau_s1 = T * np.sin(alpha) * L2
        tau_s = np.array([tau_s0, tau_s1]) * self.dir_s
        return tau_s