"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import pid
import utils


class MomentCtrl:
    def __init__(self, model, dt=1e-3, **kwargs):
        self.model = model
        self.sin45 = np.sin(-45 * np.pi / 180)
        self.a = -45 * np.pi / 180
        self.b = 45 * np.pi / 180
        self.v_des = 8000 * (2 * np.pi / 60)
        self.q = np.zeros(3)
        self.dq = np.zeros(3)
        # self.ctrl = self.rw_control
        # torque PID gains
        ku = 1600
        kp_tau = [ku,        ku,        ku*0.5]
        ki_tau = [ku * 0.1, ku * 0.1, ku * 0.01]
        kd_tau = [ku * 0.04, ku * 0.04, ku * 0.005]
        self.pid_tau = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

        # speed PID gains
        ku_s = 0.00002
        kp_s = [ku_s * 1,   ku_s * 1,   ku_s * 2]
        ki_s = [ku_s * 0.1, ku_s * 0.1, ku_s * 0.1]
        kd_s = [ku_s * 0,   ku_s * 0,   ku_s * 0]
        self.pid_vel = pid.PIDn(kp=kp_s, ki=ki_s, kd=kd_s)

    def update_state(self, q_in, dq_in):
        self.q = q_in
        self.dq = dq_in
        # Make sure this only happens once per time step
        # self.dq = (self.q - self.q_previous) / self.dt
        # self.d2q = (self.dq - self.dq_previous) / self.dt

    def orient(self, Q_ref, Q_base, z_ref):
        a = self.a
        b = self.b
        ref_1 = utils.z_rotate(Q_ref, a)
        ref_2 = utils.z_rotate(Q_ref, b)
        # d = 0 * np.pi / 180  # -0.2
        setp = np.array([ref_1, ref_2, z_ref])
        theta_1 = utils.z_rotate(Q_base, a)
        theta_2 = utils.z_rotate(Q_base, b)
        theta_3 = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion
        # theta_3 = utils.quat2euler(Q_base)[2]
        theta = np.array([theta_1, theta_2, theta_3])
        return theta, setp

    def rw_control(self, Q_ref, Q_base, z_ref):
        """
        simple reaction wheel control w/ derivative on measurement pid
        """
        dq = self.dq
        theta, setp = self.orient(Q_ref, Q_base, z_ref)  # get body angle and setpoint in rw/cmg frame
        u_vel = self.pid_vel.pid_control(inp=dq.flatten(), setp=np.zeros(3))
        setp_cascaded = setp - u_vel
        u_tau = self.pid_tau.pid_control_wrap(inp=theta, setp=setp_cascaded)  # Cascaded PID Loop
        return u_tau, theta, setp_cascaded

    def rw_torque_ctrl(self, U_in):
        """
        simple reaction wheel torque control
        rotate the torque commands in z-axis by 45 degrees
        """
        sin45 = self.sin45
        tau1 = U_in[0]
        tau2 = U_in[1]
        u1 = (tau1 - tau2) / (2 * sin45)
        u2 = (tau1 + tau2) / (2 * sin45)
        return np.array([u1, u2, -U_in[2]])


