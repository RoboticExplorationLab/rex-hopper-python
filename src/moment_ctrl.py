"""
Copyright (C) 2021-2022 Benjamin Bokser
"""

import numpy as np
import pid
import utils


class MomentCtrl:
    def __init__(self, model, dt=1e-3, **kwargs):
        self.model = model
        self.a = -45 * np.pi / 180
        self.b = 45 * np.pi / 180

        if self.model["model"] == "design_rw":
            self.q = np.zeros(3)
            self.dq = np.zeros(3)
            self.ctrl = self.rw_control
            # torque PID gains
            ku = 1600  # 2000
            kp_tau = [ku, -ku, ku]
            ki_tau = [ku * 0.01, -ku * 0.01, ku * 0.02]
            kd_tau = [ku * 0.06, -ku * 0.06, ku * 0.02]
            self.pid_tau = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

            # torque PID gains (static)
            ku_ts = 1600  # 2000
            kp_tau_s = [ku_ts, -ku_ts, ku_ts]
            ki_tau_s = [ku_ts * 0.01, -ku_ts * 0.01, ku_ts * 0.02]
            kd_tau_s = [ku_ts * 0.02, -ku_ts * 0.02, ku_ts * 0.02]
            self.pid_tau_s = pid.PIDn(kp=kp_tau_s, ki=ki_tau_s, kd=kd_tau_s)

            # speed PID gains
            ku_s = 0.00001
            kp_s = [ku_s * 1, -ku_s * 1, ku_s * 2]
            ki_s = [ku_s * 0.1, -ku_s * 0.1, ku_s * 0.1]
            kd_s = [ku_s * 0, -ku_s * 0, ku_s * 0]
            self.pid_vel = pid.PIDn(kp=kp_s, ki=ki_s, kd=kd_s)

        elif self.model["model"] == "design_cmg":
            self.q = np.zeros(9)
            self.dq = np.zeros(9)
            self.ctrl = self.cmg_control
            # torque PID gains
            # for gimbals
            ku = 1600  # 2000
            kp_tau = [ku,        -ku,         ku]
            ki_tau = [ku * 0.01, -ku * 0.01,  ku * 0.02]
            kd_tau = [ku * 0.06, -ku * 0.06,  ku * 0.02]
            self.pid_tau = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

            # speed PID gains for flywheels
            ku_s = 0.01
            kp_s = [ku_s,       -ku_s,       ku_s,       -ku_s,       ku_s * 2]
            ki_s = [ku_s * 0.1, -ku_s * 0.1, ku_s * 0.1, -ku_s * 0.1, ku_s * 0.1]
            kd_s = [ku_s * 0,   -ku_s * 0,   ku_s * 0,   -ku_s * 0,   ku_s * 0]
            self.pid_vel = pid.PIDn(kp=kp_s, ki=ki_s, kd=kd_s)

    def update_state(self, q_in, qdot_in):
        self.q = q_in
        self.dq = qdot_in
        # self.dq = (self.q - self.q_previous) / self.dt
        # Make sure this only happens once per time step
        # self.d2q = (self.dq - self.dq_previous) / self.dt
        # self.q_previous = self.q
        # self.dq_previous = self.dq
        # self.d2q_previous = self.d2q

    def orient(self, Q_ref, Q_base):
        a = self.a
        b = self.b
        ref_1 = utils.z_rotate(Q_ref, a)
        ref_2 = utils.z_rotate(Q_ref, b)
        setp = np.array([ref_1 - 2 * np.pi / 180, ref_2 + 2 * np.pi / 180, 0])
        # self.setp = np.array([ref_1 - 4 * np.pi / 180, ref_2 + 4 * np.pi / 180, 0])
        theta_1 = utils.z_rotate(Q_base, a)
        theta_2 = utils.z_rotate(Q_base, b)

        theta_3 = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion

        theta = np.array([theta_1, theta_2, theta_3])
        return theta, setp

    def rw_control(self, Q_ref, Q_base):
        """
        simple reaction wheel control w/ derivative on measurement pid
        """
        dq = self.dq
        theta, setp = self.orient(Q_ref, Q_base)  # get body angle and setpoint in rw/cmg frame
        u_vel = self.pid_vel.pid_control(inp=dq.flatten(), setp=np.zeros(3))
        setp_cascaded = setp - u_vel
        u_tau = self.pid_tau.pid_control(inp=theta, setp=setp_cascaded)  # Cascaded PID Loop

        return u_tau, theta, setp_cascaded

    def cmg_control(self, Q_ref, Q_base):
        """
        simple CMG control w/ derivative on measurement pid
        """
        dq = self.dq
        theta, setp = self.orient(Q_ref, Q_base)  # get body angle and setpoint in rw/cmg frame
        qg_dot = np.array([dq[0], dq[5]])  # speed of gimbals
        qf_dot = np.array([dq[1], dq[3], dq[6], dq[8]])  # speed of gimbal flywheels
        u_vel = self.pid_vel.pid_control(inp=dq.flatten(), setp=np.zeros(3))

        u_tau = self.pid_tau.pid_control(inp=theta, setp=setp)

        return u_tau, theta, setp - u_vel