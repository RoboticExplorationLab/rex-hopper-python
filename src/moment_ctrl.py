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

            # gimbal gains
            ku = 200
            kp_tau = [ku,        -ku]
            ki_tau = [ku * 0.01, -ku * 0.01]
            kd_tau = [ku * 0.06, -ku * 0.06]
            self.pid_g = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

            # PID gains for flywheels
            ku_s = 1000
            kp_s = [ku_s,       ku_s,       ku_s,       ku_s]
            ki_s = [ku_s * 0.1, ku_s * 0.1, ku_s * 0.1, ku_s * 0.1]
            kd_s = [ku_s * 0,   ku_s * 0,   ku_s * 0,   ku_s * 0]
            self.pid_fl = pid.PIDn(kp=kp_s, ki=ki_s, kd=kd_s)

            self.pid_rwz_vel = pid.PID1(kp=0.02, ki=0.001, kd=0)
            self.pid_rwz_tau = pid.PID1(kp=1600, ki=1600*0.02, kd=1600*0.02)

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
        # q = self.q
        # qg = np.array([q[0], q[5]])  # position of gimbals
        dq = self.dq
        theta, setp = self.orient(Q_ref, Q_base)  # get body angle and setpoint in rw/cmg frame

        dqf = np.array([dq[1], dq[3], dq[6], dq[8]])  # speed of flywheels
        v_des = 6000 * (2 * np.pi / 60)
        u_fl = self.pid_fl.pid_control(inp=dqf.flatten(), setp=np.array([v_des, -v_des, v_des, -v_des]))

        u_g = self.pid_g.pid_control(inp=theta[0:2], setp=setp[0:2])  # gimbal torques

        u_rwz_vel = self.pid_rwz_vel.pid_control(inp=dq[4], setp=setp[2])
        setp_cascaded = setp[2] - u_rwz_vel
        setp[2] = setp_cascaded
        u_rwz = self.pid_rwz_tau.pid_control(inp=theta[2], setp=setp_cascaded)[0]  # Cascaded PID Loop

        # u_cmg = np.array([u_g[1], u_fl[0], 0, u_fl[1], u_rwz, u_g[0], u_fl[2], 0, u_fl[3]])
        u_cmg = np.array([u_g[1], u_fl[0], u_fl[1], u_rwz, u_g[0], u_fl[2], u_fl[3]])

        return u_cmg, theta, setp
