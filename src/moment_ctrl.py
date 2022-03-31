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
        self.v_des = 8000 * (2 * np.pi / 60)
        if self.model["model"] == "design_rw":
            self.q = np.zeros(3)
            self.dq = np.zeros(3)
            self.ctrl = self.rw_control
            # torque PID gains
            ku = 1600  # 2000
            kp_tau = [ku,        ku,        ku*0.5]
            ki_tau = [ku * 0.01, ku * 0.01, ku * 0.01]
            kd_tau = [ku * 0.06, ku * 0.06, ku * 0.02]
            self.pid_tau = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

            # speed PID gains
            ku_s = 0.00001
            kp_s = [ku_s * 1,   ku_s * 1,   ku_s * 2]
            ki_s = [ku_s * 0.1, ku_s * 0.1, ku_s * 0.1]
            kd_s = [ku_s * 0,   ku_s * 0,   ku_s * 0]
            self.pid_vel = pid.PIDn(kp=kp_s, ki=ki_s, kd=kd_s)

        elif self.model["model"] == "design_cmg":
            self.q = np.zeros(9)
            self.dq = np.zeros(9)
            self.ctrl = self.cmg_control

            # gimbal position gains
            ku = 0.001
            kp_tau = [ku,        ku]
            ki_tau = [ku * 0.01, ku * 0.01]
            kd_tau = [ku * 0.02, ku * 0.02]
            self.pid_g_pos = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

            # gimbal gains
            ku = 1
            kp_tau = [ku,       ku]
            ki_tau = [ku * 0.1, ku * 0.1]
            kd_tau = [ku * 0.2, ku * 0.2]
            self.pid_g = pid.PIDn(kp=kp_tau, ki=ki_tau, kd=kd_tau)

            # PID gains for flywheels
            ku_s = 1000
            kp_s = [ku_s,       ku_s,       ku_s,       ku_s]
            ki_s = [ku_s * 0.1, ku_s * 0.1, ku_s * 0.1, ku_s * 0.1]
            kd_s = [ku_s * 0,   ku_s * 0,   ku_s * 0,   ku_s * 0]
            self.pid_fl = pid.PIDn(kp=kp_s, ki=ki_s, kd=kd_s)

            self.pid_rwz_vel = pid.PID1(kp=0.0002, ki=0.00001, kd=0)
            self.pid_rwz_tau = pid.PID1(kp=1600, ki=1600*0.02, kd=1600*0.02)

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
        # setp = np.array([ref_1 - 0 * np.pi / 180, ref_2 + 0 * np.pi / 180, 0])
        setp = np.array([ref_1 - 2 * np.pi / 180,
                         ref_2 + 2 * np.pi / 180,
                         z_ref])
        theta_1 = utils.z_rotate(Q_base, a)
        theta_2 = utils.z_rotate(Q_base, b)
        theta_3 = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion
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

    def cmg_control(self, Q_ref, Q_base, z_ref):
        """
        simple CMG control w/ derivative on measurement pid
        """
        v_des = self.v_des
        q = self.q
        dq = self.dq

        theta, setp = self.orient(Q_ref, Q_base, z_ref)  # get body angle and setpoint in rw/cmg frame

        dqf = np.array([dq[1], dq[2], dq[5], dq[6]])  # speed of flywheels
        u_fl = self.pid_fl.pid_control(inp=dqf.flatten(), setp=np.array([v_des, -v_des, v_des, -v_des]))

        qg = np.array([q[0], q[4]]).flatten()  # position of gimbals (one per scissored pair)
        # u_g_pos = self.pid_g_pos.pid_control(inp=qg, setp=np.zeros(2))
        setp_cascaded = np.zeros(len(setp))
        setp_cascaded[0:2] = setp[0:2]  # - u_g_pos
        u_g = self.pid_g.pid_control(inp=theta[0:2], setp=setp_cascaded[0:2])  # gimbal torques

        u_rwz_vel = self.pid_rwz_vel.pid_control(inp=dq[3], setp=setp[2])
        setp_cascaded[2] = setp[2] - u_rwz_vel
        u_rwz = self.pid_rwz_tau.pid_control(inp=theta[2], setp=setp_cascaded[2])  # Cascaded PID Loop

        # u_cmg = np.array([u_g[1], u_fl[0], 0, u_fl[1], u_rwz, u_g[0], u_fl[2], 0, u_fl[3]])
        u_cmg = np.array([u_g[0], u_fl[0], u_fl[1], -u_rwz, u_g[1], u_fl[2], u_fl[3]])

        return u_cmg, theta, setp_cascaded

