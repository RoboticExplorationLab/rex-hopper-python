"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import numpy as np
import rw
import transforms3d
import utils
import pid


def raibert_x(kr, kt, pdot, pdot_ref):
    # footstep placement wrt base CoM in world frame for neutral motion
    s_z = np.abs(pdot[2])  # vertical leaping speed
    x_f0 = pdot[0:2] * kt * s_z / 2   # takes vertical leaping speed into account
    # footstep placement in world frame with desired acceleration
    x_f = x_f0 + kr * (pdot[0:2] - pdot_ref[0:2])
    return x_f  # 2x1


class Gait:
    def __init__(self, model, controller, leg, target, hconst, use_qp=False, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.dt = dt
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.init_angle = np.array([self.init_alpha, self.init_beta, self.init_gamma])
        self.controller = controller
        self.model = model
        self.leg = leg
        self.x_last = None
        self.hconst = hconst
        self.target = target  # np.hstack(np.append(np.array([0, 0, -self.hconst]), self.init_angle))

        if use_qp is True:
            self.controlf = self.controller.wb_qp_control
        else:
            self.controlf = self.controller.wb_control

        self.x_des = np.array([0, 0, 0])

        self.moment_ctrl = None

        if self.model["model"] == "design_rw":
            self.moment_ctrl = rw.rw_control
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
            self.moment_ctrl = rw.cmg_control
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

        self.pid_pdot = pid.PID1(kp=0.025, ki=0.05, kd=0.02)  # kp=0.6, ki=0.15, kd=0.02

    def u_raibert(self, state, state_prev, p, p_ref, pdot, Q_base, qrw_dot, fr):
        # continuous raibert hopping
        force = np.zeros((3, 1))
        pdot_ref = -self.pid_pdot.pid_control(inp=p, setp=p_ref)
        # pdot_ref = np.array([0, 0.2, 0])
        hconst = self.hconst
        self.target[0] = 0 # -0.08  # adjustment for balance due to bad mockup design
        if state == 'Return':
            if state_prev == 'Leap':  # find new footstep position based on desired speed and current speed
                kr = 0.3 / (np.linalg.norm(pdot_ref) + 2)  # 0.4 "speed cancellation" constant
                kt = 0.4  # 0.4 gain representing leap period accounting for vertical jump velocity at toe-off
                x_fb = np.zeros(3)
                x_fb[0:2] = raibert_x(kr, kt, pdot, pdot_ref)  # desired footstep relative to current body CoM
                self.x_des = x_fb + p  # world frame desired footstep position
                self.x_des[2] = 0  # enforce footstep location is on ground plane
            if pdot[2] >= 0:  # recognize that robot is still rising
                self.target[2] = -hconst  # pull leg up to prevent stubbing
            else:
                self.target[2] = -hconst * 6 / 3  # brace for impact
            self.controller.update_gains(150, 150 * 0.2)

        elif state == 'HeelStrike':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst * 4.5 / 3

        elif state == 'Leap':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst * 6 / 3
            if fr is not None:
                force = fr

        else:
            raise NameError('INVALID STATE')
        # Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        Q_ref = utils.vec_to_quat2(self.x_des - p)
        Q_ref = utils.Q_inv(Q_ref)  # TODO: Shouldn't be necessary, caused by some other mistake
        u = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u_rw, thetar, setp = self.moment_ctrl(self.pid_tau, self.pid_vel, Q_ref, Q_base, qrw_dot)
        # self.p_err_prev = p_err
        return u, u_rw, thetar, setp

    def u_wbc_vert(self, state, Q_base, qrw_dot, fr):
        force = np.zeros((3, 1))
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        self.target[0] = 0
        if state == 'Return':
            self.controller.update_gains(150, 150 * 0.2)
            self.target[2] = -hconst * 5 / 3
        elif state == 'HeelStrike':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst
        elif state == 'Leap':
            self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst * 5.5 / 3
            if fr is not None:
                force = fr
        else:
            raise NameError('INVALID STATE')

        u = -self.controlf(target=self.target, Q_base=Q_base, force=force)
        u_rw, thetar, setp = self.moment_ctrl(self.pid_tau, self.pid_vel, Q_ref, Q_base, qrw_dot)
        return u, u_rw, thetar, setp

    def u_wbc_static(self, Q_base, qrw_dot, fr):
        force = np.zeros((3, 1))
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        self.target[0] = 0
        self.target[2] = -self.hconst * 5.5 / 3
        self.controller.update_gains(5000, 5000 * 0.02)
        if fr is not None:
            force = fr
        u = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u_rw, thetar, setp = self.moment_ctrl(self.pid_tau_s, self.pid_vel, Q_ref, Q_base, qrw_dot)
        return u, u_rw, thetar, setp

    def u_invkin_vert(self, state, Q_base, qrw_dot, k_g, k_gd, k_a, k_ad):
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        k = k_g
        kd = k_gd
        if state == 'Return':
            self.target[2] = -hconst * 5 / 3
            k = k_a
            kd = k_ad
        elif state == 'HeelStrike':
            self.target[2] = -hconst
            k = k_g
            kd = k_gd
        elif state == 'Leap':
            self.target[2] = -hconst * 5.5 / 3
            k = k_g
            kd = k_gd
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        u = (qa - self.leg.inv_kinematics(xyz=self.target[0:3])) * k + dqa * kd
        u_rw, thetar, setp = self.moment_ctrl(self.pid_tau, self.pid_vel, Q_ref, Q_base, qrw_dot)
        return u, u_rw, thetar, setp

    def u_invkin_static(self, Q_base, qrw_dot, k, kd):
        target = self.target
        target[2] = -self.hconst * 5 / 3
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        # u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * k_kin + self.leg.dq * k_d
        u = (qa - self.leg.inv_kinematics(xyz=target[0:3])) * k + dqa * kd
        u_rw, thetar, setp = self.moment_ctrl(self.pid_tau, self.pid_vel, Q_ref, Q_base, qrw_dot)
        return u, u_rw, thetar, setp
