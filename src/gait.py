"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import numpy as np
import rw
import transforms3d
import utils
import pid


def raibert_x(kr, kt, pdot, pdot_ref):
    # footstep placement wrt base CoM in world frame for neutral motion
    s_z = np.abs(pdot[2])  # vertical leaping speed
    x_f0 = pdot[0:2] * kt * s_z / 2   # takes leaping speed into account
    # footstep placement in world frame with desired acceleration
    x_f = x_f0 + kr * (pdot[0:2] - pdot_ref[0:2])
    return x_f  # 2x1


class Gait:
    def __init__(self, controller, leg, target, hconst, use_qp=False, dt=1e-3, **kwargs):

        self.swing_steps = 0
        self.trajectory = None
        self.dt = dt
        self.init_alpha = 0
        self.init_beta = 0
        self.init_gamma = 0
        self.init_angle = np.array([self.init_alpha, self.init_beta, self.init_gamma])
        self.controller = controller
        self.leg = leg
        self.x_last = None
        self.hconst = hconst
        self.target = target  # np.hstack(np.append(np.array([0, 0, -self.hconst]), self.init_angle))
        # self.r_save = np.array([0, 0, -self.hconst])

        if use_qp is True:
            self.controlf = self.controller.wb_qp_control
        else:
            self.controlf = self.controller.wb_control

        self.x_des = np.array([0, 0, 0])
        self.p_err_sum = 0
        self.p_err_prev = 0

        '''
        # Gains for ideal torque source
        ku = 180
        kp = np.zeros((3, 3))
        kd = np.zeros((3, 3))
        ki = np.zeros((3, 3))
        np.fill_diagonal(kp, [ku, -ku, 8])
        np.fill_diagonal(ki, [ku * 0, -ku * 0, 0])
        np.fill_diagonal(kd, [ku * 0.06, -ku * 0.06, 8 * 2.5])
        '''
        # torque PID gains
        ku = 160
        kp = np.zeros((3, 3))
        kd = np.zeros((3, 3))
        ki = np.zeros((3, 3))
        np.fill_diagonal(kp, [ku, -ku, ku / 2])
        np.fill_diagonal(ki, [ku * 0, -ku * 0, ku / 2 * 0])
        np.fill_diagonal(kd, [ku * 0.02, -ku * 0.02, ku / 2 * 0.02])

        # speed PID gains
        ku_s = 0
        kp_s = np.zeros((3, 3))
        kd_s = np.zeros((3, 3))
        ki_s = np.zeros((3, 3))
        np.fill_diagonal(kp_s, [ku_s * 0, -ku_s * 0, ku_s])
        np.fill_diagonal(ki_s, [ku_s * 0, -ku_s * 0, ku_s / 2 * 0])
        np.fill_diagonal(kd_s, [ku_s * 0, -ku_s * 0, ku_s / 2 * 0.06])

        self.pid_torque = pid.PID3(kp=kp, kd=kd, ki=ki)
        self.pid_vel = pid.PID3(kp=kp_s, kd=kd_s, ki=ki_s)

    def u_raibert(self, state, state_prev, p, p_ref, pdot, Q_base, fr):
        # raibert hopping
        dt = self.dt
        force = np.zeros((3, 1))
        kp = 0.2
        ki = 0.01
        kd = 0
        kr = 0.175
        kt = 0.6  # gain representing leap period accounting for vertical jump velocity at toe-off
        p_err = (p - p_ref)
        self.p_err_sum += p_err
        p_err_diff = (p_err - self.p_err_prev) / dt
        pdot_ref = -(kp * p_err + ki * self.p_err_sum * dt + kd * p_err_diff)  # PID control for body position
        # pdot_ref = np.array([0, 0.2, 0])
        hconst = self.hconst
        self.target[0] = -0.05  # adjustment for balance due to bad mockup design
        if state == 'Return':
            if state_prev == 'Leap':  # find new footstep position based on desired speed and current speed
                x_fb = np.zeros(3)
                x_fb[0:2] = raibert_x(kr, kt, pdot, pdot_ref)  # desired footstep relative to current body CoM
                self.x_des = x_fb + p  # world frame desired footstep position
                self.x_des[2] = 0  # enforce footstep location is on ground plane

            if pdot[2] >= 0:  # recognize that robot is still rising
                self.target[2] = -hconst  # pull leg up to prevent stubbing
            else:
                self.target[2] = -hconst * 5.5 / 3  # brace for impact
            # self.controller.update_gains(150, 150 * 0.2)

        elif state == 'HeelStrike':
            # self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst * 5.15 / 3

        elif state == 'Leap':
            # self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst * 5.5 / 3
            if fr is not None:
                force = fr

        else:
            raise NameError('INVALID STATE')
        # Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        Q_ref = utils.vec_to_quat2(self.x_des - p)
        Q_ref = utils.Q_inv(Q_ref)  # TODO: Shouldn't be necessary, caused by some other mistake
        u = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u_rw, thetar, setp = rw.rw_control(self.dt, self.pid_torque, self.pid_vel, Q_ref, Q_base, qrw_dot)
        self.p_err_prev = p_err
        return u, u_rw, thetar, setp

    def u_wbc_vert(self, state, Q_base, fr):
        force = np.zeros((3, 1))
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        self.target[0] = -0.05
        if state == 'Return':
            # self.controller.update_gains(150, 150 * 0.3)
            self.target[2] = -hconst * 5 / 3
        elif state == 'HeelStrike':
            # self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst
        elif state == 'Leap':
            # self.controller.update_gains(5000, 5000 * 0.02)
            self.target[2] = -hconst * 5.5 / 3
            if fr is not None:
                force = fr
        else:
            raise NameError('INVALID STATE')

        u = -self.controlf(target=self.target, Q_base=Q_base, force=force)
        u_rw, thetar, setp = rw.rw_control(self.dt, self.pid_torque, self.pid_vel, Q_ref, Q_base, qrw_dot)
        return u, u_rw, thetar, setp

    def u_wbc_static(self, Q_base, qrw_dot, fr):
        force = np.zeros((3, 1))
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        self.target[0] = -0.05
        self.target[2] = -self.hconst * 5.5 / 3
        # self.controller.update_gains(5000, 5000 * 0.02)
        if fr is not None:
            force = fr
        u = -self.controlf(target=self.target, Q_base=np.array([1, 0, 0, 0]), force=force)
        u_rw, thetar, setp = rw.rw_control(self.dt, self.pid_torque, self.pid_vel, Q_ref, Q_base, qrw_dot)
        return u, u_rw, thetar, setp

    def u_invkin_vert(self, state, Q_base, k_g, k_gd, k_a, k_ad):
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)  # 2.5 * np.pi / 180
        hconst = self.hconst
        k = k_g
        kd = k_gd
        if state == 'Return':
            self.target = np.array([0, 0, -hconst * 5 / 3])
            k = k_a
            kd = k_ad
        elif state == 'HeelStrike':
            self.target[2] = -hconst
            k = k_g
            kd = k_gd
        elif state == 'Leap':
            self.target = np.array([0, 0, -hconst * 5.5 / 3])
            k = k_g
            kd = k_gd
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        u = (qa - self.leg.inv_kinematics(xyz=self.target[0:3])) * k + dqa * kd
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)
        return u, u_rw, thetar, setp

    def u_invkin_static(self, Q_base, k, kd):
        target = self.target
        target[2] = -self.hconst * 5 / 3
        Q_ref = transforms3d.euler.euler2quat(0, 0, 0)
        dqa = np.array([self.leg.dq[0], self.leg.dq[2]])
        qa = np.array([self.leg.q[0], self.leg.q[2]])
        # u = (self.leg.q - self.leg.inv_kinematics(xyz=self.target[0:3])) * k_kin + self.leg.dq * k_d
        u = (qa - self.leg.inv_kinematics(xyz=target[0:3])) * k + dqa * kd
        u_rw, self.err_sum, self.err_prev, thetar, setp = rw.rw_control(self.dt, Q_ref, Q_base,
                                                                        self.err_sum, self.err_prev)
        return u, u_rw, thetar, setp
