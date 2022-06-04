"""
Copyright (C) 2020-2022 Benjamin Bokser
"""
import numpy as np
import utils
import pid
import time


def raibert_x(kr, kt, pdot, pdot_ref):
    x_f0 = pdot * kt * np.abs(pdot[2]) / 2   # footstep in world frame for neutral motion, uses vertical leaping speed
    x_f = x_f0 + kr * (pdot - pdot_ref)  # footstep placement in world frame with desired acceleration
    return x_f  # 3x1


class Gait:
    def __init__(self, model, moment, controller, leg, target, hconst, t_st, X_f, gain=5000, dt=1e-3, **kwargs):
        self.swing_steps = 0
        self.trajectory = None
        self.dt = dt
        self.controller = controller
        self.moment = moment
        self.model = model
        self.leg = leg
        self.t_st = t_st  # time spent in stance
        self.hconst = hconst
        self.target = target
        self.n_a = model["n_a"]
        self.k_wbc = gain  # wbc gain
        self.k_k = model["k_k"][0]
        self.kd_k = model["k_k"][1]
        self.a_kt = model["a_kt"]
        self.x_des = np.array([0, 0, 0])
        # self.pid_pdot = pid.PID1(kp=0.025, ki=0.05, kd=0.02)  # kp=0.6, ki=0.15, kd=0.02
        kp = [0.02, 0.08,  0]
        ki = [0.2,  0.2,   0]
        kd = [0.01, 0.02,  0]
        self.pid_pdot = pid.PIDn(kp=kp, ki=ki, kd=kd)
        self.z_ref = 0  # keep z_ref persistent
        self.X_f = X_f
        self.u = np.zeros(self.n_a)

    def u_mpc(self, sh, X_in, U_in, pf_refk):
        # mpc-based hopping
        Q_base = X_in[3:7]
        # z = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion
        # Q_z = np.array([np.cos(z / 2), 0, 0, np.sin(z / 2)]).T  # Q_base converted to just the z-axis rotation
        # rz_psi = utils.rz(utils.quat2euler(Q_base)[2])
        if sh == 0:
            pfw_ref = pf_refk - X_in[0:3]  # vec from CoM to footstep ref in world frame
            pfb_ref = utils.Z(utils.Q_inv(Q_base), pfw_ref)  # world frame -> body frame
            # pfb_ref = pfb_ref/np.linalg.norm(pfb_ref) * self.hconst * 4.5 / 3
            self.u[0:2] = self.controller.wb_pos_control(target=pfb_ref)
            # self.u[0:2] = self.controller.invkin_pos_control(target=pfb_ref, kp=self.k_k, kd=self.kd_k)
        elif sh == 1:
            # self.u[0:2] = self.controller.wb_f_control(force=utils.Z(Q_base, -U_in[0:3]))  # world frame to body frame
            self.u[0:2] = self.controller.wb_f_control(force=-U_in[0:3])  # no rotation necessary, already in b frame
        else:
            raise NameError('INVALID STATE')
        self.u[2:] = self.moment.rw_torque_ctrl(U_in[3:6])
        return self.u / self.a_kt  # convert from Nm to A

    def u_raibert(self, state, state_prev, X_in, x_ref):
        # continuous raibert hopping
        p = X_in[0:3]
        Q_base = X_in[3:7]
        pdot = X_in[7:10]
        p_ref = x_ref[100, 0:3]  # raibert hopping only looks at position ref
        z = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion
        # z = utils.quat2euler(Q_base)[2]
        Q_z = np.array([np.cos(z / 2), 0, 0, np.sin(z / 2)]).T
        Q_z_inv = utils.Q_inv(Q_z)
        pdot_ref = -self.pid_pdot.pid_control(inp=utils.Z(Q_z_inv, p), setp=utils.Z(Q_z_inv, p_ref))  # adjust for yaw
        pdot_ref = utils.Z(Q_z, pdot_ref)  # world frame -> body frame
        if np.linalg.norm(self.X_f[0:2] - X_in[0:2]) >= 1:
            v_ref = p_ref - p
            self.z_ref = np.arctan2(v_ref[1], v_ref[0])  # desired yaw
        k_b = (np.clip(np.linalg.norm(self.X_f[0:2] - X_in[0:2]), 0.5, 1) + 2)/3  # "Braking" gain based on dist
        hconst = self.hconst * k_b
        kr = .15 / k_b  # .15 / k_b "speed cancellation" constant
        kt = 0.4  # gain representing leap period accounting for vertical jump velocity at toe-off
        if state == 'Return':
            if state_prev == 'Leap':  # find new footstep position based on desired speed and current speed
                self.x_des = raibert_x(kr, kt, pdot, pdot_ref) + p  # world frame desired footstep position
                self.x_des[2] = 0  # enforce footstep location is on ground plane
            if pdot[2] >= 0:  # recognize that robot is still rising
                self.target[2] = -hconst  # pull leg up to prevent stubbing
            else:
                self.target[2] = -hconst * 5.5 / 3  # brace for impact
        elif state == 'HeelStrike':
            self.target[2] = -hconst * 4.5 / 3
        elif state == 'Leap':
            self.target[2] = -hconst * 5.5 / 3
        else:
            raise NameError('INVALID STATE')

        Q_ref = utils.Q_inv(utils.vec_to_quat(self.x_des - p))
        self.u[0:2] = self.controller.wb_pos_control(target=self.target)
        self.u[2:], thetar, setp = self.moment.rw_control(Q_ref, Q_base, self.z_ref)
        return self.u, thetar, setp

    def u_wbc_vert(self, state, state_prev, X_in, x_ref):
        Q_base = X_in[3:7]
        Q_ref = np.array([1, 0, 0, 0])
        hconst = self.hconst
        self.target[0] = 0  # -0.02
        if state == 'Return':
            self.target[2] = -hconst * 5 / 3
        elif state == 'HeelStrike':
            self.target[2] = -hconst
        elif state == 'Leap':
            self.target[2] = -hconst * 6.5 / 3
            # force = fr if not None else None
        else:
            raise NameError('INVALID STATE')
        self.u[0:2] = self.controller.wb_pos_control(target=self.target)
        self.u[2:], thetar, setp = self.moment.rw_control(Q_ref, Q_base, z_ref=0)
        return self.u, thetar, setp

    def u_wbc_static(self, state, state_prev, X_in, x_ref):
        Q_base = X_in[3:7]
        Q_ref = np.array([1, 0, 0, 0])
        self.target[0] = 0
        self.target[2] = -self.hconst * 5.5 / 3
        self.u[0:2] = self.controller.wb_pos_control(target=self.target)
        # self.u[0:2] = self.controller.qp_pos_control(target=self.target)
        self.u[2:], thetar, setp = self.moment.rw_control(Q_ref, Q_base, z_ref=0)
        return self.u, thetar, setp

    def u_ik_vert(self, state, state_prev, X_in, x_ref):
        time.sleep(self.dt)  # closed form inv kin runs much faster than full wbc, slow it down
        Q_base = X_in[3:7]
        Q_ref = np.array([1, 0, 0, 0])
        hconst = self.hconst
        if state == 'Return':
            self.target[2] = -hconst * 5 / 3
            self.u[0:2] = self.controller.invkin_pos_control(self.target, self.k_k/45, self.kd_k/45)
        elif state == 'HeelStrike':
            self.target[2] = -hconst
            self.u[0:2] = self.controller.invkin_pos_control(self.target, self.k_k, self.kd_k)
        elif state == 'Leap':
            self.target[2] = -hconst * 5.5 / 3
            self.u[0:2] = self.controller.invkin_pos_control(self.target, self.k_k, self.kd_k)

        self.u[2:], thetar, setp = self.moment.rw_control(Q_ref, Q_base, z_ref=0)
        return self.u, thetar, setp

    def u_ik_static(self, state, state_prev, X_in, x_ref):
        Q_base = X_in[3:7]
        target = self.target
        target[2] = -self.hconst * 5 / 3
        Q_ref = np.array([1, 0, 0, 0])
        self.u[0:2] = self.controller.invkin_pos_control(self.target, self.k_k, self.kd_k)
        self.u[2:], thetar, setp = self.moment.rw_control(Q_ref, Q_base, z_ref=0)
        return self.u, thetar, setp
