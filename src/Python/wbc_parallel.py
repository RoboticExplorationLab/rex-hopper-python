"""
Copyright (C) 2020-2021 Benjamin Bokser
"""
import numpy as np
import transforms3d

import cqp


class Control:

    def __init__(self, dt=1e-3, null_control=False, **kwargs):
        # self.qp = qp.Qp()
        self.qp = cqp.Cqp()
        self.dt = dt
        self.null_control = null_control

        self.kp = np.zeros((3, 3))
        np.fill_diagonal(self.kp, 5)

        self.kv = np.array(self.kp)*0.1

        self.kn = np.zeros((2, 2))
        np.fill_diagonal(self.kn, 100)

        self.kf = 1

        self.Mq = None
        self.Mx = None
        self.x_dd_des = None
        self.J = None
        self.x = None
        self.grav = None
        self.velocity = None
        self.q_e = None
        self.simple = False

    def wb_control(self, leg, target, b_orient, force=0, x_dd_des=None):
        target = np.array(target).reshape(-1, 1)
        b_orient = np.array(b_orient)
        Ja = leg.gen_jacA()  # 3x2
        dqa = np.array([leg.dq[0], leg.dq[2]])
        self.x = np.dot(b_orient, leg.position())

        # calculate operational space velocity vector
        self.velocity = np.dot(b_orient, (np.transpose(np.dot(Ja, dqa)))[0:3]).reshape(-1, 1)

        # calculate linear acceleration term based on PD control
        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]
        x_dd_des[:3] = (np.dot(self.kp, (target[0:3] - self.x)) + np.dot(self.kv, -self.velocity)).flatten()
        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        if self.simple == True:
            tau = Ja.T @ x_dd_des[0:3]
            u = tau

        else:
            # r_dd_des = np.array(x_dd_des[0:3])
            r_dd_des = np.array([x_dd_des[0], x_dd_des[2]]).flatten()
            x_ref = np.array([0, 0, 0, 0, 0, 0])
            x_in = np.array([leg.d2q[0], leg.d2q[1], leg.d2q[2], leg.d2q[3], 0., 0.])
            u = self.qp.qpcontrol(leg, r_dd_des, x_in, x_ref)
            u = np.array([u[1], u[0]])


        '''
        # calculate force
        Fx = np.dot(Mx, x_dd_des)
        Aq_dd = (np.dot(JEE.T, Fx).reshape(-1, ))

        Fr = np.dot(b_orient, force)
        force_control = (np.dot(JEE.T, Fr).reshape(-1, ))

        self.grav = leg.gen_grav(b_orient=b_orient)
        self.u = Aq_dd - self.grav - force_control*self.kf
        self.x_dd_des = x_dd_des
        self.Mx = Mx
        self.J = JEE

        # if null_control is selected, add a control signal in the
        # null space to try to move the leg to selected position
        if self.null_control:
            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            leg_des_angle = np.array([-30, -150])
            prop_val = ((leg_des_angle - leg.q) + np.pi) % (np.pi * 2) - np.pi
            q_des = (np.dot(self.kn, prop_val))
            #        + np.dot(self.knd, -leg.dq.reshape(-1, )))

            Fq_null = np.dot(self.Mq, q_des)

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(self.Mq)))
            null_filter = np.eye(len(leg.q)) - np.dot(JEE.T, Jdyn_inv)
            null_signal = np.dot(null_filter, Fq_null).reshape(-1, )

            self.u += null_signal
        '''
        return u
