"""
Copyright (C) 2013 Travis DeWolf
Copyright (C) 2020 Benjamin Bokser
"""
import numpy as np
import transforms3d

#import control
import cqp

class Control:

    def __init__(self, dt=1e-3, null_control=False, **kwargs):
        # self.qp = qp.Qp()
        self.qp = cqp.Cqp()
        self.dt = dt
        self.null_control = null_control

        self.kp = np.zeros((3, 3))
        self.kp[0, 0] = 500 # 5000
        self.kp[1, 1] = 500
        self.kp[2, 2] = 500

        self.kv = np.array(self.kp)*0.1

        self.ko = np.zeros((3, 3))
        self.ko[0, 0] = 1000
        self.ko[1, 1] = 1000
        self.ko[2, 2] = 1000

        self.kn = np.zeros((2, 2))
        self.kn[0, 0] = 100
        self.kn[1, 1] = 100

        self.kf = 1

        self.Mq = None
        self.Mx = None
        self.x_dd_des = None
        self.J = None
        self.x = None
        self.grav = None
        self.velocity = None
        self.q_e = None

    def wb_control(self, leg, target, b_orient, force=0, x_dd_des=None):

        self.target = target
        self.b_orient = np.array(b_orient)

        # which dim to control of [x, y, z, alpha, beta, gamma]
        # ctrlr_dof = self.ctrlr_dof

        # calculate the Jacobian
        JEE = leg.gen_jacEE()  # print(np.linalg.matrix_rank(JEE))
        # rank of matrix is 3, can only control 3 DOF with one OSC
        # adjust 2D jacobian to 3D
        JEE = np.array([JEE[0, :], np.zeros(4), JEE[1, :]])  # TODO: Make jacobian 3D by default

        # generate the mass matrix in end-effector space
        # self.Mq = leg.gen_Mq()
        # Mx = leg.gen_Mx(Mq=self.Mq, JEE=JEE)

        x_dd_des = np.zeros(6)  # [x, y, z, alpha, beta, gamma]

        # multiply with rotation matrix for base to world
        self.x = np.dot(b_orient, leg.position())
        # self.x = leg.position()[:, -1]

        # calculate operational space velocity vector
        self.velocity = np.dot(b_orient, (np.transpose(np.dot(JEE, leg.dq)).flatten())[0:3])

        # calculate linear acceleration term based on PD control
        x_dd_des[:3] = np.dot(self.kp, (self.target[0:3] - self.x)) + np.dot(self.kv, -self.velocity)
        # x_dd_des = x_dd_des[ctrlr_dof]  # get rid of dim not being controlled
        x_dd_des = np.reshape(x_dd_des, (-1, 1))

        r_dd_des = np.array([x_dd_des[0], x_dd_des[2]])
        x_ref = np.array([0, 0, 0, 0, 0, 0])
        x_in = np.array([leg.d2q[0], leg.d2q[1], leg.d2q[2], leg.d2q[3], 0., 0.])
        u = self.qp.qpcontrol(leg, r_dd_des, x_in, x_ref)

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
