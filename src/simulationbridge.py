"""
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np
import pybullet as p
import pybullet_data
import os
import actuator
import actuator_param

useRealTime = 0  # Do NOT change to real time


def reaction(numJoints, bot):  # returns joint reaction force
    reaction = np.array([j[2] for j in p.getJointStates(bot, range(numJoints))])  # 4x6 array [Fx, Fy, Fz, Mx, My, Mz]
    forces = reaction[:, 0:3]  # selected all joints [Fx, Fy, Fz]
    torques = reaction[:, 5]
    return forces, torques  # f = np.linalg.norm(reaction[:, 0:3], axis=1)  # magnitude of F


class Sim:

    def __init__(self, X_0, model, dt=1e-3, fixed=False, spring=False,
                 record=False, scale=1, gravoff=False, direct=False):
        self.dt = dt
        self.omega_xyz = None
        self.omega = None
        self.v = None
        self.record_rt = record  # record video in real time
        self.base_pos = None
        self.spring = spring
        self.L = model["linklengths"]
        self.dir_s = model["springpolarity"]
        self.model = model["model"]
        self.n_a = model["n_a"]

        # --- spring params --- #
        self.init_q = [-30 * np.pi / 180, -150 * np.pi / 180]
        self.ks = 996  # spring constant, N/m
        L0 = self.L[0]  # .15
        L2 = self.L[2]  # .3
        self.rmin = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(2.5*np.pi/180))  # 0.17
        # --- #

        self.actuator_q0 = actuator.Actuator(dt=dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_rmdx10)

        if self.model == 'design_rw':
            S = np.zeros((7, 5))
            S[0, 0] = 1
            S[2, 1] = 1
            S[4, 2] = 1
            S[5, 3] = 1
            S[6, 4] = 1
            self.S = S
            self.actuator_rw1 = actuator.Actuator(dt=dt, model=actuator_param.actuator_r100kv90)
            self.actuator_rw2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_r100kv90)
            self.actuator_rwz = actuator.Actuator(dt=dt, model=actuator_param.actuator_8318)  # mn1005kv90

        elif self.model == 'design_cmg':
            S = np.zeros((13, 9))
            S[0, 0] = 1
            S[2, 1] = 1
            S[4, 2] = 1
            S[5, 3] = 1
            S[7, 4] = 1
            S[8, 5] = 1
            S[9, 6] = 1
            S[10, 7] = 1
            S[12, 8] = 1
            self.S = S
            self.actuator_g01 = actuator.Actuator(dt=dt, model=actuator_param.actuator_ea110)  # mn1005kv90
            self.actuator_g23 = actuator.Actuator(dt=dt, model=actuator_param.actuator_ea110)
            self.actuator_rw0 = actuator.Actuator(dt=dt, model=actuator_param.actuator_mn3110kv700)
            self.actuator_rw1 = actuator.Actuator(dt=dt, model=actuator_param.actuator_mn3110kv700)
            self.actuator_rw2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_mn3110kv700)
            self.actuator_rw3 = actuator.Actuator(dt=dt, model=actuator_param.actuator_mn3110kv700)
            self.actuator_rwz = actuator.Actuator(dt=dt, model=actuator_param.actuator_8318)  # 8318

        if gravoff == True:
            GRAVITY = 0
        else:
            GRAVITY = -9.807

        if direct is True:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)
        model_path = model["urdfpath"]
        # self.bot = p.loadURDF(os.path.join(path_parent, os.path.pardir, model_path), [0, 0, 0.7 * scale],
        self.bot = p.loadURDF(os.path.join(path_parent, model_path), X_0[0:3],  # 0.31
                         robotStartOrientation, useFixedBase=fixed, globalScaling=scale,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])
        self.jointArray = range(p.getNumJoints(self.bot))
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.dt)
        self.numJoints = p.getNumJoints(self.bot)

        p.setRealTimeSimulation(useRealTime)

        self.c_link = 1

        if self.model == 'design_cmg':
            # gimbal scissor constraints
            g1 = p.createConstraint(self.bot, 4, self.bot, 6, p.JOINT_GEAR, [0, 0, 1], [0, 0, 0], [0, 0, 0])
            g2 = p.createConstraint(self.bot, 9, self.bot, 11, p.JOINT_GEAR, [0, 0, 1], [0, 0, 0], [0, 0, 0])
            p.changeConstraint(g1, gearRatio=1, maxForce=10000, erp=0.2)
            p.changeConstraint(g2, gearRatio=1, maxForce=10000, erp=0.2)

        # if self.model != 'design_rw' and self.model != 'design_cmg':
        #     vert = p.createConstraint(self.bot, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 0])

        if self.model == 'design_rw' or self.model == 'design_cmg':
            # p.createConstraint(self.bot, 3, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [-0.135, 0, 0], [0, 0, 0])
            jconn_1 = [x * scale for x in [0.135, 0, 0]]
            jconn_2 = [x * scale for x in [-0.0014381, 0, 0.01485326948]]
            linkjoint = p.createConstraint(self.bot, 1, self.bot, 3, p.JOINT_POINT2POINT, [0, 0, 0], jconn_1, jconn_2)
            p.changeConstraint(linkjoint, maxForce=1000)
            self.c_link = 3

        # increase friction of toe to ideal
        # p.changeDynamics(self.bot, self.c_link, lateralFriction=2, contactStiffness=100000, contactDamping=10000)
        p.changeDynamics(self.bot, self.c_link, lateralFriction=3)  # , restitution=0.01)

        # Record Video in real time
        if self.record_rt is True:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        for i in range(self.numJoints):
            # Disable the default velocity/position motor:
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.setJointMotorControl2(self.bot, i, p.VELOCITY_CONTROL, force=0)  # force=0.5
            # enable joint torque sensing
            p.enableJointForceTorqueSensor(self.bot, i, 1)
            # increase max joint velocity (default = 100 rad/s)
            p.changeDynamics(self.bot, i, maxJointVelocity=800)  # max 3800 rpm

        self.p = np.zeros(3)

    def sim_run(self, u):
        q_ = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
        dq_ = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))

        q = q_.flatten()
        dq = dq_.flatten()
        qa = (q.T @ self.S).flatten()
        dqa = (dq_.T @ self.S).flatten()

        tau_s = self.spring_fn(q) if self.spring else np.zeros(2)

        self.p = np.array(p.getBasePositionAndOrientation(self.bot)[0])
        Q_base_p = np.array(p.getBasePositionAndOrientation(self.bot)[1])
        # pybullet gives quaternions in xyzw format instead of wxyz, so you need to shift values
        Q_base = np.roll(Q_base_p, 1)  # move last element to first place

        tau = np.zeros(self.n_a)
        i = np.zeros(self.n_a)
        v = np.zeros(self.n_a)

        if self.model == "design_rw":
            u *= -1
            tau[0], i[0], v[0] = self.actuator_q0.actuate(i=u[0], q_dot=dqa[0]) + tau_s[0]
            tau[1], i[1], v[1] = self.actuator_q2.actuate(i=u[1], q_dot=dqa[1]) + tau_s[1]
            tau[2], i[2], v[2] = self.actuator_rw1.actuate(i=u[2], q_dot=dqa[2])
            tau[3], i[3], v[3] = self.actuator_rw2.actuate(i=u[3], q_dot=dqa[3])
            tau[4], i[4], v[4] = self.actuator_rwz.actuate(i=u[4], q_dot=dqa[4])

        elif self.model == "design_cmg":
            u *= -1
            tau[0], i[0], v[0] = self.actuator_q0.actuate(i=u[0], q_dot=dqa[0]) + tau_s[0]
            tau[1], i[1], v[1] = self.actuator_q2.actuate(i=u[1], q_dot=dqa[1]) + tau_s[1]
            tau[2], i[2], v[2] = self.actuator_g01.actuate(i=u[2], q_dot=dqa[2])
            tau[3], i[3], v[3] = self.actuator_rw0.actuate(i=u[3], q_dot=dqa[3])
            tau[4], i[4], v[4] = self.actuator_rw1.actuate(i=u[4], q_dot=dqa[4])
            tau[5], i[5], v[5] = self.actuator_rwz.actuate(i=u[5], q_dot=dqa[5])
            tau[6], i[6], v[6] = self.actuator_g23.actuate(i=u[6], q_dot=dqa[6])
            tau[7], i[7], v[7] = self.actuator_rw2.actuate(i=u[7], q_dot=dqa[7])
            tau[8], i[8], v[8] = self.actuator_rw3.actuate(i=u[8], q_dot=dqa[8])

        torque = self.S @ tau

        p.setJointMotorControlArray(self.bot, self.jointArray, p.TORQUE_CONTROL, forces=torque)
        velocities = p.getBaseVelocity(self.bot)
        self.v = velocities[0]  # base linear velocity in global Cartesian coordinates
        self.omega_xyz = velocities[1]  # base angular velocity in XYZ
        self.base_pos = p.getBasePositionAndOrientation(self.bot)
        f_sens, tau_sens = reaction(self.numJoints, self.bot)
        grf_nrml_onB = [c[7] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]
        grf_nrml = [c[9] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]
        fric1 = [c[10] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]
        fric1_dir = [c[11] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]
        fric2 = [c[12] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]
        fric2_dir = [c[13] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]
        grf_z = grf_nrml * np.array(grf_nrml_onB)
        fric_y = fric1 * np.array(fric1_dir)
        fric_x = fric2 * np.array(fric2_dir)
        grf = (grf_z + fric_y + fric_x).T
        grf = np.zeros((3, 1)) if not grf.any() else grf  # prevent empty list from being passed
        # Detect contact with ground plane
        c = bool(len([c[8] for c in p.getContactPoints(self.bot, self.plane, self.c_link)]))

        if useRealTime == 0:
            p.stepSimulation()

        return qa, dqa, Q_base, c, tau, f_sens, tau_sens, i, v, grf

    def spring_fn(self, q):
        """
        linear extension spring b/t joints 1 and 3 of parallel mechanism
        approximated by applying torques to joints 0 and 2
        """
        init_q = self.init_q
        k = self.ks
        L0 = self.L[0]
        L2 = self.L[2]
        rmin = self.rmin

        if q is None:
            q0 = init_q[0]
            q2 = init_q[2]
        else:
            q0 = q[0] + init_q[0]
            q2 = q[2] + init_q[1]

        gamma = abs(q2 - q0)
        r = np.sqrt(L0 ** 2 + L2 ** 2 - 2 * L0 * L2 * np.cos(gamma))  # length of spring

        if r < rmin:
            print("error: incorrect spring params, r = ", r, " and rmin = ", rmin, "\n gamma = ", gamma)

        T = k * (r - rmin)  # spring tension force
        alpha = np.arccos((-L0 ** 2 + L2 ** 2 + r ** 2) / (2 * L2 * r))
        beta = np.arccos((-L2 ** 2 + L0 ** 2 + r ** 2) / (2 * L0 * r))
        tau_s0 = -T * np.sin(beta) * L0
        tau_s1 = T * np.sin(alpha) * L2
        tau_s = np.array([tau_s0, tau_s1])
        return tau_s * self.dir_s
