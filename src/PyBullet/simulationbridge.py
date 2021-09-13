"""
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np
import transforms3d
import pybullet as p
import pybullet_data
import os

import actuator

useRealTime = 0

def reaction_torques():
    # returns joint reaction torques
    reaction_force = [j[2] for j in p.getJointStates(bot, range(2))]  # j[2]=jointReactionForces
    #  [Fx, Fy, Fz, Mx, My, Mz]
    reaction_force = np.array(reaction_force)
    torques = reaction_force  # selected all joints My
    # torques[0] = reaction_force[0, 5]  # selected joint 1 Mz
    # torques[4] = reaction_force[4, 5]  # selected joint 5 Mz
    return torques


class Sim:

    def __init__(self, dt=1e-3, model='serial'):
        self.dt = dt
        self.omega_xyz = None
        self.omega = None
        self.v = None
        self.record_rt = False  # record video in real time
        self.base_pos = None

        GRAVITY = -9.807
        # physicsClient = p.connect(p.GUI)
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        curdir = os.getcwd()
        path_parent = os.path.dirname(curdir)

        if model == 'serial' or model == 'belt':
            model_path = "res/flyhopper_mockup/urdf/flyhopper_mockup.urdf"
        elif model == 'parallel':
            model_path = "res/flyhopper_parallel/urdf/flyhopper_parallel.urdf"
        else:
            print("error: model choice invalid")
            model_path = None

        self.bot = p.loadURDF(os.path.join(path_parent, model_path), [0, 0, 0.7],  # 0.31
                         robotStartOrientation, useFixedBase=0,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)

        vert = p.createConstraint(self.bot, -1, -1, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 0])
        self.jointArray = range(p.getNumJoints(self.bot))
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.dt)
        self.numJoints = p.getNumJoints(self.bot)
        p.setRealTimeSimulation(useRealTime)
        # p.changeDynamics(self.bot, 2, lateralFriction=0.5)
        # '''
        if model == 'parallel':
            linkjoint = p.createConstraint(self.bot, 1, self.bot, 3,
                                     p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0], [.15, 0, 0])
            p.changeConstraint(linkjoint, maxForce=10000)
        elif model == 'belt':
            # vert = p.createConstraint(self.bot, -1, -1, 1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0.3, 0, 0])
            belt = p.createConstraint(self.bot, 0, self.bot, 1,
                                     p.JOINT_GEAR, [0, 1, 0], [0, 0, 0], [0, 0, 0])
            p.changeConstraint(belt, gearRatio=0.5, gearAuxLink=-1, maxForce=10000)
        # '''
        # print(p.getJointInfo(self.bot, 3))
        # Record Video in real time
        if self.record_rt is True:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")

        # Disable the default velocity/position motor:
        for i in range(self.numJoints):
            p.setJointMotorControl2(self.bot, i, p.VELOCITY_CONTROL, force=0)  # force=0.5
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.enableJointForceTorqueSensor(self.bot, i, 1)  # enable joint torque sensing

        self.model = model

    def sim_run(self, u):

        base_or_p = np.array(p.getBasePositionAndOrientation(self.bot)[1])
        # pybullet gives quaternions in xyzw format
        # transforms3d takes quaternions in wxyz format, so you need to shift values
        b_orient = np.zeros(4)
        b_orient[0] = base_or_p[3]  # w
        b_orient[1] = base_or_p[0]  # x
        b_orient[2] = base_or_p[1]  # y
        b_orient[3] = base_or_p[2]  # z
        b_orient = transforms3d.quaternions.quat2mat(b_orient)

        q_dot = np.zeros(2)
        torque = np.zeros(2)
        q = np.zeros(2)

        if self.model == "serial":
            command = -u
            # Pull values in from simulator, select relevant ones, reshape to 2D array
            q = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
            q_dot = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))

            torque[0] = actuator.actuate(v=command[0], q_dot=q_dot[0], gr_out=7)
            torque[1] = actuator.actuate(v=command[1], q_dot=q_dot[1], gr_out=12)
            # q[1] *= -1  # This seems to be correct 8-25-21

        elif self.model == "parallel":
            command = np.zeros(4)
            command[0] = -u[1]  # readjust to match motor polarity
            command[2] = -u[0]  # readjust to match motor polarity

            q_all = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
            q[0] = q_all[2]
            q[1] = q_all[0]  # This seems to be correct 9-06-21
            q_dot = np.zeros(4)
            q_dot_all = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
            q_dot[0] = q_dot_all[2]
            q_dot[1] = q_dot_all[0]  # This seems to be correct 9-06-21
            torque = np.zeros(4)
            torque[0] = actuator.actuate(v=command[0], q_dot=q_dot[0], gr_out=12)
            torque[1] = actuator.actuate(v=command[1], q_dot=q_dot[1])
            torque[2] = actuator.actuate(v=command[2], q_dot=q_dot[2], gr_out=12)
            torque[3] = actuator.actuate(v=command[3], q_dot=q_dot[3])

        elif self.model == "belt":
            command = np.zeros(2)
            command[0] = -u[0] # only 1 DoF actuated

            # Pull values in from simulator, select relevant ones, reshape to 2D array
            q = np.reshape([j[0] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
            q = q[0] # only 1 DoF actuated, remove extra.

            q_dot_all = np.reshape([j[1] for j in p.getJointStates(1, range(0, self.numJoints))], (-1, 1))
            q_dot[0] = q_dot_all[0]

            torque[0] = actuator.actuate(v=command[0], q_dot=q_dot[0])

        # print(self.reaction_torques()[0:4])
        p.setJointMotorControlArray(self.bot, self.jointArray, p.TORQUE_CONTROL, forces=torque)
        velocities = p.getBaseVelocity(self.bot)
        self.v = velocities[0]  # base linear velocity in global Cartesian coordinates
        self.omega_xyz = velocities[1]  # base angular velocity in Euler XYZ
        self.base_pos = p.getBasePositionAndOrientation(self.bot)
        # base angular velocity in quaternions
        # self.omega = transforms3d.euler.euler2quat(omega_xyz[0], omega_xyz[1], omega_xyz[2], axes='rxyz')
        # found to be intrinsic Euler angles (r)

        # Detect contact of feet with ground plane
        c = bool(len([c[8] for c in p.getContactPoints(self.bot, self.plane, 1)]))

        # dq = [j[1] for j in p.getJointStates(self.bot, range(8))]

        if useRealTime == 0:
            p.stepSimulation()

        return q, b_orient, c, torque
