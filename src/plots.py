"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt

def thetaplot(total, hist1, hist2, hist3, hist4, hist5, hist6):

    fig, axs = plt.subplots(2, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), hist1*180/np.pi, color='blue')
    axs[0, 0].set_title('Theta_x')
    axs[0, 0].set_ylabel('Theta_x (deg)')

    axs[0, 1].plot(range(total), hist2*180/np.pi, color='blue')
    axs[0, 1].set_title('Theta_y')
    axs[0, 1].set_ylabel('Theta_y (deg)')

    axs[0, 2].plot(range(total), hist3*180/np.pi, color='blue')
    axs[0, 2].set_title('Theta_z')
    axs[0, 2].set_ylabel('Theta_z (deg)')

    axs[1, 0].plot(range(total), hist4*60/(2*np.pi), color='blue')
    axs[1, 0].set_title('RW1 Angular Velocity')
    axs[1, 0].set_ylabel('Angular Velocity, RPM')

    axs[1, 1].plot(range(total), hist5*60/(2*np.pi), color='blue')
    axs[1, 1].set_title('RW2 Angular Velocity')
    axs[1, 1].set_ylabel('Angular Velocity, RPM')

    axs[1, 2].plot(range(total), hist6*60/(2*np.pi), color='blue')
    axs[1, 2].set_title('RWZ Angular Velocity')
    axs[1, 2].set_ylabel('Angular Velocity, RPM')

    plt.show()


def tauplot(total, hist1, hist2, hist3, hist4, hist5, hist6):

    fig, axs = plt.subplots(2, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), hist1, color='blue')
    axs[0, 0].set_title('q0 torque')
    axs[0, 0].set_ylabel("q0 torque (Nm)")

    axs[0, 1].plot(range(total), hist2, color='blue')
    axs[0, 1].set_title('q1 torque')
    axs[0, 1].set_ylabel("q1 torque (Nm)")

    axs[0, 2].plot(range(total), hist3, color='blue')
    axs[0, 2].set_title('base z position')
    axs[0, 2].set_ylabel("z position (m)")

    axs[1, 0].plot(range(total), hist4, color='blue')
    axs[1, 0].set_title('Magnitude of X Reaction Force on joint1')
    axs[1, 0].set_ylabel("Reaction Force Fx, N")

    axs[1, 1].plot(range(total), hist5, color='blue')
    axs[1, 1].set_title('Magnitude of Z Reaction Force on joint1')  # .set_title('angular velocity q1_dot')
    axs[1, 1].set_ylabel("Reaction Force Fz, N")  # .set_ylabel("angular velocity, rpm")

    axs[1, 2].plot(range(total), hist6, color='blue')
    axs[1, 2].set_title('Flight Time')
    axs[1, 2].set_ylabel("Flight Time, s")

    plt.show()