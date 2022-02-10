"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt


def rwplot(total, thetahist, rw_tauhist, rw_whist, setphist):

    fig, axs = plt.subplots(3, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), thetahist[:, 0]*180/np.pi, color='blue', label='Measurement')
    axs[0, 0].plot(range(total), setphist[:, 0] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].set_title('Theta_1')
    axs[0, 0].set_ylabel('Theta_1 (deg)')

    axs[0, 1].plot(range(total), thetahist[:, 1]*180/np.pi, color='blue', label='Measurement')
    axs[0, 1].plot(range(total), setphist[:, 1] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].set_title('Theta_2')
    axs[0, 1].set_ylabel('Theta_2 (deg)')

    axs[0, 2].plot(range(total), thetahist[:, 2]*180/np.pi, color='blue', label='Measurement')
    axs[0, 2].plot(range(total), setphist[:, 2] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 2].legend(loc="lower right")
    axs[0, 2].set_title('Yaw')
    axs[0, 2].set_ylabel('Yaw (deg)')

    axs[1, 0].plot(range(total), rw_tauhist[:, 0], color='blue')
    axs[1, 0].set_title('RW1 Torque')
    axs[1, 0].set_ylabel('Torque, Nm')

    axs[1, 1].plot(range(total), rw_tauhist[:, 1], color='blue')
    axs[1, 1].set_title('RW2 Torque')
    axs[1, 1].set_ylabel('Torque, Nm')

    axs[1, 2].plot(range(total), rw_tauhist[:, 2], color='blue')
    axs[1, 2].set_title('RWZ Torque')
    axs[1, 2].set_ylabel('Torque, Nm')

    axs[2, 0].plot(range(total), rw_whist[:, 0]*60/(2*np.pi), color='blue')
    axs[2, 0].set_title('RW1 Angular Velocity')
    axs[2, 0].set_ylabel('Angular Velocity, RPM')

    axs[2, 1].plot(range(total), rw_whist[:, 1]*60/(2*np.pi), color='blue')
    axs[2, 1].set_title('RW2 Angular Velocity')
    axs[2, 1].set_ylabel('Angular Velocity, RPM')

    axs[2, 2].plot(range(total), rw_whist[:, 2]*60/(2*np.pi), color='blue')
    axs[2, 2].set_title('RWZ Angular Velocity')
    axs[2, 2].set_ylabel('Angular Velocity, RPM')

    plt.show()


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


def tauplot(total, tau0hist, tau2hist, pzhist, fxhist, fzhist, fthist):

    fig, axs = plt.subplots(2, 3, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), tau0hist, color='blue')
    axs[0, 0].set_title('q0 torque')
    axs[0, 0].set_ylabel("q0 torque (Nm)")

    axs[0, 1].plot(range(total), tau2hist, color='blue')
    axs[0, 1].set_title('q1 torque')
    axs[0, 1].set_ylabel("q1 torque (Nm)")

    axs[0, 2].plot(range(total), pzhist, color='blue')
    axs[0, 2].set_title('base z position')
    axs[0, 2].set_ylabel("z position (m)")

    axs[1, 0].plot(range(total), fxhist, color='blue')
    axs[1, 0].set_title('Magnitude of X Reaction Force on joint1')
    axs[1, 0].set_ylabel("Reaction Force Fx, N")

    axs[1, 1].plot(range(total), fzhist, color='blue')
    axs[1, 1].set_title('Magnitude of Z Reaction Force on joint1')  # .set_title('angular velocity q1_dot')
    axs[1, 1].set_ylabel("Reaction Force Fz, N")  # .set_ylabel("angular velocity, rpm")

    axs[1, 2].plot(range(total), fthist, color='blue')
    axs[1, 2].set_title('Flight Time')
    axs[1, 2].set_ylabel("Flight Time, s")

    plt.show()


def electrplot(total, q0ahist, q2ahist, rw1ahist, rw2ahist, rwzahist, q0vhist, q2vhist, rw1vhist, rw2vhist, rwzvhist):

    fig, axs = plt.subplots(2, 5, sharex='all')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), q0ahist, color='blue')
    axs[0, 0].set_title('q0 current')
    axs[0, 0].set_ylabel("current (A)")
    #axs[0, 0].plot(range(total), q0ahist[:, 1], color='red')

    axs[0, 1].plot(range(total), q2ahist, color='blue')
    axs[0, 1].set_title('q1 current')
    axs[0, 1].set_ylabel("current (A)")
    #axs[0, 1].plot(range(total), q2ahist[:, 1], color='red')

    axs[0, 2].plot(range(total), rw1ahist, color='blue')
    axs[0, 2].set_title('rw1 current')
    axs[0, 2].set_ylabel("current (A)")
    #axs[0, 2].plot(range(total), rw1ahist[:, 1], color='red')

    axs[0, 3].plot(range(total), rw2ahist, color='blue')
    axs[0, 3].set_title('rw2 current')
    axs[0, 3].set_ylabel("current (A)")
    #axs[0, 3].plot(range(total), rw2ahist[:, 1], color='red')

    axs[0, 4].plot(range(total), rwzahist, color='blue')
    axs[0, 4].set_title('rwz current')
    axs[0, 4].set_ylabel("current (A)")
    #axs[0, 4].plot(range(total), rwzahist[:, 1], color='red')

    axs[1, 0].plot(range(total), q0vhist[:, 0], color='blue')
    axs[1, 0].set_title('q0 voltage')
    axs[1, 0].set_ylabel("voltage (V)")
    #axs[1, 0].plot(range(total), q0vhist[:, 1], color='red')

    axs[1, 1].plot(range(total), q2vhist[:, 0], color='blue')
    axs[1, 1].set_title('q1 voltage')
    axs[1, 1].set_ylabel("voltage (V)")
    # axs[1, 1].plot(range(total), q2vhist[:, 1], color='red')

    axs[1, 2].plot(range(total), rw1vhist[:, 0], color='blue')
    axs[1, 2].set_title('rw1 voltage')
    axs[1, 2].set_ylabel("voltage (V)")
    #axs[1, 2].plot(range(total), rw1vhist[:, 1], color='red')

    axs[1, 3].plot(range(total), rw2vhist[:, 0], color='blue')
    axs[1, 3].set_title('rw2 voltage')
    axs[1, 3].set_ylabel("voltage (V)")
    #axs[1, 3].plot(range(total), rw2vhist[:, 1], color='red')

    axs[1, 4].plot(range(total), rwzvhist[:, 0], color='blue')
    axs[1, 4].set_title('rwz voltage')
    axs[1, 4].set_ylabel("voltage (V)")
    #axs[1, 4].plot(range(total), rwzvhist[:, 1], color='red')

    plt.show()


def electrtotalplot(total, q0ahist, q2ahist, rw1ahist, rw2ahist, rwzahist,
                    q0vhist, q2vhist, rw1vhist, rw2vhist, rwzvhist, dt):

    # ainhist = np.sum([q0ahist[:, 0], q2ahist[:, 0], rw1ahist[:, 0], rw2ahist[:, 0], rwzahist[:, 0]], axis=0)
    ainhist = np.sum([q0ahist, q2ahist, rw1ahist, rw2ahist, rwzahist], axis=0)
    # vinhist = np.sum([q0vhist[:, 0], q2vhist[:, 0], rw1vhist[:, 0], rw2vhist[:, 0], rwzvhist[:, 0]], axis=0)
    vbackemfhist = np.sum([q0vhist[:, 1], q2vhist[:, 1], rw1vhist[:, 1], rw2vhist[:, 1], rwzvhist[:, 1]], axis=0)
    vmeanhist = np.average([q0vhist[:, 0], q2vhist[:, 0], rw1vhist[:, 0], rw2vhist[:, 0], rwzvhist[:, 0]], axis=0)
    # ainhist_cat = np.vstack([q0ahist[:, 0], q2ahist[:, 0], rw1ahist[:, 0], rw2ahist[:, 0], rwzahist[:, 0]]).T
    ainhist_cat = np.vstack([q0ahist, q2ahist, rw1ahist, rw2ahist, rwzahist]).T
    vinhist_cat = np.vstack([q0vhist[:, 0], q2vhist[:, 0], rw1vhist[:, 0], rw2vhist[:, 0], rwzvhist[:, 0]])
    power_array = ainhist_cat @ vinhist_cat
    powerinhist = np.diag(power_array)

    # powerinhist = np.multiply(ainhist.T, vinhist.T)
    # powerbackhist = np.multiply(abackemfhist.T, vbackemfhist.T)
    # print(np.shape(ahist))
    # print(np.shape(vhist))
    # print(np.shape(powerhist))
    # print(np.shape(vmeanhist))
    fig, axs = plt.subplots(3, 1, sharex='all')
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), ainhist, color='blue', label='a_in')
    # axs[0].plot(range(total), abackemfhist, color='red', label='back emf current')
    axs[0].set_title('total current draw')
    axs[0].set_ylabel("current (A)")
    # axs[0].legend(loc="upper left")

    print("Mean current draw is ", np.mean(ainhist), " A")
    print("Peak current draw is ", np.amax(ainhist), " A")

    axs[1].plot(range(total), vmeanhist, color='blue', label='v_mean')
    # axs[1].plot(range(total), vbackemfhist, color='red', label='back emf voltage')
    axs[1].set_title('mean voltage draw')
    axs[1].set_ylabel("voltage (V)")
    # axs[1].legend(loc="upper left")

    axs[2].plot(range(total), powerinhist, color='blue', label='power in')
    # axs[2].plot(range(total), powerbackhist, color='red', label='back emf power')
    axs[2].set_title('total power')
    axs[2].set_ylabel("power (W)")
    # axs[2].legend(loc="upper left")

    print("Mean power draw is ", np.mean(powerinhist), " W")
    print("Peak power draw is ", np.amax(powerinhist), " W")
    energy = np.trapz(powerinhist, dx=dt)
    print("Total energy used is ", energy, " Joules, or ", energy/(48*3600), " Ah in ", np.shape(powerinhist)[0]*dt, " seconds")
    plt.show()


def posplot(p_ref, phist, xfhist):

    plt.plot(phist[:, 0], phist[:, 1], color='blue', label='body position')
    plt.title('Body XY Position')
    plt.ylabel("y (m)")
    plt.xlabel("x (m)")
    plt.scatter(xfhist[:, 0], xfhist[:, 1], color='red', marker="o", label='footstep setpoints')
    plt.scatter(0, 0, color='green', marker="x", s=100, label='starting position')
    plt.scatter(p_ref[0], p_ref[1], color='orange', marker="x", s=100, label='position setpoint')
    plt.legend(loc="upper left")

    plt.show()
