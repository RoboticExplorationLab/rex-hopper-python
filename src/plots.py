"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use(['science', 'no-latex'])
plt.rcParams['lines.linewidth'] = 2
import matplotlib.ticker as plticker

plt.rcParams['font.size'] = 16


def thetaplot(total, theta_hist, setp_hist, tau_hist, dq_hist):
    fig, axs = plt.subplots(3, 3, sharex='all', sharey='row')
    plt.xlabel("Timesteps")

    axs[0, 0].plot(range(total), theta_hist[:, 0] * 180 / np.pi, color='blue', label='Measurement')
    axs[0, 0].plot(range(total), setp_hist[:, 0] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_title('Theta 1')
    axs[0, 0].set_ylabel('Angle (deg)')

    axs[0, 1].plot(range(total), theta_hist[:, 1] * 180 / np.pi, color='blue', label='Measurement')
    axs[0, 1].plot(range(total), setp_hist[:, 1] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 1].set_title('Theta 2')

    axs[0, 2].plot(range(total), theta_hist[:, 2] * 180 / np.pi, color='blue', label='Measurement')
    axs[0, 2].plot(range(total), setp_hist[:, 2] * 180 / np.pi, color='red', label='Setpoint')
    axs[0, 2].set_title('Yaw')

    axs[1, 0].plot(range(total), tau_hist[:, 2], color='orange')
    axs[1, 0].set_title('rw0')
    axs[1, 0].set_ylabel('Torque (Nm)')

    axs[1, 1].plot(range(total), tau_hist[:, 3], color='orange')
    axs[1, 1].set_title('rw1')

    axs[1, 2].plot(range(total), tau_hist[:, 4], color='orange')
    axs[1, 2].set_title('rwz')

    axs[2, 0].plot(range(total), dq_hist[:, 2] * 60 / (2 * np.pi), color='g')
    axs[2, 0].set_title('rw0')
    axs[2, 0].set_ylabel('Angular Vel (RPM')

    axs[2, 1].plot(range(total), dq_hist[:, 3] * 60 / (2 * np.pi), color='g')
    axs[2, 1].set_title('rw0')

    axs[2, 2].plot(range(total), dq_hist[:, 4] * 60 / (2 * np.pi), color='g')
    axs[2, 2].set_title('rw0')

    plt.show()


def tauplot(model, total, n_a, tau_hist, u_hist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, tau_hist[:, k], c='r', label="actual")
        ax.plot(totalr, u_hist[:, k], c='g', label="control")
        ax.set_ylabel('Torque, Nm')
        ax.set_title(model["aname"][k])
        ax.legend()

    plt.xlabel("Timesteps")

    plt.show()


def dqplot(model, total, n_a, dq_hist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, dq_hist[:, k] * 60 / (2 * np.pi))
        ax.set_ylabel('Angular Velocity, RPM')
        ax.set_title(model["aname"][k])
    plt.xlabel("Timesteps")
    plt.show()


def f_plot(total, f_hist, grf_hist, s_hist, statem_hist):
    fig, axs = plt.subplots(5, sharex='all')
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), grf_hist[:, 0], color='r', label="Actual GRF")
    axs[0].plot(range(total), f_hist[:, 0], color='b', label="Force Ref")
    axs[0].set_title('X Ground Reaction Force')
    axs[0].set_ylabel("Force, N")
    axs[0].set_ylim(-300, 300)

    axs[1].plot(range(total), grf_hist[:, 1], color='r')
    axs[1].plot(range(total), f_hist[:, 1], color='b')
    axs[1].set_title('Y Ground Reaction Force')
    axs[1].set_ylabel("Force, N")
    axs[1].set_ylim(-300, 300)

    axs[2].plot(range(total), grf_hist[:, 2], color='r')
    axs[2].plot(range(total), f_hist[:, 2], color='b')
    axs[2].set_title('Z Ground Reaction Force')
    axs[2].set_ylabel("Force, N")
    axs[2].set_ylim(-300, 300)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper center')

    # axs[3].plot(range(total), s_hist[:, 0], color='green', lw='2', ls="--", label='Contact Schedule')
    axs[3].plot(range(total), statem_hist, color='cyan', lw='1', ls="-", label='State Machine States')
    axs[3].set_title('Original Contact Schedule')
    axs[3].set_ylabel("True/False")

    axs[4].plot(range(total), s_hist[:, 1], color='purple', lw='2', ls="--", label='Updated Schedule')
    axs[4].plot(range(total), s_hist[:, 2], color='orange', lw='1', ls="-", label='Actual')
    axs[4].set_title('Actual and Scheduled Contact')
    axs[4].set_ylabel("True/False")
    axs[4].legend(loc="upper right")

    plt.show()


def currentplot(total, n_a, a_hist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, a_hist[:, k])
        ax.set_ylabel("current (A)")
    plt.xlabel("Timesteps")
    plt.show()


def voltageplot(total, n_a, v_hist):
    cols = 3
    rows = n_a // cols
    rows += n_a % cols
    position = range(1, n_a + 1)
    fig = plt.figure(1)
    totalr = range(total)
    for k in range(n_a):
        ax = fig.add_subplot(rows, cols, position[k])
        ax.plot(totalr, v_hist[:, k])
        ax.set_ylabel("voltage (V)")
    plt.xlabel("Timesteps")
    plt.show()


def etotalplot(total, a_hist, v_hist, dt):
    ain_hist = np.sum(a_hist, axis=1)
    vmean_hist = np.average(v_hist, axis=1)
    power_array = a_hist @ v_hist.T
    power_hist = np.diag(power_array)
    fig, axs = plt.subplots(3)
    plt.xlabel("Timesteps")

    axs[0].plot(range(total), ain_hist, color='blue', label='a_in')
    axs[0].set_title('Total Current Draw')
    axs[0].set_ylabel("Current (A)")

    print("Mean current draw is ", np.mean(ain_hist), " A")
    print("Peak current draw is ", np.amax(ain_hist), " A")

    axs[1].plot(range(total), vmean_hist, color='blue', label='v_mean')
    axs[1].set_title('Mean Voltage draw')
    axs[1].set_ylabel("Voltage (V)")

    axs[2].plot(range(total), power_hist, color='blue', label='power in')
    axs[2].set_title('Total Power')
    axs[2].set_ylabel("Power (W)")

    print("Mean power draw is ", np.mean(power_hist), " W")
    print("Peak power draw is ", np.amax(power_hist), " W")
    energy = np.trapz(power_hist, dx=dt)
    print("Total energy used is ", energy, " Joules, or ", energy / (48 * 3600), " Ah in ", np.shape(power_hist)[0] * dt,
          " s")
    plt.show()


def set_axes_equal(ax: plt.Axes):
    """
    https://stackoverflow.com/questions/13685386/
    matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def posplot_3d(p_hist, pf_hist, ref_traj, pf_ref, pf_list, pf_list0, dist):
    ax = plt.axes(projection='3d')
    # ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, dist)
    ax.set_ylim3d(-dist/2, dist/2)
    ax.set_zlim3d(0, 2)
    ax.scatter(*p_hist[0, :], color='green', marker="*", s=200, label='Starting Position')
    ax.scatter(*ref_traj[-1, 0:3], marker="*", s=200, color='orange', label='Target Position')
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], color='green', ls='--', label='Ref CoM Traj')
    ax.plot(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='cyan', ls='--', label='Ref Foot Traj')
    ax.scatter(pf_list0[:, 0], pf_list0[:, 1], pf_list0[:, 2], color='cyan', marker="x", s=200, label='Ref Footsteps')

    ax.plot(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], color='red', label='CoM Position')
    ax.plot(pf_hist[:, 0], pf_hist[:, 1], pf_hist[:, 2], color='blue', label='Foot Position')
    ax.scatter(pf_list[:, 0], pf_list[:, 1], pf_list[:, 2],
               marker="x", s=200, color='blue', label='Updated Footsteps')

    ax.legend()
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    ax.set_box_aspect([1, 1, 1])  # make aspect ratio equal for all axes
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax)  # IMPORTANT - this is also required

    plt.show()


def animate_line(N, ref_traj, pf_ref, p_hist, pf_hist, ref, pfr, com, pf, ax):
    ref._offsets3d = (ref_traj[0:3, :N])
    pfr._offsets3d = (pf_ref[0:3, :N])
    com._offsets3d = (p_hist[0:3, :N])
    pf._offsets3d = (pf_hist[0:3, :N])

    # ax.view_init(elev=10., azim=N)


def posplot_animate(p_hist, pf_hist, ref_traj, pf_ref, ref_traj0, dist):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Body Position')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim3d(0, dist)
    ax.set_ylim3d(-dist/2, dist/2)
    ax.set_zlim3d(0, 2)

    ax.scatter(*p_hist[0, :], color='green', marker="*", s=200, label='Starting Position')
    ax.scatter(*ref_traj0[-1, 0:3], color='orange', marker="*", s=200, label='Target Position')
    ax.plot(ref_traj0[:, 0], ref_traj0[:, 1], ref_traj0[:, 2], ls='--', c='g', label='Reference CoM Traj')
    intervals = 2
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.yaxis.set_minor_locator(loc)
    ax.zaxis.set_minor_locator(loc)
    # Add the grid
    ax.grid(which='minor', axis='both', linestyle='-')
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30

    N = len(p_hist)

    ref = ax.scatter(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], c='m', lw=2, label='Updated Ref CoM Traj')
    pfr = ax.scatter(pf_ref[:, 0], pf_ref[:, 1], pf_ref[:, 2], color='y',  label='Updated Ref Foot Traj')
    com = ax.scatter(p_hist[:, 0], p_hist[:, 1], p_hist[:, 2], c='r', lw=2, label='CoM Position')
    pf = ax.scatter(pf_hist[:, 0], pf_hist[:, 1], pf_hist[:, 2], color='b', label='Foot Position')

    ax.legend()
    line_ani = animation.FuncAnimation(fig, animate_line, frames=N,
                                       fargs=(ref_traj.T, pf_ref.T, p_hist.T, pf_hist.T,
                                              ref, pfr, com, pf, ax),
                                       interval=2, blit=False)
    # line_ani.save('basic_animation.mp4', fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])

    plt.show()
