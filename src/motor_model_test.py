import actuator
import numpy as np
import matplotlib.pyplot as plt


def test(dt, i_max, omega_max, tau_stall, gr_out, kt=None, r=None, verbose=False):

    motor_test = actuator.Actuator(dt=dt, i_max=i_max, gr_out=gr_out,
                                   tau_stall=tau_stall, omega_max=omega_max, kt=kt, r=r)
    print("omega_max = ", omega_max)
    print("theoretical kt = ", motor_test.v_max/omega_max)
    print("input kt = ", motor_test.kt_m)
    print("theoretical r = ", (motor_test.v_max ** 2) / (omega_max * tau_stall))
    print("input r = ", motor_test.r)

    n = 200
    q_dot_max = omega_max / gr_out * 3
    tau = np.zeros((n, n))
    q_dot_k = np.zeros(n)

    j = -1
    g = 0
    for i in np.linspace(-i_max, i_max, n):
        j += 1
        k = -1
        if j > (g + 19) and verbose is True:
            print(round(j / n * 100, 1), " percent complete")
            g = j
        for q_dot in np.linspace(-q_dot_max, q_dot_max, n):
            k += 1
            # tau[j, k] = motor_test.actuate_sat(i=i, q_dot=q_dot)
            tau[j, k] = motor_test.actuate(i=i, q_dot=q_dot)
            q_dot_k[k] = q_dot

        plt.scatter(tau[j, :], q_dot_k, color='red', marker="o", s=2)
        # plt.scatter(tau_sat[j, :], q_dot_k, color='green', marker="o")

    plt.title('Motor Model')
    plt.ylabel("q_dot (rad/s)")
    plt.xlabel("tau (N*m)")

    plt.show()
    return None


# RMD-X10
print("RMD-X10")
gr_out = 7
omega_max = 250 * gr_out * (2 * np.pi / 60)
test(dt=1/1000, i_max=13, omega_max=omega_max, tau_stall=50 / gr_out, gr_out=gr_out) # , kt=1.73/gr_out, r=0.3)
print("\n")
# EA110 100KV
print("EA110-100KV")
omega_max = 3490 * (2 * np.pi / 60)
test(dt=1/1000, i_max=92.5, omega_max=omega_max, tau_stall=11.24, gr_out=1) # , kt=8.4/100, r=33/1000)
print("\n")
# EA110 100KV
print("R100-90KV")
omega_max = 3800 * (2 * np.pi / 60)
test(dt=1/1000, i_max=104, omega_max=omega_max, tau_stall=11, gr_out=1) #, kt=0.106, r=51/1000)

