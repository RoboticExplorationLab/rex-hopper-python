"""
Copyright (C) 2022 Benjamin Bokser
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'no-latex'])

import actuator
import actuator_param


def test(dt, model, verbose=False):

    motor_test = actuator.Actuator(dt=dt, model=model)
    
    print("omega_max = ", motor_test.omega_max / (2 * np.pi / 60), " RPM")
    print("kt = ", motor_test.kt)
    print("r = ", motor_test.r)
    print("tau_max (before gearing) = ", motor_test.tau_max)
    gr = model["gr"]
    omega_max = motor_test.omega_max
    i_max = motor_test.i_max

    n = 200
    q_dot_max = omega_max / gr * 1.2
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
            tau[j, k], current, voltage = motor_test.actuate(i=i, q_dot=q_dot)
            q_dot_k[k] = q_dot

        plt.scatter(tau[j, :], q_dot_k, color='red', marker="o", s=2)
        # plt.scatter(tau_sat[j, :], q_dot_k, color='green', marker="o")

    plt.title('Motor Operating Region')
    plt.ylabel("Angular Velocity (rad/s)")
    plt.xlabel("Tau (N*m)")

    plt.show()
    return None


model = actuator_param.actuator_r100kv90
print(model["name"])
test(dt=1/1000, model=model)
print("\n")

