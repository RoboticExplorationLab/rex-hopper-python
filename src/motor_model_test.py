import actuator
import numpy as np
import matplotlib.pyplot as plt

i_max = 13
gr_out = 7
omega_max = 190 * 7 * (2 * np.pi / 60)
q_dot_max = omega_max / gr_out * 4

tau_stall = 50 / 7
motor_test = actuator.Actuator(i_max=i_max, gr_out=gr_out, tau_stall=tau_stall, omega_max=omega_max)
v_max = motor_test.v_max
print("omega_max = ", motor_test.omega_max)
print("v_max = ", v_max)
n = 100

tau = np.zeros((n, n))
tau_sat = np.zeros((n, n))
q_dot_k = np.zeros(n)

j = -1
for i in np.linspace(-i_max, i_max, n):
    j += 1
    k = -1
    print(j/n*100, " percent complete")
    for q_dot in np.linspace(-q_dot_max, q_dot_max, n):
        k += 1
        # tau_sat[j, k] = motor_test.actuate_sat(i=i, q_dot=q_dot)
        tau[j, k] = motor_test.actuate(i=i, q_dot=q_dot)
        q_dot_k[k] = q_dot

    plt.scatter(tau[j, :], q_dot_k, color='red', marker="o", s=2)
    # plt.scatter(tau_sat[j, :], q_dot_k, color='green', marker="o")

plt.title('Motor Model')
plt.ylabel("q_dot (rad/s)")
plt.xlabel("tau (N*m)")

plt.show()
