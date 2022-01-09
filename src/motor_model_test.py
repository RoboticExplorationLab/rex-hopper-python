import actuator
import numpy as np
import matplotlib.pyplot as plt

i_max = 13
gr_out = 4
omega_max = 190 * 7 * (2 * np.pi / 60)
print("omega_max = ", omega_max)
tau_stall = 50 / 7
motor_test = actuator.Actuator(i_max=i_max, v_max=48, gr_out=gr_out, tau_stall=tau_stall, omega_max=omega_max)

n = 100

tau = np.zeros((n, n))
tau_sat = np.zeros((n, n))
omega_k = np.zeros(n)

j = -1
for i in np.linspace(-i_max, i_max, n):
    j += 1
    k = -1
    print(j/n*100, " percent complete")
    for omega in np.linspace(-omega_max, omega_max, n):
        k += 1
        # tau_sat[j, k] = motor_test.actuate_sat(i=i, omega=omega)
        tau[j, k] = motor_test.actuate(i=i, omega=omega)
        omega_k[k] = omega

    plt.scatter(tau[j, :], omega_k, color='red', marker="o")
    # plt.scatter(tau_sat[j, :], omega_k, color='green', marker="o")

plt.title('Motor Model')
plt.ylabel("omega (rad/s)")
plt.xlabel("tau (N*m)")

plt.show()
