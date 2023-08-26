import numpy as np
import matplotlib.pyplot as plt

# Constants
C_m = 1.0  # (uF/cm^2)
g_Na = 120.0  # (mS/cm^2)
g_K = 36.0  # (mS/cm^2)
g_L = 0.3  # (mS/cm^2)
E_Na = 50.0  # (mV)
E_K = -77.0  # (mV)
E_L = -55  # (mV)

# Initial conditions
V0 = -65.0  # initial membrane potential (mV)
m0 = 0.07 # IDK these values were hit and trial 
h0 = 0.4  
n0 = 0.4  

initial_state = np.array([V0, m0, h0, n0])

# Time variables
T = 30 # (ms)
deltaT = 0.01 # (ms)
num = int(5*T/deltaT)
t_span = (0, 5*T)  # simulation time span (ms)
t_eval = np.linspace(*t_span, num=num)

# External current
def ext_current(t):
    return 15.0 if 2*T <= t < 3*T else 0.0

# Hodgkin-Huxley model equations
def hodgkin_huxley(t, y):
    V, m, h, n = y

    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))
    beta_n = 0.125 * np.exp(-0.0125 * (V + 65))
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))
    beta_m = 4.0 * np.exp(-0.0556 * (V + 65))
    alpha_h = 0.07 * np.exp(-0.05 * (V + 65))
    beta_h = 1.0 / (1 + np.exp(-0.1 * (V + 35)))

    dVdt = (ext_current(t) - g_Na * m**3 * h * (V - E_Na) - g_K * n**4 * (V - E_K) - g_L * (V - E_L)) / C_m
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n

    return np.array([dVdt, dmdt, dhdt, dndt])

# Fourth-order Runge-Kutta method
def runge_kutta_4(func, y0, t_values):
    h = t_values[1] - t_values[0]
    y = y0
    y_values = [y]

    for t in t_values[:-1]:
        k1 = h * func(t, y)
        k2 = h * func(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * func(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * func(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        y_values.append(y)

    return np.array(y_values)

# Solve using Runge-Kutta method
sol_rk = runge_kutta_4(hodgkin_huxley, initial_state, t_eval)

# Extract state variables
V_values = sol_rk[:, 0]
m_values = sol_rk[:, 1]
h_values = sol_rk[:, 2]
n_values = sol_rk[:, 3]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_eval, V_values, label='Membrane Potential (mV)')
plt.plot(t_eval, [ext_current(t) for t in t_eval], label="Input current (uA)")
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Neuron Simulation')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_eval, [ext_current(t) for t in t_eval], label="Input current (uA)")
plt.plot(t_eval, [g_Na * m**3 * h * (v - E_Na) * 1.0e-1 for m,h,v in zip(m_values, h_values, V_values)], label="Sodium channel current (10uA)")
plt.plot(t_eval, [g_K * n**4 * (v - E_K) * 1.0e-1 for v,n in zip(V_values, n_values)], label="Potassium channel current (10uA)")
plt.plot(t_eval, [g_L * (v - E_L) for v in V_values], label="Leak current (uA)")
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Neuron Simulation')
plt.legend()
plt.grid()
plt.show()
