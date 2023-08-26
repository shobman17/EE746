import numpy as np
import matplotlib.pyplot as plt

# Constants
C_m = 1.0  # (uF/cm^2)
g_Na = 120.0  # (mS/cm^2)
g_K = 36.0  # (mS/cm^2)
g_L = 0.3  # (mS/cm^2)
E_Na = 50.0  # (mV)
E_K = -77.0  # (mV)
E_L = -55.0  # (mV)

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

# Extract currents
iNa = [g_Na * m**3 * h * (v - E_Na) for m,h,v in zip(m_values, h_values, V_values)]
iK = [g_K * n**4 * (v - E_K) for v,n in zip(V_values, n_values)]
iL = [g_L * (v - E_L) for v in V_values]

# Bunch everything together
channels = ["Na", "K", "Leakage"]
E = [E_Na, E_K, E_L]
i = [iNa, iK, iL]

# Plot the membrane potential
plt.figure(figsize=(10, 6))
plt.plot(t_eval, V_values, label='Membrane Potential (mV)')
plt.plot(t_eval, [ext_current(t) for t in t_eval], label="Input current (uA)")
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Neuron Simulation')
plt.legend()
plt.grid()
plt.show()

# Plot the currents
plt.figure(figsize=(10, 6))
plt.plot(t_eval, [ext_current(t) for t in t_eval], label="Input current (uA)")
for j in range(3):
    plt.plot(t_eval, i[j], label = f"{channels[j]} channel current (uA)")
plt.xlabel('Time (ms)')
plt.ylabel('Ion channel currents (uA)')
plt.title('Hodgkin-Huxley Neuron Simulation')
plt.legend()
plt.grid()
plt.show()

one_cycle = np.intersect1d(np.where(t_eval < 70), np.where(t_eval > 60)) # One action potential cycle
one_cycle = one_cycle.tolist()
t_one = t_eval[one_cycle]

power = []
for j in range(3):
    power.append([ix*(E[j] - v) for v,ix in zip(V_values[one_cycle[0]:one_cycle[-1] + 1], i[j][one_cycle[0]: one_cycle[-1] + 1])])


# Plot the instantaneous power
plt.figure(figsize=(10, 6))
plt.plot(t_one, [ext_current(t) for t in t_one], label="Input current (uA)")
for j in range(3):
    plt.plot(t_one, np.array(power[j])*1.0e-3, label = f"{channels[j]} channel power (uW)")
plt.xlabel('Time (ms)')
plt.ylabel('Instantaneous power (uW)')
plt.title('Hodgkin-Huxley Neuron Simulation for one action cycle')
plt.legend()
plt.grid()
plt.show()

I_ext = []
for t in t_one:
    I_ext.append(ext_current(t))

membrane_power = [v*(I - i_Na - i_K - i_L) for v,I,i_Na,i_K,i_L in zip(V_values[one_cycle[0]: one_cycle[-1] + 1], I_ext, iNa[one_cycle[0]: one_cycle[-1] + 1], iK[one_cycle[0]: one_cycle[-1] + 1], iL[one_cycle[0]: one_cycle[-1] + 1])]
power.append(membrane_power)

# Plot membrane charge/discharge power
plt.figure(figsize=(10, 6))
plt.plot(t_one, I_ext, label="Input current (uA)")
plt.plot(t_one, np.array(power[3])*1e-3, label = "Membrance charge/discharge power (uW)")
plt.xlabel('Time (ms)')
plt.ylabel('Power (uW)')
plt.title('Hodgkin-Huxley Neuron Simulation for one action cycle')
plt.legend()
plt.grid()
plt.show()

# Integrating the power
timestep = t_one[1] - t_one[0]
ener = [0,0,0,0] # Na, K, l, membrane
for t in range(len(t_one)):
    for i in range(4):
        ener[i] += power[i][t]*timestep

print(f"Na channel energy spent = {ener[0]} pJ")
print(f"K channel energy spent = {ener[1]} pJ")
print(f"Leakage channel energy spent = {ener[2]} pJ")
print(f"Membrane capacitance channel energy spent = {ener[3]} pJ")