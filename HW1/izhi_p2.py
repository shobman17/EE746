import numpy as np
import matplotlib.pyplot as plt

# Izhikevich neuron parameters
a = [0.03,0.01,0.03]
b = [-2,5,1]
c = [-50,-56,-40]
d = [100,130,150]
kz = [0.7,1.2,1.5]

# Spike threshold
v_threshold = [35,50,25]
Er = [-60,-75,-60]
Et = [-40,-45,-40]
names = ["RS","IB","CH"]
# Time parameters
dt = 0.01  # Time step
T = 100  # Total simulation time
num_steps = int(T / dt)

# Input current
I = [400,500,600]  # Applied current

n_neuron = 0


def izhikevich_derivatives(v, u, n_neuron, n_i):
    dv = kz[n_neuron]*(v - Et[n_neuron])*(v - Er[n_neuron]) - u + I[n_i]
    du = a[n_neuron] * (b[n_neuron] * (v-Er[n_neuron]) - u)

    return dv, du

def runge_kutta_update(v, u, dt, n_neuron, n_i):
    k1v, k1u = izhikevich_derivatives(v, u, n_neuron, n_i)
    k2v, k2u = izhikevich_derivatives(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u, n_neuron, n_i)
    k3v, k3u = izhikevich_derivatives(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u, n_neuron, n_i)
    k4v, k4u = izhikevich_derivatives(v + dt * k3v, u + dt * k3u, n_neuron, n_i)

    v_new = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    u_new = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)

    
    if v_new >= v_threshold[n_neuron]:
        v_new = c[n_neuron]
        u_new += d[n_neuron]
        

    return v_new, u_new
# Initial conditions
v = Er[2]  # Membrane potential
u = b[0] * v  # Recovery variable

# Arrays to store results
v_values = np.zeros((3,3,num_steps))


# Simulate using Runge-Kutta
for j in range(3):
    for k in range(3):
        for i in range(num_steps):
            v_values[k][j][i] = v
            v, u  = runge_kutta_update(v, u, dt, j, k)
       

# Plot the results
time = np.arange(0, T, dt)

fig,ax = plt.subplots(3,2)

for j in range(3):
    ax[0][0].plot(time, v_values[j][0], label=str(I[j]))
#ax[0][0].set_xlabel('Time (ms)')
ax[0][0].set_ylabel('Membrane Potential (mV)')
ax[0][0].set_title('Izhikevich Neuron Simulation for RS neuron')
ax[0][0].legend()
ax[0][0].grid()

for j in range(3):
    ax[1][0].plot(time, v_values[j][1], label=str(I[j]))
#ax[1][0].set_xlabel('Time (ms)')
ax[1][0].set_ylabel('Membrane Potential (mV)')
ax[1][0].set_title('Izhikevich Neuron Simulation for IB neuron')
ax[1][0].legend()
ax[1][0].grid()

for j in range(3):
    ax[2][0].plot(time, v_values[j][2], label=str(I[j]))
#ax[2][0].set_xlabel('Time (ms)')
ax[2][0].set_ylabel('Membrane Potential (mV)')
ax[2][0].set_title('Izhikevich Neuron Simulation for CH neuron')
ax[2][0].legend()
ax[2][0].grid()

for j in range(3):
    ax[0][1].plot(time, v_values[0][j], label=names[j])
#ax[0][1].set_xlabel('Time (ms)')
ax[0][1].set_ylabel('Membrane Potential (mV)')
ax[0][1].set_title('Izhikevich Neuron Simulation for Iapp = 400')
ax[0][1].legend()
ax[0][1].grid()

for j in range(3):
    ax[1][1].plot(time, v_values[1][j], label=names[j])
#ax[1][1].set_xlabel('Time (ms)')
ax[1][1].set_ylabel('Membrane Potential (mV)')
ax[1][1].set_title('Izhikevich Neuron Simulation for Iapp = 500')
ax[1][1].legend()
ax[1][1].grid()

for j in range(3):
    ax[2][1].plot(time, v_values[2][j], label=names[j])
#ax[2][1].set_xlabel('Time (ms)')
ax[2][1].set_ylabel('Membrane Potential (mV)')
ax[2][1].set_title('Izhikevich Neuron Simulation for Iapp = 600')
ax[2][1].legend()
ax[2][1].grid()



plt.show()


print (v_values[0])