# Second order runge kutta from https://lpsa.swarthmore.edu/NumInt/NumIntSecond.html 

import numpy as np
import matplotlib.pyplot as plt

N = 10 # Number of neurons
membrane_V = np.empty((N,1)) # holds all membrane potentials

T = 500e-3 # Simulation time
deltaT = 1e-4 # Time step
M = int(T/deltaT)
timesteps = np.arange(0,T,deltaT)

C = 300e-12
gL = 30e-9
Vt = 20e-3
EL = -70e-3

Ic = 2.7e-9 # Minimum current for a spike
spike_times = [[] for _ in range(N)] # records time of spikes

def LIF_neuron(V,I): # Returns dV/dt for LIF model
    return (I - gL*(V - EL))/C

def update(t, V, I): # update potential from current using runge kutta second order
    
    k1 = LIF_neuron(V,I)
    V1 = V + deltaT*k1
    k2 = LIF_neuron(V1, I)
    V_next = V + (deltaT/2)*(k1 + k2) # bruh

    for i in range(V_next.shape[0]): # Check for spikes
        if(V_next[i] > Vt):
            V_next[i] = EL
            spike_times[i].append(t)
            
    return V_next

if __name__ == "__main__":


    # Part (b)
    '''membrane_V.fill(EL)
    current = np.random.rand(N,M)*1e-9 # Will replace
    output = np.zeros((N,M))   
    for t in range(M):
        I = current[:,[t]]
        membrane_V = update(membrane_V, I)
        output[:,[0]] = membrane_V
    print(output.shape)'''

    # Part (c)
    membrane_V.fill(EL)
    alpha = 0.1
    current = np.empty((N,M))
    output = np.zeros((N,M))
    time = np.arange(0,T, deltaT)

    for i in range(N):
        current_val = (1 + alpha*(i+1))*Ic
        current_row = np.array([current_val]*M)
        current[[i],:] = current_row
    
    for t in range(M):
        I = current[:,[t]]
        membrane_V = update(time[t], membrane_V, I)
        output[:,[t]] = membrane_V

    fig, ax = plt.subplots(4,1, sharex = True, sharey = True)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    ax[0].plot(time*1000, output[1,:]*1000, label = "Neuron 2 current = %.2E A" % current[1,0], color = "c")
    ax[1].plot(time*1000, output[3,:]*1000, label = "Neuron 4 current = %.2E A" % current[3,0], color = "m")
    ax[2].plot(time*1000, output[5,:]*1000, label = "Neuron 6 current = %.2E A" % current[5,0], color = "y")
    ax[3].plot(time*1000, output[7,:]*1000, label = "Neuron 8 current = %.2E A" % current[7,0], color = "k") 

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper center", ncol = 4)

    plt.xlabel("Time (in ms)")
    plt.ylabel("Membrane potential (in mV)")
    plt.title("Leaky Integrate and Fire model")
    plt.show()

    Ic_app = np.array([(1 + alpha*(i+1))*Ic for i in range(N)])
    mean_spike_time = np.array([np.mean(np.diff(np.array(i))) for i in spike_times])
    
    plt.plot(Ic_app*1e9, mean_spike_time*1000)
    plt.xlabel("Applied current (in nA)")
    plt.ylabel("Mean time difference between spikes (in ms)")
    plt.title("Difference in spike time vs Applied current")
    plt.show()
    
