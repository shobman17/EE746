import numpy as np
import matplotlib.pyplot as plt
import math

class HH_neuron:
    def __init__(self, V = 0, n = 0, m = 0, h = 0):
        self.C = 1e-6
        self.ENa = 50e-3
        self.EK = -77e-3
        self.El = -55e-3
        self.gNA = 120e-3
        self.gK = 36e-3
        self.gl = 0.3e-3
        self.V = V # Need to tune this. NEEDS TO BE GIVEN IN MILLIVOLTS
        self.n = n # Need to tune this
        self.m = m # Need to tune this
        self.h = h # Need to tune this
        self.dV = 0
        self.dn = 0
        self.dm = 0
        self.dh = 0
    
    def calculate_difference(self, I_ext):
        """
        Calculates the difference rate and updates the class variables accordingly
        """

        epsilon = 1e-5 # prevent divide by zero errors
        # V is multiplied by 1000 to convert it into mV
        self.V *= 1e3
        a_n = (0.01*(self.V + 55))/(1 - math.exp(-(55+self.V)/10) + epsilon)
        b_n = 0.125*math.exp(-(self.V + 65)/80)
        a_m = (0.1*(self.V + 40))/(1 - math.exp(-(40+self.V)/10) + epsilon)
        b_m = 4*math.exp(-0.0556*(self.V + 65))
        a_h = 0.07*math.exp(-0.05*(self.V + 65))
        b_h = 1/(1 + math.exp(-0.1*(self.V + 35)) + epsilon)
        self.V *= 1e-3

        self.dn = a_n*(1-self.n) - b_n*self.n
        self.dm = a_m*(1-self.m) - b_m*self.m
        self.dh = a_h*(1-self.h) - b_h*self.h

        self.dV = (I_ext - self.iNa - self.iK - self.il)/self.C
        
    def update_temp(self, t):
        """
        Update the value of V and other variables crudely
        Done to implement Runge-Kutta  
        """

        self.V = self.V + self.dV*t
        self.n = self.n + self.dn*t
        self.m = self.m + self.dm*t
        self.h = self.h + self.dh*t

    def update(self,I,t):
        """
        Update all variables according to Runge-Kutta fourth order
        """

        og_state = [self.V, self.n, self.m, self.h]

        self.calculate_difference(I)
        k1 = np.array([self.dV, self.dn, self.dm, self.dh])

        self.update_temp(t/2)
        self.calculate_difference(I) # y = y_og + k1*t/2 here. Should we update current here too?
        k2 = np.array([self.dV, self.dn, self.dm, self.dh])

        self.V, self.n, self.m, self.h = og_state
        self.update_temp(t/2)
        self.calculate_difference(I) # y = y_og + k2*t/2 here
        k3 = np.array([self.dV, self.dn, self.dm, self.dh])

        self.V, self.n, self.m, self.h = og_state
        self.update_temp(t)
        self.calculate_difference(I) # y = y_og + k3*t here
        k4 = np.array([self.dV, self.dn, self.dm, self.dh])

        self.V, self.n, self.m, self.h = og_state
        weighted_avg = (k1/6 + k2/3 + k3/3 + k4/6).tolist()
        self.dV = weighted_avg[0]
        self.dn = weighted_avg[1]
        self.dm = weighted_avg[2]
        self.dh = weighted_avg[3]
        self.update_temp(t)
    
    @property
    def iNa(self):
        return self.gNA*self.h*(self.m**3)*(self.V - self.ENa)
    
    @property
    def iK(self):
        return self.gK*(self.n**4)*(self.V - self.EK)
    
    @property
    def il(self):
        return self.gl*(self.V - self.El)
    
if __name__ == "__main__":

    neuron = HH_neuron(V = -55e-3) # apparently this works
    V_trace = []
    T = 30e-3
    deltaT = 0.01e-3
    I0 = 15e-6
    time = np.arange(0,15*T, deltaT)
    Iext = [I0 if (t < 3*T and t >= 2*T) else 0 for t in time]

    for t in range(len(time)):
        V_trace.append(neuron.V)
        #neuron.calculate_difference(Iext[0])
        #neuron.update_temp(deltaT)
        neuron.update(Iext[t], deltaT)

    
    print(V_trace[-1])
    print(V_trace[8800])
    plt.plot(time, V_trace, label = "Membrane potential")
    plt.legend()
    plt.show()

    
