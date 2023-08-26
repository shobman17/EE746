import numpy as np
import matplotlib.pyplot as plt

C = np.array([200, 130, 200])*1e-12
g_L = np.array([10, 18, 10])*1e-9
E_L = np.array([-70, -58, -58])*1e-3
V_T = np.array([-50, -50, -50])*1e-3
delta_T = np.array([2, 2, 2])*1e-3
a = np.array([2, 4, 2])*1e-9
tou_w = np.array([30, 150, 120])*1e-3
b = np.array([0, 120, 100])*1e-12
V_r = np.array([-58, -50, -46])*1e-3

def update(V_curr, U_curr, i, I_app, time_step):
    if V_curr > V_T[i]:
        V_curr = V_r[i]
        U_curr = U_curr + b[i]
    V_next = V_curr + (-g_L[i]*(V_curr - E_L[i]) + \
                       g_L[i]*delta_T[i]*np.e**((V_curr-V_T[i])/delta_T[i]) - \
                       U_curr + I_app)*time_step/C[i]
    U_next = U_curr + (a[i]*(V_curr-E_L[i]) - U_curr)*time_step/tou_w[i]
    return V_next, U_next

time_span = 0.5
num_steps = 5000
time_step = time_span/num_steps
time_trace = np.linspace(0+time_step,time_span,num=num_steps)

V_ss = {"RS":-60e-3, "IB":-60e-3, "CH":-60e-3}
U_ss = {"RS":0, "IB":0, "CH":0}
I_app = 0

V_trace, U_trace = {}, {}
V_trace["RS"] = []
V_trace["IB"] = []
V_trace["CH"] = []
U_trace["RS"] = []
U_trace["IB"] = []
U_trace["CH"] = []

fig, axe = plt.subplots(2, 1, figsize=(10,10))
types = ["RS", "IB", "CH"]
colors = ["pink", "violet", "cyan"]

V, U = {}, {}
V["RS"] = V_ss["RS"]
V["IB"] = V_ss["IB"]
V["CH"] = V_ss["CH"]
U["RS"] = U_ss["RS"]
U["IB"] = U_ss["IB"]
U["CH"] = U_ss["CH"]

for _ in range(num_steps):
    V_trace["RS"].append(V["RS"])
    V_trace["IB"].append(V["IB"])
    V_trace["CH"].append(V["CH"])
    U_trace["RS"].append(U["RS"])
    U_trace["IB"].append(U["IB"])
    U_trace["CH"].append(U["CH"])
    V["RS"], U["RS"] = update(V["RS"], U["RS"], 0, I_app, time_step)
    V["IB"], U["IB"] = update(V["IB"], U["IB"], 1, I_app, time_step)
    V["CH"], U["CH"] = update(V["CH"], U["CH"], 2, I_app, time_step)

for i in range(3):
    print(types[i], "Steady state U(t):", U_trace[types[i]][-1])
    print(types[i], "Steady state V(t):", V_trace[types[i]][-1])
    V_ss[types[i]] = V_trace[types[i]][-1]
    U_ss[types[i]] = U_trace[types[i]][-1]
    axe[0].plot(time_trace, U_trace[types[i]], color=colors[i], label=types[i])
    axe[1].plot(time_trace, V_trace[types[i]], color=colors[i], label=types[i])

axe[0].set_title("Recovery Variable U(t) (A)")
axe[0].grid()
axe[1].set_title("Membrane Potential V(t) (V)")
axe[1].grid()
plt.show()

fig, axe = plt.subplots(3, 2, figsize=(10,10))

I_app_label = ["250pA", "350pA", "450pA"]
for (j, I_app) in enumerate([250e-12, 350e-12, 450e-12]):

    V_trace, U_trace = {}, {}
    V_trace["RS"] = []
    V_trace["IB"] = []
    V_trace["CH"] = []
    U_trace["RS"] = []
    U_trace["IB"] = []
    U_trace["CH"] = []

    V, U = {}, {}
    V["RS"] = V_ss["RS"]
    V["IB"] = V_ss["IB"]
    V["CH"] = V_ss["CH"]
    U["RS"] = U_ss["RS"]
    U["IB"] = U_ss["IB"]
    U["CH"] = U_ss["CH"]

    for _ in range(num_steps):
        V_trace["RS"].append(V["RS"])
        V_trace["IB"].append(V["IB"])
        V_trace["CH"].append(V["CH"])
        U_trace["RS"].append(U["RS"])
        U_trace["IB"].append(U["IB"])
        U_trace["CH"].append(U["CH"])
        V["RS"], U["RS"] = update(V["RS"], U["RS"], 0, I_app, time_step)
        V["IB"], U["IB"] = update(V["IB"], U["IB"], 1, I_app, time_step)
        V["CH"], U["CH"] = update(V["CH"], U["CH"], 2, I_app, time_step)

    for i in range(3):
        axe[i][0].plot(time_trace, U_trace[types[i]], color=colors[j], label=I_app_label[j])
        axe[i][0].set_title("Recovery variable U(t) (A) in {type}".format(type=types[i]))
        axe[i][0].grid()

        axe[i][1].plot(time_trace, V_trace[types[i]], color=colors[j], label=I_app_label[j])
        axe[i][1].set_title("Membrane potential V(t) (V) in {type}".format(type=types[i]))
        axe[i][1].grid()

for i in range(3):
    axe[i][0].legend()
    axe[i][1].legend()
plt.tight_layout()
plt.show()