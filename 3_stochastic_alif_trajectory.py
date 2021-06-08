import os
import numpy as np
import matplotlib.pyplot as plt

class stochastic_adaptive_leaky_if():
    def __init__(self, mu, tau_a, delta_a, D, v_r, v_t, dt):
        self.mu = mu
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.D = D
        self.v_r = v_r
        self.v_t = v_t
        self.dt = dt

    def forward(self, v, a):
        v += (self.mu - v - a) * self.dt
        a += (-a / self.tau_a) * self.dt
        spike = False
        if v > self.v_t:
            v = self.v_r
            a += self.delta_a
            spike = True
        return v, a, spike

    def forward_stochastic(self, v, a):
        xi = np.random.normal(0, 1)
        v += (self.mu - v - a) * self.dt + np.sqrt(2*D*self.dt)*xi
        a += (-a / self.tau_a) * self.dt
        spike = False
        if v > self.v_t:
            v = self.v_r
            a += self.delta_a
            spike = True
        return v, a, spike

    def forward_for_T(self, v, a, T):
        t = 0
        while (t < T):
            v, a, spike = self.forward(v, a)
            t += self.dt
        return v, a

    def limit_cycle(self):
        v: float = 0
        a: float = 0
        spikes: int = 0
        while (spikes < 100):
            v, a, spike = self.forward(v, a)
            if spike:
                spikes += 1
        t = 0
        v_s = []
        a_s = []
        t_s = []
        while (True):
            v_s.append(v)
            a_s.append(a)
            t_s.append(t)
            v, a, spike = self.forward(v, a)
            if spike:
                return [v_s, a_s, t_s]


if __name__ == "__main__":
    # Initialize Adaptive leaky IF model
    mu: float = 5.0
    D: float = 0.01
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0
    v_thr: float = 1
    dt: float = 0.0001

    phase:float = 2*np.pi/2 #1.57. 3.14, 4.71

    alif = stochastic_adaptive_leaky_if(mu, tau_a, delta_a, D, v_res, v_thr, dt)

    home = os.path.expanduser("~")
    isochrone = np.loadtxt(home + "/Data/isochrones/isochrones_file_mu5.00_3.14.dat")

    volts = []
    adaps = []
    v, a  = isochrone[40]
    for i in range(10000):
        volts.append(v)
        adaps.append(a)
        v, a, spike = alif.forward_stochastic(v, a)

    file_str = home + "/Data/isochrones/stochastic_trajectory_D{:.2f}_{:.2f}.dat".format(D, phase)
    with open(file_str, "w") as file:
        for (v, a) in zip(volts, adaps):
            file.write("{:.4f} {:.4f} \n".format(v, a))