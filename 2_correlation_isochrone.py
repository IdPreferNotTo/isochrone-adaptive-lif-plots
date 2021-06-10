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


def intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    # segment_a = [[ax1, ay1], [ax2, ay2]], segment_b = [[bx1, by1], [bx2, by2]]

    # Check for overlap: If the largest x coordinate of segment a is smaller than the smallest x coordinate
    # of segment b then there can be no intersection. (same for y)
    if (max(ax1, ax2) < min(bx1, bx2)):
        return False
    if (max(ay1, ay2) < min(by1, by2)):
        return False
    # If the is a mutual interval calculate the x coordinate of that intersection point and check if it is in the interval.
    # Calculate fa(a) = Aa*x + Ba = y and fb(x) = Ab*x + Bb = y
    Aa = (ay1 - ay2) / (ax1 - ax2)  # slope of segment a
    Ab = (by1 - by2) / (bx1 - bx2)  # slope of segment b
    Ba = ay1 - Aa * ax1  # y intercept of segment a
    Bb = by1 - Ab * bx1  # y intercep of segment b
    x_intersection_a_b = (Bb - Ba) / (Aa - Ab)  # x coordinate of intersection point
    if (x_intersection_a_b < max(min(ax1, ax2), min(bx1, bx2)) or
            x_intersection_a_b > min(max(ax1, ax2), max(bx1, bx2))):
        return False
    else:
        return True


if __name__ == "__main__":
    # Initialize Adaptive leaky IF model
    mu: float = 5.0
    D: float = 1.00
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0
    v_thr: float = 1
    dt: float = 0.0001

    phase:float = 3*np.pi/2 #1.57. 3.14, 4.71

    alif = stochastic_adaptive_leaky_if(mu, tau_a, delta_a, D, v_res, v_thr, dt)
    print("Get limit cycle")
    v_lc, a_lc, ts = alif.limit_cycle()

    v = v_lc[50]
    a = a_lc[50]
    ISIs = []
    t = 0
    spikes = 0
    while spikes < 5000:
        v, a, spike = alif.forward_stochastic(v, a)
        t += dt
        if spike == True:
            print(spikes)
            spikes += 1
            ISIs.append(t)
            t = 0

    v = v_lc[50]
    a = a_lc[50]
    ISIs_iso = []
    V_iso_pass = []
    A_iso_pass = []
    t = 0
    spikes = 0
    home = os.path.expanduser("~")
    isochrone = np.loadtxt(home + "/Data/isochrones/isochrones_file_mu5.00_{:.2f}.dat".format(phase))

    ref_steps = 0
    while spikes < 5000:
        # Let the point (v, a) evolve until it hits the presumed isochrone
        v_before = v
        a_before = a
        v, a, spike = alif.forward_stochastic(v, a)
        if spike == False:
            v_after = v
            a_after = a
        else:
            v_after = 1.
            a_after = a - delta_a
            has_spiked = True
        t += dt
        ref_steps += 1
        # For every segment of the isochrone check if this segment and the segment that describes
        # the change of v, a ((v_tmp, a_tmp), (v, a)) intersect.
        for (v_iso0, a_iso0), (v_iso1, a_iso1) in zip(isochrone[:-1], isochrone[1:]):
            if abs(v_iso0 - v_iso1) > 0.5:
                continue
            passed_isochrone = intersect(v_before, a_before, v_after, a_after, v_iso0, a_iso0, v_iso1, a_iso1)
            if passed_isochrone and ref_steps > 2000:
                print(spikes)
                ref_steps = 0
                spikes += 1
                ISIs_iso.append(t)
                V_iso_pass.append(v_after)
                A_iso_pass.append(a_after)
                t = 0

    file_str = home + "/Data/isochrones/ISI_thr_D{:.2f}_{:.2f}.dat".format(D, phase)
    with open(file_str, "w") as file:
        for isi in ISIs:
            file.write("{:.8f} \n".format(isi))

    file_str = home + "/Data/isochrones/ISI_iso_D{:.2f}_{:.2f}.dat".format(D, phase)
    with open(file_str, "w") as file:
        for isi, v, a in zip(ISIs_iso, V_iso_pass, A_iso_pass):
            file.write("{:.8f} {:.4f} {:.4f}\n".format(isi, v, a))

