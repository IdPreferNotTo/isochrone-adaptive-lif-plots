import os
import numpy as np
import matplotlib.pyplot as plt

class stochastic_adaptive_leaky_if():
    def __init__(self, mu, gamma, tau_a, delta_a, D, v_r, v_t, dt):
        self.mu = mu
        self.gamma = gamma
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.D = D
        self.v_r = v_r
        self.v_t = v_t
        self.dt = dt

    def forward(self, v, a):
        xi = np.random.normal(0, 1)
        v += (self.mu - self.gamma * v - a) * self.dt + np.sqrt(2*D*dt)*xi
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

    def backward(self, v, a):
        v -= (self.mu - self.gamma * v - a) * self.dt
        a -= (-a / self.tau_a) * self.dt
        reset = False
        if v < self.v_r:
            v = self.v_t
            a -= self.delta_a
            reset = True
        return v, a, reset

    def backward_for_T(self, v, a, T):
        t = T
        while (t > 0):
            v, a, spike = self.backward(v, a)
            t -= self.dt
            if a < 0 or v > v_thr:
                return None, None
        return v, a

    def limit_cycle(self):
        t: float = 0
        v: float = 0
        a: float = 0

        spikes: int = 0
        while (spikes < 100):
            xi = np.random.normal(0, 1)
            v += (self.mu - self.gamma * v - a) * self.dt + np.sqrt(2*D*dt)*xi
            a += (-a / self.tau_a) * self.dt
            t += self.dt
            if v > self.v_t:
                v = self.v_r
                a += self.delta_a
                spikes += 1

        t = 0
        v_s = []
        a_s = []
        t_s = []
        while (True):
            v_s.append(v)
            a_s.append(a)
            t_s.append(t)

            xi = np.random.normal(0, 1)
            v += (self.mu - self.gamma * v - a) * self.dt + np.sqrt(2 * D * dt) * xi
            a += (-a / self.tau_a) * self.dt
            t += self.dt
            if v > self.v_t:
                v_s.append(v)
                a_s.append(a)
                t_s.append(t)
                return [v_s, a_s, t_s]

    def period(self):
        t: float = 0
        v: float = 0
        a: float = 0

        spikes: int = 0
        while (spikes < 100):
            xi = np.random.normal(0, 1)
            v += (self.mu - self.gamma * v - a) * self.dt + np.sqrt(2 * D * dt) * xi
            a += (-a / self.tau_a) * self.dt
            t += self.dt
            if v > self.v_t:
                v = self.v_r
                a += self.delta_a
                spikes += 1

        t = 0
        spikes = 0
        while (spikes < 1000):
            xi = np.random.normal(0, 1)
            v += (self.mu - self.gamma * v - a) * self.dt + np.sqrt(2 * D * dt) * xi
            a += (-a / self.tau_a) * self.dt
            t += self.dt
            if v > self.v_t:
                v = self.v_r
                a += self.delta_a
                spikes += 1
        return t / spikes


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
    gamma: float = 1.0
    D: float = 0.1
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0
    v_thr: float = 1
    dt: float = 0.0001

    alif = stochastic_adaptive_leaky_if(mu, gamma, tau_a, delta_a, D, v_res, v_thr, dt)
    isi = alif.period()
    v_lc, a_lc, ts = alif.limit_cycle()

    v = v_lc[50]
    a = a_lc[50]
    ISIs = []
    t = 0
    spikes = 0
    while spikes < 5_000:
        v, a, spike = alif.forward(v, a)
        t += dt
        if spike == True:
            print(spikes)
            spikes += 1
            ISIs.append(t)
            t = 0

    v = v_lc[50]
    a = a_lc[50]
    ISIs_iso = []
    t = 0
    spikes = 0
    home = os.path.expanduser("~")
    isochrone = np.loadtxt(home + "/Data/isochrones/isochrones_file.dat")
    has_spiked = False
    while spikes < 5_000:
        # Let the point (v, a) evolve until it hits the presumed isochrone
        v_before = v
        a_before = a
        v, a, spike = alif.forward(v, a)
        if spike == False:
            v_after = v
            a_after = a
        else:
            v_after = 1.
            a_after = a - delta_a
            has_spiked = True
        t += dt
        # For every segment of the isochrone check if this segment and the segment that describes
        # the change of v, a ((v_tmp, a_tmp), (v, a)) intersect.
        for (v_iso0, a_iso0), (v_iso1, a_iso1) in zip(isochrone[:-1], isochrone[1:]):
            passed_isochrone = intersect(v_before, a_before, v_after, a_after, v_iso0, a_iso0, v_iso1, a_iso1)
            if passed_isochrone and has_spiked:
                print(spikes)
                has_spiked = False
                spiked = False
                spikes += 1
                ISIs_iso.append(t)
                t = 0

    file_str = home + "/Data/isochrones/ISI_thr.dat"
    with open(file_str, "w") as file:
        for isi in ISIs:
            file.write("{:.8f} \n".format(isi))

    file_str = home + "/Data/isochrones/ISI_iso.dat"
    with open(file_str, "w") as file:
        for isi in ISIs_iso:
            file.write("{:.8f} \n".format(isi))

