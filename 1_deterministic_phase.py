import os
from math import atan2, cos, sin, fmod, isclose
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



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


class bob_the_isochrone_builder:
    def __init__(self, model, isi, v0, a0, stepwidth):
        self.model = model
        self.isi = isi
        self.v_init = v0
        self.a_init = a0
        self.r = stepwidth
        self.isochrone = [[v0, a0]]


    def get_isochrone(self):
        return self.isochrone


    def next_point_right(self, list):
        if len(list) >= 2:
            v1, a1 = list[-1]
            v0, a0 = list[-2]
            dv = v1 - v0
            da = a1 - a0

            da = self.r * (da / abs(dv))
            dv = self.r * (dv / abs(dv))

            v_candidate = v1 + dv
            a_candidate = a1 + da
        else:
            v1, a1 = list[0]
            dif = fmod(v1, 0.05)
            v_candidate = v1 + 0.05 - dif
            a_candidate = a1
        return v_candidate, a_candidate


    def next_point_left(self, list):
        if len(list) >= 2:
            v1, a1 = list[0]
            v0, a0 = list[1]
            dv = v1 - v0
            da = a1 - a0

            da = self.r * (da / abs(dv))
            dv = self.r * (dv / abs(dv))

            v_candidate = v1 + dv
            a_candidate = a1 + da
        else:
            v1, a1 = list[0]
            dif = fmod(v1, 0.05)
            if dif <= 0:
                dif = 0.05
            v_candidate = v1 - dif
            a_candidate = a1
        return v_candidate, a_candidate


    def return_time_point_isochrone(self, v, a):
        t = 0
        ref_steps = 0
        # Let the point (v, a) evolve until it hits the presumed isochrone
        while True:
            v_before = v
            a_before = a
            v, a, spike = self.model.forward(v, a)
            if spike == False:
                v_after = v
                a_after = a
            else:
                v_after = v + 1.
                a_after = a - delta_a
            t += dt
            ref_steps += 1
            # For every segment of the isochrone check if this segment and the segment that describes
            # the change of v, a ((v_tmp, a_tmp), (v, a)) intersect.
            for (v_iso_before, a_iso_before), (v_iso_after, a_iso_after) in zip(self.isochrone[:-1], self.isochrone[1:]):
                if abs(v_iso_before - v_iso_after) > 0.5:
                    continue
                passed_isochrone = intersect(v_before, a_before, v_after, a_after, v_iso_before, a_iso_before,
                                             v_iso_after, a_iso_after)
                if passed_isochrone and (ref_steps > 1):
                    return t


    def construct_isochrone_branch_left_from_point(self, v0, a0, vres, vmin, insert_point):
        # Add initial points (v0, a0) to left part of isochrone
        print("Start left branch of isochrone")
        isochrone_left = [[v0, a0]]
        while True:
            v_candidate, a_candidate = self.next_point_left(isochrone_left)
            if v_candidate > vmin - self.r/2.:
                # Insert (v0, a0) to the general isochrone this is important because i calculate the return time to this
                # isochrone. Note that (v0, a0) is added at the beginning of isochrone
                self.isochrone.insert(insert_point, [v_candidate, a_candidate])
                on_isochrone = False
                # Calculate return time from (v0, a0) to isochrone and adjust (v0, a0) until |T - ISI| / ISI < 0.0005
                while not on_isochrone:
                    T = self.return_time_point_isochrone(v_candidate, a_candidate)
                    if abs(T - self.isi) / self.isi > 0.001:
                        if T > self.isi:
                            print("ISI too long, {:.4f}, v={:.4f}, a={:.4f}".format(abs(T - self.isi) / self.isi, v_candidate, a_candidate))
                            a_candidate = a_candidate - 0.0005
                        else:
                            print("ISI too short, {:.4f}, v={:.4f}, a={:.4f}".format(abs(T - self.isi) / self.isi, v_candidate, a_candidate))
                            a_candidate = a_candidate + 0.0005
                        self.isochrone[insert_point] = [v_candidate, a_candidate]
                    else:
                        # Return time on point
                        print("on point")
                        on_isochrone = True
                        if isclose(v_candidate, vres, abs_tol=0.001):
                            self.start_point_lower_branch = [1, a_candidate - self.model.get_deltaa()]
                        isochrone_left.insert(0, [v_candidate, a_candidate])
            else:
                return isochrone_left


    def construct_isochrone_branch_right_from_point(self, v0, a0, vmax):
        print("Start right branch of isochrone")
        isochrone_right = [[v0, a0]]
        while True:
            v_candidate, a_candidate = self.next_point_right(isochrone_right)
            if v_candidate < vmax + self.r/2.:
                # The if statement is a little ugly but ensures that v_candidate = vmax is still captured. (note float-aronis)
                self.isochrone.append([v_candidate, a_candidate])
                on_isochrone = False
                while not on_isochrone:
                    T = self.return_time_point_isochrone(v_candidate, a_candidate)
                    if abs(T - self.isi) / self.isi > 0.001:
                        if T > self.isi:
                            print("ISI too long, {:.4f}, v={:.4f}, a={:.4f}".format(abs(T - self.isi) / self.isi, v_candidate, a_candidate))
                            a_candidate = a_candidate - 0.0005
                        else:
                            print("ISI too short, {:.4f}, v={:.4f}, a={:.4f}".format(abs(T - self.isi) / self.isi, v_candidate, a_candidate))
                            a_candidate = a_candidate + 0.0005
                        self.isochrone[-1] = [v_candidate, a_candidate]
                    else:
                        # Return time on point
                        print("on point")
                        on_isochrone = True
                        if isclose(v_candidate, vmax, abs_tol=0.001):
                            self.start_point_upper_branch = [0, a_candidate + self.model.get_deltaa()]
                        isochrone_right.append([v_candidate, a_candidate])
            else:
                return isochrone_right


class adaptive_leaky_if():
    def __init__(self, mu, tau_a, delta_a, v_r, v_t, dt):
        self.mu = mu
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.v_r = v_r
        self.v_t = v_t
        self.dt = dt


    def get_deltaa(self):
        return self.delta_a

    def forward(self, v, a):
        dv = (self.mu - v - a) * self.dt
        da = (-a / self.tau_a) * self.dt
        v += dv
        a += da
        spike = False
        if v > self.v_t:
            dv = v - self.v_t
            v = self.v_r + dv
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
        v -= (self.mu - v - a) * self.dt
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


    def period(self):
        v: float = 0
        a: float = 0
        spikes: int = 0
        while (spikes < 100):
            v, a, spike = self.forward(v, a)
            if spike:
                spikes += 1
        t = 0
        spikes = 0
        while (spikes < 100):
            v, a, spike = self.forward(v, a)
            t += dt
            if spike:
                spikes += 1
        return t / spikes


if __name__ == "__main__":
    # Initialize Adaptive leaky integrate-and-fire (LIF) model
    mu: float = 5.0
    tau_a: float = 2.0
    delta_a: float = 1.0
    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.0001
    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)

    # Find ISI and Limit Cycle of the adaptive LIF
    isi = alif.period()
    print(isi)
    v_lc, a_lc, ts = alif.limit_cycle()

    # Use point of the Limit Cycle with certain phase as initial point to construct isochrone that belongs to that phase
    phase = 2*np.pi/2
    idx_phase = int(len(v_lc)*(phase)/(2*np.pi))
    v_lc0 = v_lc[idx_phase]
    a_lc0 = a_lc[idx_phase]
    print(v_lc0, a_lc0)

    # Initialize isochrone builder - Idea: start at a certain point (v0, a0) and construct the left (from v0 to vmin)
    # and right (from v0 to vmax) part of the isochrone relative to (v0, a0). Simultaneously the whole isochrone is saved
    # internally. More important: he isochrone has to be constructed in a certain order. This is so because it may happen
    # that a trajectory starting at some point (v_candidate, a_candidate) that should cross the isochrone at (v_iso, a_iso)
    # wont do so simple because (v_iso, a_iso) is not constructed yet.

    # Essentially the Isochrone should be constructed from bottom (small a) to top (large a) which is not possible because
    # the starting point lies on the LC.
    stepwidth: float = 0.05
    iso = bob_the_isochrone_builder(alif, isi, v_lc0, a_lc0, stepwidth)
    vmin = -1
    vres = 0
    vmax = 1

    iso_left_b1 = iso.construct_isochrone_branch_left_from_point(v_lc0, a_lc0, vres, 0, insert_point = 0)
    iso_right_b1 = iso.construct_isochrone_branch_right_from_point(v_lc0, a_lc0, vmax)

    v0_ub, a0_ub = iso.start_point_upper_branch
    v0_lb, a0_lb = iso.start_point_lower_branch

    # Initial values are not automatically added to the whole isochrone. This is convenient most of the time but
    # requires to manually add this point if wanted.
    iso.isochrone.insert(0, [v0_lb, a0_lb])
    iso_left_b0 = iso.construct_isochrone_branch_left_from_point(v0_lb, a0_lb, vres, 0, insert_point=0)

    v0_lb, a0_lb = iso.start_point_lower_branch

    v_b1_1, a_b1_1 = iso_left_b1[0]
    l = len(iso_left_b0)
    iso_left_left_b1 = iso.construct_isochrone_branch_left_from_point(v_b1_1, a_b1_1, vres, vmin, insert_point=l)

    l = iso.get_isochrone()
    iso_left_b2 = iso.construct_isochrone_branch_left_from_point(v0_ub, a0_ub, vres, vmin, insert_point = len(l))


    iso.isochrone.append([v0_ub, a0_ub])
    iso_right_b2 = iso.construct_isochrone_branch_right_from_point(v0_ub, a0_ub, vmax)


    fig = plt.figure(tight_layout=True, figsize=(6, 9 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")
    ax.plot(v_lc, a_lc, c="k")
    isochrone = iso.get_isochrone()
    ax.plot([x[0] for x in isochrone], [x[1] for x in isochrone])

    # Check if Isochrone is good
    v, a = isochrone[-1]
    for i in range(5):
        ax.scatter(v, a, c="C3", zorder=4)
        v, a = alif.forward_for_T(v, a, isi)

    v, a = isochrone[0]
    for i in range(6):
        ax.scatter(v, a, c="C3", zorder=4)
        v, a = alif.forward_for_T(v, a, isi)

    # Write Isochrone to file
    home = os.path.expanduser("~")

    file_str = home + "/Data/isochrones/isochrones_file_mu{:.2f}_{:.2f}.dat".format(mu, phase)
    with open(file_str, "w") as f:
        for v, a in isochrone:
            f.write("{:.4f} {:.4f} \n".format(v, a))
    #plt.savefig(home + "/Data/isochrones/Isochrone_mu{:.2f}_{:.2f}.pdf".format(mu, phase))
    plt.show()
