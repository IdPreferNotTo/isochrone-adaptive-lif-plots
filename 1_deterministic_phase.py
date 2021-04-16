import os
from math import atan2, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def expand_isochrone(isochrone):
    # Returns (v, a, bool v in [0, 1]?). (v,a) is a candidate for a pooint on the isochrone. the third value indicates
    # whether v is in the interval [0, 1].
    v1, a1 = isochrone[-1]
    v0, a0 = isochrone[-2]
    dv = v1 - v0
    da = a1 - a0
    dx = np.sqrt(dv ** 2 + da ** 2)
    dv = dv / dx
    da = da / dx
    v_candidate = v1 + 0.02 * dv
    a_candidate = a1 + 0.02 * da
    if v_candidate > 1. or v_candidate < 0.:
        return v_candidate, a_candidate, False
    else:
        return v_candidate, a_candidate, True


def adjust_point_to_match_isochrone(v_new, a_new, v_fix, a_fix, t, isi):
    dv = v_new - v_fix
    da = a_new - a_fix
    r = np.sqrt(dv ** 2 + da ** 2)
    phi = atan2(da, dv)  # return value of atan2 \in [-pi, pi]
    if t > isi:
        print("but ISI too long")
        # Return time too large
        v = v_fix + r * cos(phi - 0.02 * np.pi)
        a = a_fix + r * sin(phi - 0.02 * np.pi)
        return v, a
    elif t < isi:
        print("but ISI too short")
        # Return time too small
        v = a_fix + r * cos(phi + 0.02 * np.pi)
        a = a_fix + r * sin(phi + 0.02 * np.pi)
        return v, a


def first_passage_time_last_point_isochrone_to_isochrone(isochrone, dt):
    t = 0
    v, a = isochrone[-1]
    has_spiked = False
    # Let the point (v, a) evolve until it hits the presumed isochrone
    while True:
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
                return t


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


class adaptive_leaky_if():
    def __init__(self, mu, gamma, tau_a, delta_a, v_r, v_t, dt):
        self.mu = mu
        self.gamma = gamma
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.v_r = v_r
        self.v_t = v_t
        self.dt = dt

    def forward(self, v, a):
        v += (self.mu - self.gamma * v - a) * self.dt
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
            v += (self.mu - self.gamma * v - a) * self.dt
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

            v += (self.mu - self.gamma * v - a) * self.dt
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
            v += (self.mu - self.gamma * v - a) * self.dt
            a += (-a / self.tau_a) * self.dt
            t += self.dt
            if v > self.v_t:
                v = self.v_r
                a += self.delta_a
                spikes += 1

        t = 0
        spikes = 0
        while (spikes < 1000):
            v += (self.mu - self.gamma * v - a) * self.dt
            a += (-a / self.tau_a) * self.dt
            t += self.dt
            if v > self.v_t:
                v = self.v_r
                a += self.delta_a
                spikes += 1
        return t / spikes


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
    alif = adaptive_leaky_if(mu, gamma, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    v_lc, a_lc, ts = alif.limit_cycle()

    # Initialize Plot
    fig = plt.figure(tight_layout=True, figsize=(6, 9 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    axins = inset_axes(ax, width="30%", height="30%", loc=4)
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")
    ax.plot(v_lc, a_lc, c="k")
    axins.plot(v_lc, a_lc, c="k")
    # Chose any point in phasespace and integrate forward until close to the LC
    v = 0.1  # dont chose 0.00
    a = 2.8  # dont chose 0.00
    ax.scatter(v, a, ec="C3", fc="w", s=100, zorder=5)
    delta_a = 1
    delta_v = 1
    while delta_v > 0.005 and delta_a > 0.005:
        v_tmp = v
        a_tmp = a
        v, a = alif.forward_for_T(v, a, isi)
        delta_v = abs(v - v_tmp) / v_tmp
        delta_a = abs(a - a_tmp) / a_tmp

    # Save the point that is close to the limit cycle
    v_left = v
    a_left = a
    # Let this point evolve for a few more periods to get ON the limit cycle
    for i in range(10):
        v, a = alif.forward_for_T(v, a, isi)
    v_lc = v
    a_lc = a
    # Interpolate v_right, a_right, so that (v_left, a_left), (v_lc, a_lc) and (v_right, a_right) are on the same line
    v_right = v_lc + (v_lc - v_left)
    a_right = a_lc + (a_lc - a_left)

    # Split the isochrone into two parts. Below/left and ontop/right of the LC.
    isochrone_right = [[v_lc, a_lc], [v_right, a_right]]
    isochrone_left = [[v_lc, a_lc], [v_left, a_left]]

    on_isochrone = False
    while not on_isochrone:
        t = first_passage_time_last_point_isochrone_to_isochrone(isochrone_right, dt)
        if abs(t - isi) > 0.001 * isi:
            v1, a1 = isochrone_right[-1]
            v0, a0 = isochrone_right[-2]
            dv = v1 - v0
            da = a1 - a0
            r = np.sqrt(dv ** 2 + da ** 2)
            phi = atan2(da, dv)  # return value of atan2 \in [-pi, pi]
            if t > isi:
                print("but ISI to long")
                # Return time too large
                v = v0 + r * cos(phi - 0.02 * np.pi)
                a = a0 + r * sin(phi - 0.02 * np.pi)
            elif t < isi:
                print("but ISI too short")
                # Return time too small
                v = v0 + r * cos(phi + 0.02 * np.pi)
                a = a0 + r * sin(phi + 0.02 * np.pi)
            isochrone_right[-1] = [v, a]
        else:
            # Return time on point
            print("and its in point")
            on_isochrone = True

    on_isochrone = False
    while not on_isochrone:
        t = first_passage_time_last_point_isochrone_to_isochrone(isochrone_left, dt)
        if abs(t - isi) > 0.001 * isi:
            v1, a1 = isochrone_left[-1]
            v0, a0 = isochrone_left[-2]
            dv = v1 - v0
            da = a1 - a0
            r = np.sqrt(dv ** 2 + da ** 2)
            phi = atan2(da, dv)  # return value of atan2 \in [-pi, pi]
            if t > isi:
                print("but ISI to long")
                # Return time too large
                v = v0 + r * cos(phi + 0.02 * np.pi)
                a = a0 + r * sin(phi + 0.02 * np.pi)
            elif t < isi:
                print("but ISI too short")
                # Return time too small
                v = v0 + r * cos(phi - 0.02 * np.pi)
                a = a0 + r * sin(phi - 0.02 * np.pi)
            isochrone_left[-1] = [v, a]
            t = 0
        else:
            # Return time on point
            print("and its in point")
            on_isochrone = True

    # Plot (v_left, a_left), (v_lc, a_lc) and (v_right, a_right)
    ax.scatter(v_left, a_left, c="C0", zorder=5)
    ax.scatter(v_lc, a_lc, c="C0", zorder=5)
    ax.scatter(v_right, a_right, c="C0", zorder=5)

    axins.scatter(v_left, a_left, c="C0", zorder=5)
    axins.scatter(v_lc, a_lc, c="C0", zorder=5)
    axins.scatter(v_right, a_right, c="C0", zorder=5)
    axins.set_xlim([0.9 * v_lc, 1.1 * v_lc])
    axins.set_ylim([0.99 * a_lc, 1.01 * a_lc])

    # Check if (v_right, a_right) is actually on the isochrone, i.e. when the right part of the
    # presumed isochrone is passed.
    while True:
        t = 0
        on_isochrone = False
        # Find a candidate for a new point on the isochrone
        v_candidate, a_candidate, v_in_range = expand_isochrone(isochrone_right)
        # If is v-component is outside [0, 1] stop the loop
        if not v_in_range:
            break
        isochrone_right.append([v_candidate, a_candidate])
        # Check if return T time from this (v,a) back to the isochrone is T = ISI
        while not on_isochrone:
            t = first_passage_time_last_point_isochrone_to_isochrone(isochrone_right, dt)
            if abs(t - isi) > 0.001 * isi:
                v1, a1 = isochrone_right[-1]
                v0, a0 = isochrone_right[-2]
                dv = v1 - v0
                da = a1 - a0
                r = np.sqrt(dv ** 2 + da ** 2)
                phi = atan2(da, dv)  # return value of atan2 \in [-pi, pi]
                if t > isi:
                    print("but ISI to long")
                    # Return time too large
                    v = v0 + r * cos(phi - 0.02 * np.pi)
                    a = a0 + r * sin(phi - 0.02 * np.pi)
                elif t < isi:
                    print("but ISI too short")
                    # Return time too small
                    v = v0 + r * cos(phi + 0.02 * np.pi)
                    a = a0 + r * sin(phi + 0.02 * np.pi)
                isochrone_right[-1] = [v, a]
                t = 0
            else:
                # Return time on point
                print("and its in point")
                ax.scatter(v_candidate, a_candidate, c="k")
                axins.scatter(v_candidate, a_candidate, c="k")
                on_isochrone = True

    # Check if Isochrone is good
    v, a = isochrone_right[-1]
    for i in range(6):
        ax.scatter(v, a, c="C3", zorder=6)
        axins.scatter(v, a, c="C3")
        v, a = alif.forward_for_T(v, a, isi)
    #ax.plot([x[0] for x in isochrone_right], [x[1] for x in isochrone_right])

    # Turn the left part of the Isochrone
    print("SWITCHING ISOCHRONES")

    while True:
        t = 0
        on_isochrone = False
        v_candidate, a_candidate, v_in_range = expand_isochrone(isochrone_left)
        if not v_in_range:
            break
        isochrone_left.append([v_candidate, a_candidate])

        # Let the point (v_cand, a_cand) evolve until it hits the presumed isochrone
        while not on_isochrone:
            t = first_passage_time_last_point_isochrone_to_isochrone(isochrone_left, dt)
            if abs(t - isi) > 0.001 * isi:
                v1, a1 = isochrone_left[-1]
                v0, a0 = isochrone_left[-2]
                dv = v1 - v0
                da = a1 - a0
                r = np.sqrt(dv ** 2 + da ** 2)
                phi = atan2(da, dv)  # return value of atan2 \in [-pi, pi]
                if t > isi:
                    print("but ISI to long")
                    # Return time too large
                    v = v0 + r * cos(phi + 0.02 * np.pi)
                    a = a0 + r * sin(phi + 0.02 * np.pi)
                    isochrone_left[-1] = [v, a]
                    t = 0
                elif t < isi:
                    print("but ISI too short")
                    # Return time too small
                    v = v0 + r * cos(phi - 0.02 * np.pi)
                    a = a0 + r * sin(phi - 0.02 * np.pi)
                    isochrone_left[-1] = [v, a]
                    t = 0
            else:
                # Return time on point
                print("and its in point")
                ax.scatter(v_candidate, a_candidate, c="k")
                axins.scatter(v_candidate, a_candidate, c="k")
                on_isochrone = True

    v, a = isochrone_left[-1]
    for i in range(6):
        ax.scatter(v, a, c="C3", zorder=6)
        axins.scatter(v, a, c="C3")
        v, a = alif.forward_for_T(v, a, isi)
    #ax.plot([x[0] for x in isochrone_left], [x[1] for x in isochrone_left])

    isochrone = isochrone_left + isochrone_right
    ax.plot([x[0] for x in isochrone], [x[1] for x in isochrone])
    home = os.path.expanduser("~")
    ax.grid(True, ls="--")
    plt.savefig(home + "/Data/isochrones/isochrone_double_check.pdf", transparent=True)
    plt.show()
