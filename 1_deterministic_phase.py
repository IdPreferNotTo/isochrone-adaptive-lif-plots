import os
from math import atan2, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def expand_isochrone(isochrone):
    v_in_range = True
    while v_in_range:
        on_isochrone = False
        # Check if return T time from this (v,a) back to the isochrone is T = ISI
        while not on_isochrone:
            t = first_passage_time_last_point_isochrone_to_isochrone(isochrone, dt)
            if abs(t - isi) > 0.001 * isi:
                v_new, a_new = isochrone[-1]
                v_fix, a_fix = isochrone[-2]
                if t > isi:
                    print("ISI too long")
                    v, a = move_to_reduce_return_time(v_new, a_new, v_fix, a_fix)
                else:
                    print("ISI too short")
                    v, a = move_to_increase_return_time(v_new, a_new, v_fix, a_fix)
                isochrone[-1] = [v, a]
            else:
                # Return time on point
                print("on point")
                on_isochrone = True
                # Find a candidate for a new point on the isochrone
                v_candidate, a_candidate, v_in_range = expand_isochrone_by_one_point(isochrone)
                # If is v-component is outside [0, 1] stop the loop
                isochrone.append([v_candidate, a_candidate])
    return isochrone


def move_to_reduce_return_time(v0, a0, v_fix , a_fix):
    # Consider two points (v, a) and (v_fix, a_fix). The latter one is know to lie on the isochrone. Define (v_fix, a_fix)
    # as the center of a circle so that the point (v, a) can be described by radius and angle. Change the angle of (v, a)
    # so that the return time to the isochrone is reduced.

    # The flow of (v_fix, a_fix) will pass the circle with radius 0.02 at some angle. Find that angle. Moving the
    # point (v,a) closer to the angle phi_fix will eventually reduces the return time.
    dx = 0.
    v = v_fix
    a = a_fix
    while dx < 0.02:
        v, a, spike = alif.forward(v, a)
        if spike == False:
            dx = np.sqrt((v-v_fix)**2 + (a-a_fix)**2)
        if spike == True:
            dx = np.sqrt((v + 1 - v_fix)**2 + (a - delta_a - a_fix)**2)
    v_fix2 = v
    a_fix2 = a
    dv_fix = v_fix2 - v_fix
    da_fix = a_fix2 - a_fix
    phi_fix = atan2(da_fix, dv_fix)


    # Determine the angle between phi between (v_fix, a_fix) and (v, a).
    dv = v0 - v_fix
    da = a0 - a_fix
    r = np.sqrt(dv ** 2 + da ** 2)
    phi = atan2(da, dv)
    # Calculate difference between phi_fix and phi
    dphi = phi - phi_fix
    # Move phi so that the difference is reduced
    v_new = v_fix + r*cos(phi - 0.02*dphi)
    a_new = a_fix + r*sin(phi - 0.02*dphi)
    return v_new, a_new


def move_to_increase_return_time(v0, a0, v_fix, a_fix):
    dx = 0.
    v = v_fix
    a = a_fix
    while dx < 0.02:
        v, a, spike = alif.forward(v, a)
        if spike == False:
            dx = np.sqrt((v - v_fix) ** 2 + (a - a_fix) ** 2)
        if spike == True:
            dx = np.sqrt((v + 1 - v_fix) ** 2 + (a - delta_a - a_fix) ** 2)
    v_fix2 = v
    a_fix2 = a
    dv_fix = v_fix2 - v_fix
    da_fix = a_fix2 - a_fix
    phi_fix = atan2(da_fix, dv_fix)

    # Determine the angle between phi between (v_fix, a_fix) and (v, a).
    dv = v0 - v_fix
    da = a0 - a_fix
    r = np.sqrt(dv ** 2 + da ** 2)
    phi = atan2(da, dv)
    # Calculate difference between phi_fix and phi
    dphi = phi - phi_fix
    # Move phi so that the difference is increased
    v_new = v_fix + r * cos(phi + 0.02 * dphi)
    a_new = a_fix + r * sin(phi + 0.02 * dphi)
    return v_new, a_new


def initial_pair(v_lc, a_lc, ISI, dt):
    # Find a point above the LC that has the same phase as (v_lc, a_lc)
    v_candidate = v_lc + 0.01
    a_candidate = a_lc + 0.01
    while True:
        t = 0
        passed_isochrone = False
        v = v_candidate
        a = a_candidate
        # Integrate v,a until the isochrone is passed
        while not passed_isochrone:
            v_tmp = v
            a_tmp = a
            v, a, spike = alif.forward(v, a)
            t += dt
            passed_isochrone = intersect(v_tmp, a_tmp, v, a, v_lc, a_lc, v_candidate, a_candidate)
        # If the isochrone is passed save the passage time T.
        T = t
        # If T is too large or too small change (v, a) accordingly
        if abs(T - ISI) > 0.0001 * isi:
            if T > ISI:
                v_candidate, a_candidate = move_to_reduce_return_time(v_candidate, a_candidate, v_lc, a_lc)
            if T < ISI:
                v_candidate, a_candidate = move_to_increase_return_time(v_candidate, a_candidate, v_lc, a_lc)
        # If T matches the deterministic ISI end.
        else:
            v_plus_iso = v_candidate
            a_plus_iso = a_candidate
            break

    # Find a point below the LC that has the same phase as (v_lc, a_lc)
    v_candidate = v_lc - 0.01
    a_candidate = a_lc - 0.01
    while True:
        t = 0
        passed_isochrone = False
        v = v_candidate
        a = a_candidate
        # Integrate v,a until the isochrone is passed
        while not passed_isochrone:
            v_tmp = v
            a_tmp = a
            v, a, spike = alif.forward(v, a)
            t += dt
            passed_isochrone = intersect(v_tmp, a_tmp, v, a, v_lc, a_lc, v_candidate, a_candidate)
        # If the isochrone is passed save the passage time T.
        T = t
        # If T is too large or too small change (v, a) accordingly
        if abs(T - ISI) > 0.0001 * isi:
            if T > ISI:
                v_candidate, a_candidate = move_to_reduce_return_time(v_candidate, a_candidate, v_lc, a_lc)
            if T < ISI:
                v_candidate, a_candidate = move_to_increase_return_time(v_candidate, a_candidate, v_lc, a_lc)
        # If T matches the deterministic ISI end.
        else:
            v_minus_iso = v_candidate
            a_minus_iso = a_candidate
            break

    return [v_plus_iso, a_plus_iso], [v_minus_iso, a_minus_iso]


def expand_isochrone_by_one_point(isochrone):
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
    def __init__(self, mu, tau_a, delta_a, v_r, v_t, dt):
        self.mu = mu
        self.tau_a = tau_a
        self.delta_a = delta_a
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
    # Initialize Adaptive leaky IF model
    mu: float = 5.0
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.00005
    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    print(isi)
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

    twopi = len(v_lc)
    pihalf = int(twopi/4)
    pi = int(twopi/2)
    pithreehalf = int(3*twopi/4)

    phases = [v_lc[pihalf], a_lc[pihalf]], [v_lc[pi], a_lc[pi]], [v_lc[pithreehalf], a_lc[pithreehalf]]
    for v_lc0, a_lc0 in phases[1:2]:
        [v_right, a_right], [v_left, a_left] = initial_pair(v_lc0, a_lc0, isi, dt)
        # Split the isochrone into two parts. Below/left and ontop/right of the LC.
        isochrone_right = [[v_lc0, a_lc0], [v_right, a_right]]
        isochrone_left = [[v_lc0, a_lc0], [v_left, a_left]]

        # Plot (v_left, a_left), (v_lc, a_lc) and (v_right, a_right)
        #ax.scatter(v_left, a_left, c="C0", zorder=5)
        #ax.scatter(v_lc0, a_lc0, c="C0", zorder=5)
        #ax.scatter(v_right, a_right, c="C0", zorder=5)

        axins.scatter(v_left, a_left, c="C0", zorder=5)
        axins.scatter(v_lc0, a_lc0, c="C0", zorder=5)
        axins.scatter(v_right, a_right, c="C0", zorder=5)
        axins.set_xlim([0.9 * v_lc0, 1.1 * v_lc0])
        axins.set_ylim([0.99 * a_lc0, 1.01 * a_lc0])


        # THIS IS WHERE EVERTHING HAPPENS!!!
        isochrone_right = expand_isochrone(isochrone_right)
        print("SWITCHING ISOCHRONES")
        isochrone_left = expand_isochrone(isochrone_left)

        # Check if Isochrone is good
        v, a = isochrone_right[-1]
        for i in range(5):
            ax.scatter(v, a, c="C3", zorder=4)
            axins.scatter(v, a, c="C3")
            v, a = alif.forward_for_T(v, a, isi)

        v, a = isochrone_left[-1]
        for i in range(6):
            ax.scatter(v, a, c="C3", zorder=4)
            axins.scatter(v, a, c="C3")
            v, a = alif.forward_for_T(v, a, isi)

        # Combine both Isochrones to a single list
        isochrone = isochrone_left[::-1] + isochrone_right
        ax.plot([x[0] for x in isochrone], [x[1] for x in isochrone])

    # Write Isochrone to file
    home = os.path.expanduser("~")
    file_str = home + "/Data/isochrones/isochrones_file_mu{:.2f}.dat".format(mu)
    with open(file_str, "w") as f:
        for v, a in isochrone:
            f.write("{:.4f} {:.4f} \n".format(v, a))

    ax.grid(True, ls="--")
    plt.savefig(home + "/Data/isochrones/Isochrone_mu{:.2f}.pdf".format(mu))
    plt.show()
