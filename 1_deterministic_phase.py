import os
from math import atan2, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def initial_pair(v_lc, a_lc, ISI, r, dt):
    # Find a point above the LC that has the same phase as (v_lc, a_lc)
    v_candidate = v_lc + 0.01
    a_candidate = a_lc + 0.01
    while True:
        t = 0
        passed_isochrone = False
        v = v_candidate
        a = a_candidate
        has_spiked = False
        # Integrate v,a until the isochrone is passed
        while not passed_isochrone or not has_spiked:
            v_before = v
            a_before = a
            v, a, spike = alif.forward(v, a)
            if spike == True:
                v_after = 1.
                a_after = a - delta_a
                has_spiked = True
            else:
                v_after = v
                a_after = a
            t += dt
            passed_isochrone = intersect(v_before, a_before, v_after, a_after, v_lc, a_lc, v_candidate, a_candidate)
        # If the isochrone is passed save the passage time T.
        T = t
        # If T is too large or too small change (v, a) accordingly
        if abs(T - ISI)/ISI > 0.0001:
            if T > ISI:
                print("ISI too long, {:.4f}".format(abs(T - ISI)/ISI))
                v_candidate, a_candidate = move_to_reduce_return_time(v_candidate, a_candidate, v_lc, a_lc, r)
            else:
                print("ISI too short, {:.4f}".format(abs(T - ISI)/ISI))
                v_candidate, a_candidate = move_to_increase_return_time(v_candidate, a_candidate, v_lc, a_lc, r)
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
        has_spiked = False
        # Integrate v,a until the isochrone is passed
        while not passed_isochrone or not has_spiked:
            v_before = v
            a_before = a
            v, a, spike = alif.forward(v, a)
            if spike == True:
                v_after = 1.
                a_after = a - delta_a
                has_spiked = True
            else:
                v_after = v
                a_after = a
            t += dt
            passed_isochrone = intersect(v_before, a_before, v_after, a_after, v_lc, a_lc, v_candidate, a_candidate)
        # If the isochrone is passed save the passage time T.
        T = t
        # If T is too large or too small change (v, a) accordingly
        if abs(T - ISI)/ISI > 0.0001:
            if T > ISI:
                #print("ISI too long, {:.4f}".format(abs(T - ISI)/ISI))
                v_candidate, a_candidate = move_to_reduce_return_time(v_candidate, a_candidate, v_lc, a_lc, r)
            else:
                #print("ISI too short, {:.4f}".format(abs(T - ISI)/ISI))
                v_candidate, a_candidate = move_to_increase_return_time(v_candidate, a_candidate, v_lc, a_lc, r)
        # If T matches the deterministic ISI end.
        else:
            v_minus_iso = v_candidate
            a_minus_iso = a_candidate
            break

    return [v_plus_iso, a_plus_iso], [v_minus_iso, a_minus_iso]


def expand_isochrone_by_one_point(isochrone, r):
    # Returns (v, a, bool v in [0, 1]?). (v,a) is a candidate for a pooint on the isochrone. the third value indicates
    # whether v is in the interval [0, 1].
    v1, a1 = isochrone[-1]
    v0, a0 = isochrone[-2]
    dv = v1 - v0
    da = a1 - a0
    dx = np.sqrt(dv ** 2 + da ** 2)
    dv = dv / dx
    da = da / dx
    v_candidate = v1 + r * dv
    a_candidate = a1 + r * da
    if v_candidate > 1.:
        return v_candidate, a_candidate, True
    elif v_candidate < 0.:
        return v_candidate, a_candidate, True
    else:
        return v_candidate, a_candidate, False


def project_on_boundary(v0, a0, v_fix, a_fix):
    dv = v0 - v_fix
    da = a0 - a_fix
    if(v0 > 1.):
        dv_to_boundary = 1. - v0
    if(v0 < 0.):
        dv_to_boundary = 0. - v0
    v_new = v0 + dv_to_boundary
    a_new = a0 + da * (dv_to_boundary/dv)
    return v_new, a_new


def expand_isochrone(isochrone, ISI, r):
    while True:
        # Find a candidate for a new point on the isochrone
        v_candidate, a_candidate, behind_boundary = expand_isochrone_by_one_point(isochrone, r)
        if not behind_boundary:
            on_isochrone = False
            isochrone.append([v_candidate, a_candidate])
            # Check if return T time from this (v,a) back to the isochrone is T = ISI
            while not on_isochrone:
                T = first_passage_time_last_point_isochrone_to_isochrone(isochrone, dt)
                if abs(T - ISI) / ISI > 0.0005:
                    v_new, a_new = isochrone[-1]
                    v_fix, a_fix = isochrone[-2]
                    if T > ISI:
                        print("ISI too long, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - ISI) / ISI, v_new, a_new))
                        v, a = move_to_reduce_return_time(v_new, a_new, v_fix, a_fix, r)
                    else:
                        print("ISI too short, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - ISI) / ISI, v_new, a_new))
                        v, a = move_to_increase_return_time(v_new, a_new, v_fix, a_fix, r)
                    isochrone[-1] = [v, a]
                else:
                    # Return time on point
                    print("on point")
                    on_isochrone = True
        else:
            v_fix, a_fix = isochrone[-1]
            v_candidate, a_candidate = project_on_boundary(v_candidate, a_candidate, v_fix, a_fix)
            isochrone.append([v_candidate, a_candidate])
            while True:
                T = first_passage_time_last_point_isochrone_to_isochrone(isochrone, dt)
                if abs(T - ISI) / ISI > 0.0005:
                    v_new, a_new = isochrone[-1]
                    if T > ISI:
                        print("ISI too long, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - ISI) / ISI, v_new, a_new))
                        a = a_new - 0.0001
                    else:
                        print("ISI too short, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - ISI) / ISI, v_new, a_new))
                        a = a_new + 0.0001
                    isochrone[-1] = [v_new, a]
                else:
                    # Return time on point
                    print("on point")
                    return isochrone

# def expand_isochrone(isochrone, ISI, r):
#     behind_boundary = False
#     while not behind_boundary:
#         on_isochrone = False
#         # Find a candidate for a new point on the isochrone
#         v_candidate, a_candidate, behind_boundary = expand_isochrone_by_one_point(isochrone, r)
#         # If is v-component is outside [0, 1] stop the loop
#         isochrone.append([v_candidate, a_candidate])
#         # Check if return T time from this (v,a) back to the isochrone is T = ISI
#         while not on_isochrone:
#             T = first_passage_time_last_point_isochrone_to_isochrone(isochrone, dt)
#             if abs(T - ISI)/ISI > 0.0001:
#                 v_new, a_new = isochrone[-1]
#                 v_fix, a_fix = isochrone[-2]
#                 if T > ISI:
#                     print("ISI too long, {:.4f}".format(abs(T - ISI)/ISI))
#                     v, a = move_to_reduce_return_time(v_new, a_new, v_fix, a_fix, r)
#                 else:
#                     print("ISI too short, {:.4f}".format(abs(T - ISI)/ISI))
#                     v, a = move_to_increase_return_time(v_new, a_new, v_fix, a_fix, r)
#                 isochrone[-1] = [v, a]
#             else:
#                 # Return time on point
#                 print("on point")
#                 on_isochrone = True
#     return isochrone


def move_to_reduce_return_time(v0, a0, v_fix, a_fix, r):
    # Consider two points (v, a) and (v_fix, a_fix). The latter one is know to lie on the isochrone. Define (v_fix, a_fix)
    # as the center of a circle so that the point (v, a) can be described by radius and angle. Change the angle of (v, a)
    # so that the return time to the isochrone is reduced.

    # The flow of (v_fix, a_fix) will pass the circle with radius r at some angle. Find that angle (compared to the x). Moving the
    # point (v,a) closer to the angle phi_fix will eventually reduces the return time.
    dx = 0.
    v = v_fix
    a = a_fix
    while dx < r:
        v, a, spike = alif.forward(v, a)
        if spike == False:
            dx = np.sqrt((v-v_fix)**2 + (a-a_fix)**2)
        else:
            dx = np.sqrt((v + 1 - v_fix)**2 + (a - delta_a - a_fix)**2)
    v_fix2 = v
    a_fix2 = a
    dv_fix = v_fix2 - v_fix
    da_fix = a_fix2 - a_fix
    phi_fix = atan2(da_fix, dv_fix)


    # Determine the angle between phi between (v_fix, a_fix) and (v, a).
    dv = v0 - v_fix
    da = a0 - a_fix
    phi = atan2(da, dv)
    # Calculate difference between phi_fix and phi
    dphi = phi - phi_fix
    # Move phi so that the difference is reduced
    v_new = v_fix + r*cos(phi - 0.002*dphi)
    a_new = a_fix + r*sin(phi - 0.002*dphi)
    return v_new, a_new


def move_to_increase_return_time(v0, a0, v_fix, a_fix, r):
    dx = 0.
    v = v_fix
    a = a_fix
    while dx < r:
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
    phi = atan2(da, dv)
    # Calculate difference between phi_fix and phi
    dphi = phi - phi_fix
    # Move phi so that the difference is increased
    v_new = v_fix + r * cos(phi + 0.002 * dphi)
    a_new = a_fix + r * sin(phi + 0.002 * dphi)
    return v_new, a_new


def first_passage_time_last_point_isochrone_to_isochrone(isochrone, dt):
    t = 0
    v, a = isochrone[-1]
    ref_steps = 0
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
        t += dt
        ref_steps += 1
        # For every segment of the isochrone check if this segment and the segment that describes
        # the change of v, a ((v_tmp, a_tmp), (v, a)) intersect.
        for (v_iso_before, a_iso_before), (v_iso_after, a_iso_after) in zip(isochrone[:-1], isochrone[1:]):
            if v_iso_before == 1 and v_iso_after == 0:
                continue
            if v_iso_before == 0 and v_iso_after == 1:
                continue
            passed_isochrone = intersect(v_before, a_before, v_after, a_after, v_iso_before, a_iso_before, v_iso_after, a_iso_after)
            if passed_isochrone and (ref_steps > 1):
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


def start_new_branch(isochrone, delta_a, r):
    v0, a0 = isochrone[-2]
    v1, a1 = isochrone[-1]
    if v1 == 1:
        v0_new = v0 - 1.
        a0_new = a0 + delta_a
        v1_new = v1 - 1.
        a1_new = a1 + delta_a
    elif v1 == 0:
        v0_new = v0 + 1.
        a0_new = a0 - delta_a
        v1_new = v1 + 1.
        a1_new = a1 - delta_a
    else:
        print("Can not start new Branch. Endpoint (v,a)=({:.2f}, {:.2f}) not valid".format(v1, a1))
    dv = v1_new - v0_new
    da = a1_new - a0_new
    dx = np.sqrt(dv ** 2 + da ** 2)
    dv = dv / dx
    da = da / dx
    v2_new = v1_new + r * dv
    a2_new = a1_new + r * da
    return v1_new, a1_new, v2_new, a2_new


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
    r: float = 0.05
    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    print(isi)
    v_lc, a_lc, ts = alif.limit_cycle()

    # Initialize Plot
    fig = plt.figure(tight_layout=True, figsize=(6, 9 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")
    ax.plot(v_lc, a_lc, c="k")
    phase = 3*np.pi/2
    idx_phase = int(len(v_lc)*(phase)/(2*np.pi))
    v_lc0 = v_lc[idx_phase]
    a_lc0 = a_lc[idx_phase]

    print("Finding initial pair...")
    [v_right, a_right], [v_left, a_left] = initial_pair(v_lc0, a_lc0, isi, r, dt)
    print("done")
    # Split the isochrone into two parts. Below/left and ontop/right of the LC.
    isochrone_right = [[v_lc0, a_lc0], [v_right, a_right]]
    isochrone_left = [[v_lc0, a_lc0], [v_left, a_left]]

    # THIS IS WHERE EVERTHING HAPPENS!!!
    isochrone_right = expand_isochrone(isochrone_right, isi, r)
    print("START NEW BRANCH")
    v1_new, a1_new, v2_new, a2_new = start_new_branch(isochrone_right, delta_a, r)
    isochrone_right.append([v1_new, a1_new])
    isochrone_right.append([v2_new, a2_new])
    on_isochrone = False
    while not on_isochrone:
        T = first_passage_time_last_point_isochrone_to_isochrone(isochrone_right, dt)
        if abs(T - isi) / isi > 0.0001:
            v_new, a_new = isochrone_right[-1]
            v_fix, a_fix = isochrone_right[-2]
            if T > isi:
                print("ISI too long, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - isi) / isi, v_new, a_new))
                v, a = move_to_reduce_return_time(v_new, a_new, v_fix, a_fix, r)
            else:
                print("ISI too short, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - isi) / isi, v_new, a_new))
                v, a = move_to_increase_return_time(v_new, a_new, v_fix, a_fix, r)
            isochrone_right[-1] = [v, a]
        else:
            on_isochrone = True
    isochrone_right = expand_isochrone(isochrone_right, isi, r)

    print("SWITCHING ISOCHRONES")
    isochrone_left = expand_isochrone(isochrone_left, isi, r)
    print("START NEW BRANCH")
    v1_new, a1_new, v2_new, a2_new = start_new_branch(isochrone_left, delta_a, r)
    isochrone_left.append([v1_new, a1_new])
    isochrone_left.append([v2_new, a2_new])
    on_isochrone = False
    while not on_isochrone:
        T = first_passage_time_last_point_isochrone_to_isochrone(isochrone_left, dt)
        if abs(T - isi) / isi > 0.0001:
            v_new, a_new = isochrone_left[-1]
            v_fix, a_fix = isochrone_left[-2]
            if T > isi:
                print("ISI too long, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - isi) / isi, v_new, a_new))
                v, a = move_to_reduce_return_time(v_new, a_new, v_fix, a_fix, r)
            else:
                print("ISI too short, {:.4f}, v={:.3f}, a={:.3f}".format(abs(T - isi) / isi, v_new, a_new))
                v, a = move_to_increase_return_time(v_new, a_new, v_fix, a_fix, r)
            isochrone_left[-1] = [v, a]
        else:
            on_isochrone = True
    isochrone_left = expand_isochrone(isochrone_left, isi, r)

    # Check if Isochrone is good
    v, a = isochrone_right[-1]
    for i in range(5):
        ax.scatter(v, a, c="C3", zorder=4)
        v, a = alif.forward_for_T(v, a, isi)

    v, a = isochrone_left[-1]
    for i in range(6):
        ax.scatter(v, a, c="C3", zorder=4)
        v, a = alif.forward_for_T(v, a, isi)

    # Combine both Isochrones to a single list (both lists have to same starting point (on the limit cycle) which can be
    # a problem for later calculation of the return time to the isochrone, hence i simply remove that entry, i.e. the
    # first entry, of the right_isochrone)
    isochrone = isochrone_left[::-1] + isochrone_right[1:]
    ax.plot([x[0] for x in isochrone], [x[1] for x in isochrone])

    # Write Isochrone to file
    home = os.path.expanduser("~")
    file_str = home + "/Data/isochrones/isochrones_file_mu{:.2f}_{:.2f}.dat".format(mu, phase)
    with open(file_str, "w") as f:
        for v, a in isochrone:
            f.write("{:.4f} {:.4f} \n".format(v, a))

    ax.grid(True, ls="--")
    plt.savefig(home + "/Data/isochrones/Isochrone_mu{:.2f}_{:.2f}.pdf".format(mu, phase))
    plt.show()
