import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import rc
from matplotlib import rcParams
import os

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
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Computer Modern Roman'
    rc('text', usetex=True)
    fig = plt.figure(tight_layout=True, figsize=(4, 8/3))
    gs = gs.GridSpec(2, 2)
    x0 = 0.10
    y0 = 0.10
    wspacer = 0.1
    hspacer = 0.1
    width = (0.95 - x0 - wspacer)/2
    height = (0.95 - y0 - hspacer)/2

    ax1 = fig.add_axes([x0, y0 + height + hspacer, width, height])
    ax2 = fig.add_axes([x0 + width + wspacer, y0 + height + hspacer, width, height])
    ax3 = fig.add_axes([x0 + width + wspacer, y0, width, height])
    ax4 = fig.add_axes([x0, y0, width, height])

    home = os.path.expanduser("~")

    mu: float = 2.0
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.0001

    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    print(isi)
    v_lc, a_lc, ts = alif.limit_cycle()
    v_s = []
    a_s = []
    v = 0.55
    a = a_lc[int(len(a_lc)/2)]
    for i in range(19_900):
        v, a, fired = alif.forward(v,a)
        v_s.append(v)
        a_s.append(a)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.plot(v_lc, a_lc, lw = 1., c="k", label="Limit cycle")
    v_pi = v_lc[int(len(a_lc)/2)]
    a_pi = a_lc[int(len(a_lc)/2)]
    ax1.scatter(v_pi, a_pi, s=20, ec="C3", fc="w", zorder=3)
    ax1.annotate("$p_0$", (v_pi-0.01, a_pi - 0.01))
    ax1.scatter(0.55, a_pi, s=20, ec="C3", fc="w", zorder=3)
    ax1.annotate("$p_1$", (0.54 , a_pi - 0.01))
    ax1.plot([v_pi, 0.55], [a_pi, a_pi], c="C3")
    ax1.annotate(r"$l_{0\to 1}$", (0.525 , a_pi + 0.005))

    ax1.scatter(0.55, a_pi + 0.01, s=20, ec="C3", fc="w", zorder=3)
    ax1.plot([0.55, 0.55], [a_pi, a_pi + 0.01], c="C3", ls=":")
    ax1.annotate("$\hat{p}_1$", (0.555, a_pi + 0.015))

    ax1.set_xlim([0.95*v_pi, 1.1*v_pi])
    ax1.set_ylim([0.975*a_pi, 1.025*a_pi])
    #ax1.scatter(0.55, 0.967303, s=20, ec="C3", fc="w", zorder=3)
    #ax1.plot([v_pi, 0.55], [a_pi, 0.967303], c="C3")
    ax1.plot(v_s, a_s, c="k", ls=":")

    con1 = ConnectionPatch(
        # in axes coordinates
        xyA=(1.0, 1.0), coordsA=ax1.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(1.1*v_pi, 0.975*a_pi), coordsB=ax2.transData,
        arrowstyle="-", color="C7")

    con2 = ConnectionPatch(
        # in axes coordinates
        xyA=(1.0, 0.0), coordsA=ax1.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(0.95*v_pi, 0.975*a_pi), coordsB=ax2.transData,
        arrowstyle="-", color="C7")

    ax2.add_artist(con1)
    ax2.add_artist(con2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.plot(v_lc, a_lc, lw = 1., c="k", label="Limit cycle")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    rect = Rectangle((0.95*v_pi, 0.975*a_pi), width=0.15*v_pi, height=0.05*a_pi, edgecolor="C7", facecolor="w")
    ax2.add_patch(rect)

    isochrone_right_v = [0.514048, 0.6, 0.7, 0.8, 0.9, 1.0]
    isochrone_right_a = [0.954298, 0.98839, 1.03256, 1.08174, 1.13891, 1.20908]

    ax2.scatter(isochrone_right_v, isochrone_right_a, s=10, ec="C3", fc="w", zorder=3)
    ax2.plot(isochrone_right_v, isochrone_right_a, c="C3", zorder=1)

    ax2.plot([0.7, 0.7], [1.2, 1.03256], c="k", ls=":")
    ax2.plot([0.8, 0.8], [1.2, 1.08174], c="k", ls=":")
    ax2.plot([0.7, 0.8], [1.2, 1.2], c="k")
    ax2.annotate(r"$\delta_V$", (0.71, 1.25))
    ax2.set_xlim([0.7*v_pi, 1])
    ax2.set_ylim([0.7*a_pi, a_lc[0]])

    v_s = []
    a_s = []
    v = 0.8
    a = 1.08174
    for i in range(20_000):
        v, a, fired = alif.forward(v,a)
        v_s.append(v)
        a_s.append(a)
    vs1 = []
    as1 = []
    vs2 = []
    as2 = []
    switch=False
    for v, v_before, a in zip(v_s[1:], v_s[:-1], a_s):
        if abs(v - v_before) > 0.5:
            switch = True

        if switch==False:
            vs1.append(v)
            as1.append(a)
        else:
            vs2.append(v)
            as2.append(a)
    ax2.plot(vs1, as1, c="k", ls=":")
    ax2.plot(vs2, as2, c="k", ls=":")

    ax3.plot(v_lc, a_lc, c="k", lw = 1.)
    ax3.set_xticks([])
    ax3.set_yticks([])
    rect = Rectangle((0.7*v_pi, 0.7*a_pi), width=1 - 0.7*v_pi, height=a_lc[0] - 0.7*a_pi, edgecolor="C7", facecolor="w")
    ax3.add_patch(rect)
    con1 = ConnectionPatch(
        # in axes coordinates
        xyA=(0.0, 0.0), coordsA=ax2.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(0.7*v_pi, a_lc[0]), coordsB=ax3.transData,
        arrowstyle="-", color="C7")

    con2 = ConnectionPatch(
        # in axes coordinates
        xyA=(1.0, 0.0), coordsA=ax2.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(1., a_lc[0]), coordsB=ax3.transData,
        arrowstyle="-", color="C7")

    ax3.add_artist(con1)
    ax3.add_artist(con2)
    ax3.plot(isochrone_right_v, isochrone_right_a, c="C3", zorder=1)

    isochrone_left_v = [0, 0.1, 0.2, 0.3, 0.4, 0.514048]
    isochrone_left_a = [0.776341, 0.807932, 0.840524, 0.874115, 0.911707, 0.954298]
    ax3.scatter(isochrone_left_v, isochrone_left_a, s=10, ec="C3", fc="w", zorder=3)
    ax3.plot(isochrone_left_v, isochrone_left_a, c="C3", zorder=1)
    ax3.set_xlim([0, 1.05])
    ax3.set_ylim([a_lc[-1], a_lc[0]+0.05])

    rect = Rectangle((0, a_lc[-1]), width=1.05, height=a_lc[0] - a_lc[-1] + 0.05, edgecolor="C7", facecolor="w")
    ax4.add_patch(rect)
    con1 = ConnectionPatch(
        # in axes coordinates
        xyA=(0.0, 1.0), coordsA=ax3.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(1.05, a_lc[0] +0.05), coordsB=ax4.transData,
        arrowstyle="-", color="C7")

    con2 = ConnectionPatch(
        # in axes coordinates
        xyA=(0.0, 0.0), coordsA=ax3.transAxes,
        # x in axes coordinates, y in data coordinates
        xyB=(1.05, a_lc[-1]), coordsB=ax4.transData,
        arrowstyle="-", color="C7")

    ax4.add_artist(con1)
    ax4.add_artist(con2)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.plot(v_lc, a_lc, c="k", lw = 1.)
    ax4.set_yticklabels([])
    ax4.scatter([0, 1], [0.776341, 1.20908], s=10, ec="C3", fc="w", zorder=3)
    ax4.plot(isochrone_left_v, isochrone_left_a, c="C3", zorder=1)
    ax4.plot(isochrone_right_v, isochrone_right_a, c="C3", zorder=1)
    ax4.scatter([1, 0], [0.776341-1, 1.20908+1], s=10, ec="C3", fc="w", zorder=3)
    ax4.plot([1, 1, 0], [1.20908, 1.20908+1, 1.20908+1], c="C3", ls=":")
    ax4.plot([0, 0, 1], [0.776341, 0.776341-1, 0.776341-1], c="C3", ls=":")
    ax4.annotate("$p_T$", (0.9, 1.20908 - 0.35))
    ax4.annotate("$p_R$", (0.05, 1.20908 + 1 - 0.3))
    ax4.annotate("$p_R$", (0.05, 0.776341 + 0.25), zorder=1)
    ax4.annotate("$p_T$", (0.9, 0.776341 -1. + 0.25))
    ax4.set_xlabel("$v$")
    ax4.set_ylabel("$a$")
    ax3.set_xlabel("$v$")
    ax1.set_ylabel("$a$")

    ax1.text(-0.175, 1.00, "(a)", size=10, transform=ax1.transAxes)
    ax2.text(-0.175, 1.00, "(b)", size=10, transform=ax2.transAxes)
    ax3.text(-0.175, 1.00, "(c)", size=10, transform=ax3.transAxes)
    ax4.text(-0.175, 1.00, "(d)", size=10, transform=ax4.transAxes)
    #plt.savefig(home + "/Data/isochrones/fig2.pdf", transparent=True)

    plt.show()
