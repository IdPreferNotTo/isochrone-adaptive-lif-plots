import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os

import styles as st

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
    mu: float = 2.0
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0.
    v_thr: float = 1.
    dt: float = 0.00005

    alif = adaptive_leaky_if(mu, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.period()
    print(isi)
    v_lc, a_lc, ts = alif.limit_cycle()

    st.set_default_plot_style()
    fig = plt.figure(tight_layout=True, figsize=(4, 4))
    gs = gs.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    st.remove_top_right_axis([ax1, ax2])

    ax1.set_xlabel("$v$")
    ax1.set_xlim([-0.2, 1.1])
    ax1.set_ylabel("$a$")
    ax1.set_ylim([0, 2])
    ax1.set_yticks([0, 0.5, 1, 1.5, 2])
    ax1.plot(v_lc, a_lc, c="k", label="Limit cycle")

    ax1.axvline(1, lw=0.7, c="k", ls="--")
    ax1.axvline(0, lw=0.7, c="k", ls=":")
    ax1.plot([1, 1], [a_lc[-1], a_lc[0]], c="k")
    ax1.arrow(1, a_lc[int(len(a_lc) / 2)], dx=0, dy=0.1, shape="full", fc="k", lw=0, length_includes_head=True,
              head_length=0.1, head_width=.05)
    ax1.annotate("increase", xy=(1 + 0.05, a_lc[int(len(a_lc) / 2)]), va="center", rotation=-90)

    ax1.plot([0, 1], [a_lc[0], a_lc[0]], c="k")
    ax1.arrow(0.5, a_lc[0], dx=-0.1, dy=0.0, shape="full", fc="k", lw=0, length_includes_head=True,
              head_length=.05, head_width=.1)
    ax1.annotate("reset", xy=(0.4, a_lc[0] + 0.2), va="center", backgroundcolor="white")
    x, y = np.meshgrid(np.linspace(-0.2, 1, 10), np.linspace(0, 2, 10))
    dv = mu - x - y
    da = -y / tau_a
    dx = np.sqrt(dv ** 2 + da ** 2)
    strm = ax1.streamplot(x, y, dv, da, color="C7", arrowsize=0.75, linewidth=0.3)

    return_times = []
    adaps = np.linspace(1, 3, 100)
    for adap in adaps:
        t = 0
        v = 0
        a = adap
        fire = False
        while not fire:
            t += dt
            v, a, fire = alif.forward(v=v, a=a)
        return_times.append(t)

    ax2.set_xlim([0, 2])
    ax2.set_ylim([1, 4])
    ax2.set_ylabel(r"$T_0((v_T, a^*) \to (v_T, a))$")
    ax2.set_xlabel("$a^*$")
    ax2.axhline(isi, ls=":", color="C7")
    ax2.axvline(a_lc[-1], ls="--", color="C7")
    ax2.plot([a-1 for a in adaps], return_times, lw=1, color=st.colors[1])

    home = os.path.expanduser("~")
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/10_why_correlations.pdf",
                transparent=True)
    plt.show()

