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

    st.set_default_plot_style()
    fig = plt.figure(tight_layout=True, figsize=(4, 3))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")
    ax.axhline(a_lc[0], c="k", ls=":")
    ax.axhline(a_lc[-1], c ="k", ls=":")
    ax.arrow(1.05, a_lc[-1], 0, a_lc[0] - a_lc[-1], fc="k", length_includes_head=True, head_width=0.03,
             head_length=0.1, lw=0.5, clip_on=False)
    ax.arrow(1.05, a_lc[0], 0, a_lc[-1] - a_lc[0], fc="k", length_includes_head=True, head_width=0.03,
             head_length=0.1, lw=0.5, clip_on=False)
    ax.text(1.1, (a_lc[0] + a_lc[-1])/2, "$\Delta$", va="center", ha="center")

    home = os.path.expanduser("~")
    phase1: float = 1 * np.pi / 2
    phase2: float = 2 * np.pi / 2
    phase3: float = 3 * np.pi / 2
    phase4: float = 4 * np.pi / 2
    phases = [phase1, phase2, phase3, phase4]
    prc1 = [[0.2, 3.22753], [0.248558, 3.24553], [0.25, 3.24553]]
    prc2 = [[0.5, 2.98598], [0.500542, 2.98598], [0.55, 3.00801]]
    prc3 = [[0.75, 2.74718], [0.752029, 2.74718], [0.8, 2.77366]]
    prc4 = [[0.95, 2.49587], [0.999854, 2.52763], [1, 2.52763]]
    prcs = [prc1, prc2, prc3, prc4]

    for n, (phase, prc) in enumerate(zip(phases, prcs)):
        v1, a1 = prc[0]
        v2, a2 = prc[1]
        v3, a3 = prc[2]
        dv = v3 - v1
        da = (a3 - a1)/dv

        a = lambda v : a2 + da * (v - v2)

        if n==0:
            label = "$\phi = \pi/2$"
        elif n==1:
            label = "$\phi = \pi$"
        elif n==2:
            label = "$\phi = 3\pi/2$"
        elif n==3:
            label = "$\phi = 2\pi$"
        ax.plot([0, 1], [a(0), a(1)], c=st.colors[n], label=label)

    ax.plot(v_lc, a_lc, c="k", label="LC")
    ax.axvline(1, c="k")
    ax.axvline(0, c="k", ls="--")
    ax.legend(fancybox=False, framealpha=1., loc=3, ncol=2)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([1, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$v_R$", "$v_T$"])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


    #ax.grid()
    plt.savefig(home + "/Desktop/Presentations/Neuro MRT Phase/Figures/9_prc_LIFA.png",
                transparent=True)
    plt.show()