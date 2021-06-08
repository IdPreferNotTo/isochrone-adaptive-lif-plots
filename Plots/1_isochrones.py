import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
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

    home = os.path.expanduser("~")
    isochrone = np.loadtxt(home + "/Data/isochrones/isochrones_file_mu5.00_3.14.dat")
    stochstic_traj = np.loadtxt(home + "/Data/isochrones/stochastic_trajectory_D0.01_3.14.dat")

    v_isos = []
    a_isos = []
    v_iso = []
    a_iso = []
    for (v1, a1) in isochrone:
        if v1 == 1:
            v_iso.append(v1)
            a_iso.append(a1)
            v_isos.append(list(v_iso))
            a_isos.append(list(a_iso))
            v_iso = []
            a_iso = []
        else:
            v_iso.append(v1)
            a_iso.append(a1)


    fig = plt.figure(tight_layout=True, figsize=(6, 9 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")
    ax.plot(v_lc, a_lc, c="k", label="Limit cycle")
    ax.plot(v_isos[0], a_isos[0], c="k", ls="--", label="Isochrone $\phi = \pi$")
    for v_iso, a_iso in zip(v_isos[1:], a_isos[1:]):
        ax.plot(v_iso, a_iso, c="k", ls="--")

    ax.axvline(1, c="k", ls=":")
    ax.axvline(0, c="k", ls=":")
    ax.legend(fancybox=False, framealpha=1.)

    x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(1, 5, 20))
    dv = mu - x - y
    da = -y/tau_a
    dx = np.sqrt(dv**2 + da**2)
    strm = ax.streamplot(x, y, dv, da, color=dx, linewidth=0.75, cmap="viridis")
    fig.colorbar(strm.lines)

    v_st, a_st = np.transpose(stochstic_traj[:6300])
    ax.scatter(v_st, a_st, c="C0", s=2, zorder=5)


    ax.set_xlim([-0.1, 1.1])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.savefig(home + "/Data/isochrones/Trajectory_Isochrone_mu{:.2f}_D{:.2f}.pdf".format(mu, 0.01))
    plt.show()