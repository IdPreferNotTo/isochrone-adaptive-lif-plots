import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.colors as mcolors


class adaptive_leaky_if():
    def __init__(self, mu, gamma, tau_a, delta_a, v_r, v_t, dt):
        self.mu = mu
        self.gamma = gamma
        self.tau_a = tau_a
        self.delta_a = delta_a
        self.v_r = v_r
        self.v_t = v_t
        self.dt = dt

    def isi(self):
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


    def mean_first_passage_time(self, v, a):
        t = 0
        while v < self.v_t:
            v += (self.mu - self.gamma * v - a) * self.dt
            a += (-a / self.tau_a) * self.dt
            t += self.dt
        return t


if __name__ == "__main__":
    mu: float = 5.0
    gamma: float = 1.0
    D: float = 0.1
    tau_a: float = 2.0
    delta_a: float = 1.0

    v_res: float = 0
    v_thr: float = 1
    dt: float = 0.005
    alif =  adaptive_leaky_if( mu, gamma, tau_a, delta_a, v_res, v_thr, dt)
    isi = alif.isi()

    a_s = np.linspace(0, 2, 100)
    v_s = np.linspace(0, 1, 100)
    mfpt = []
    for a in a_s:
        Ts = []
        for v in v_s:
            T = alif.mean_first_passage_time(v, a)
            Ts.append(T)
            print(T)
        mfpt.append(list(Ts))

    fig = plt.figure(tight_layout=True, figsize=(6, 9 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlabel("$v$")
    ax.set_ylabel("$a$")

    #R = []
    #R_row = []
    #for r in rs:
    #    R_row.append(r)
    #    if len(R_row) == 50:
    #        R.append(list(R_row))
    #        R_row = []

    cs = ax.pcolormesh(v_s, a_s, mfpt, linewidth=0, rasterized=True, vmin=0., vmax=isi, shading="auto")
    ax.contour(v_s, a_s, mfpt, c="k")
    cbar = fig.colorbar(cs)
    cbar.set_label('$\phi$', rotation=270)
    plt.show()


