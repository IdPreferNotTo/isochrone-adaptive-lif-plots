import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def limit_cycle_adaptive_leaky_if(mu, gamma, D, tau_a, delta_a, v_r, v_t, dt):
    t: float = 0
    v: float = 0
    a: float = 0

    spike_count_transient: int = 0
    while (spike_count_transient) < 100:
        v += (mu - gamma * v - a) * dt
        a += (-a / tau_a) * dt
        t += dt
        if v >= v_t:
            v = v_r
            a += delta_a
            spike_count_transient += 1

    t = 0
    v_s = []
    a_s = []
    while (True) :
        v += (mu - gamma * v - a) * dt
        a += (-a / tau_a) * dt
        t += dt

        v_s.append(v)
        a_s.append(a)
        if v >= v_t:
            return [v_s, a_s]



def stochastic_period_adapative_leaky_if(mu, gamma, D, tau_a, delta_a, v_r, v_t, dt):
    t: float = 0
    v: float = 0
    a: float = 0

    spike_count_transient: int = 0
    spike_count : int = 0
    while(spike_count_transient) < 100:
        xi = np.random.normal(0, 1)
        v += (mu - gamma * v - a) * dt + np.sqrt(2 * D * dt) * xi
        a += (-a / tau_a) * dt
        t += dt
        if v >= v_t:
            v = v_r
            a += delta_a
            spike_count_transient += 1

    t = 0
    while (spike_count) < 1000:
        xi = np.random.normal(0, 1)
        v += (mu - gamma * v - a) * dt + np.sqrt(2 * D * dt) * xi
        a += (-a / tau_a) * dt
        t += dt
        if v >= v_t:
            v = v_r
            a += delta_a
            spike_count += 1
    return t/spike_count



def line_passing_adaptive_leaky_if(v0, a0, mu, gamma, D, tau_a, delta_a, v_r, v_t, dt):
    t: float = 0
    v: float = v0
    a: float = a0

    reset: bool = False

    while(True):
        xi = np.random.normal(0, 1)
        v += (mu - gamma*v - a)*dt + np.sqrt(2*D*dt)*xi
        a += (-a/tau_a)*dt
        t += dt
        if v >= v_t:
            v = v_r
            a += delta_a
            reset = True

        if v > v0 and reset==True:
            return [t, v, a]


if __name__ == "__main__":
    mu: float = 1.5
    gamma: float = 1.0
    D: float = 0.1
    tau_a: float = 1.0
    delta_a: float = 0.5

    v_r: float = 0
    v_t: float = 1
    dt: float = 0.005


    isi = stochastic_period_adapative_leaky_if(mu, gamma, D, tau_a, delta_a, v_r, v_t, dt)
    v_lc, a_lc = limit_cycle_adaptive_leaky_if(mu, gamma, D, tau_a, delta_a, v_r, v_t, dt)
    v0: float = 0.2
    a0: float = 0.5
    data = []
    i = 100
    for _ in range(5000):
        [t, v, a] = line_passing_adaptive_leaky_if(v_lc[i], a_lc[i], mu, gamma, D, tau_a, delta_a, v_r, v_t, dt)
        data.append([t/isi, v, a])
    times, v_s, a_s = np.transpose(data)


    fig = plt.figure(tight_layout=True, figsize=(6, 9 / 2))
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.set_xlim([0.2, 1])
    axins = inset_axes(ax, width="40%", height="40%", loc=1)
    axins.set_xlabel("$v$")
    axins.set_ylabel("$a$")
    axins.scatter(v_lc[i], a_lc[i], c="k")
    axins.plot(v_lc, a_lc)
    axins.axvline(v_lc[i], c="C3")
    ax.scatter(a_s, times,s=1)

    times_over_a = [[] for _ in range(50)]
    a_start = 0.
    a_stop = 1.
    for set in data:
        idx = int((set[2] - a_start) / (a_stop -a_start) * 50)
        times_over_a[idx].append(set[0])
    mean_times_over_a = [np.mean(x) for x in times_over_a]
    ax.plot(np.linspace(0, 1, 50), mean_times_over_a, c="k")
    ax.axhline(1., ls="--", c="C7")
    ax.axvline(a_lc[10], ls="--", c="C7")
    plt.show()

    print(isi)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
