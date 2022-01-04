import numpy as np
from matplotlib import rc
from matplotlib import rcParams
from typing import List
from scipy.integrate import quad

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def delta(a, b):
    """
    Classical delta function \delta(a-b)
    """
    if a == b:
        return 1
    else:
        return 0


def theta(a, b):
    if a < b:
        return 0
    else:
        return 1


def moments(xs, k):
    """
    Calculates the k-th moment of the sample data xs:
    1/n \sum^n (x - <x>)**k
    where n the the sample length.
    """
    moment = 0
    mu = np.mean(xs)
    for x in xs:
        moment += (x - mu)**k
    return moment/len(xs)


def gaussian_dist(xs, mean, std):
    ys = []
    for x in xs:
        y = 1/np.sqrt(2*np.pi*(std**2)) * np.exp(-((x - mean)**2)/(2*std**2))
        ys.append(y)
    return ys


def coarse_grain_list(l: List[float], f: float):
    """
    Create a coarse grained version of the original list where the elements of the new list
    are the mean of the previous list over some window that is determined by f.
    f determines how many elements are averaged l_new[i] = mean(l[f*(i):f*(i+1)])
    """
    l_new = []
    max = int(len(l)/f) - 1
    for i in range(max):
        mean = np.mean(l[f*i:f*(i+1)])
        l_new.append(mean)
    return l_new


def steady_states_theory_invert_A(r_ref, r_opn, r_cls, m, n):
    A = np.array([[-n*r_ref, 0, 0, 0, 0, 0, 0, r_cls],
                  [n*r_ref, -n*r_ref, 0, 0, 0, 0, 0, 0],
                  [0, n*r_ref, -n*r_ref, 0, 0, 0, 0, 0],
                  [0, 0, n*r_ref, -n*r_opn, 0, 0, 0, 0],
                  [0, 0, 0, r_opn, -r_cls, 0, 0, 0],
                  [0, 0, 0, r_opn, r_cls, -r_cls, 0, 0],
                  [0, 0, 0, r_opn, 0, r_cls, -r_cls, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1]])
    Ainv = np.linalg.inv(A)
    inhomgeneity = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    p0s = Ainv.dot(inhomgeneity)
    return p0s


def f_from_k_invert_A(k, r_ref, r_opn, r_cls, m, n):
    A = np.array([[-n * r_ref, 0, 0, 0, 0, 0, 0, r_cls],
                  [n * r_ref, -n * r_ref, 0, 0, 0, 0, 0, 0],
                  [0, n * r_ref, -n * r_ref, 0, 0, 0, 0, 0],
                  [0, 0, n * r_ref, -n * r_opn, 0, 0, 0, 0],
                  [0, 0, 0, r_opn, -r_cls, 0, 0, 0],
                  [0, 0, 0, r_opn, r_cls, -r_cls, 0, 0],
                  [0, 0, 0, r_opn, 0, r_cls, -r_cls, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1]])
    Ainv = np.linalg.inv(A)
    p0s = steady_states_theory_invert_A(r_ref, r_opn, r_cls, m, n)
    p0s[-1] = 0
    p0s = np.asarray(p0s)
    deltas = np.array([delta(k, 0), delta(k, 1), delta(k, 2), delta(k, 3), delta(k,4), delta(k, 5), delta(k, 6), 0])
    inhomgeneity = np.subtract(p0s, deltas)
    f_from_k = Ainv.dot(inhomgeneity)
    return f_from_k


def mean_puff_single(x):
    m = 4
    n = 4
    r_opn = 0.13 * np.power(x / 0.33, 3) * ((1 + 0.33 ** 3) / (1 + x ** 3))
    r_ref = 1.3 * np.power(x / 0.33, 3) * ((1 + 0.33 ** 3) / (1 + x ** 3))
    r_cls = 50

    p0s = steady_states_theory_invert_A(r_ref, r_opn, r_cls, m, n)

    xs = [0, 0, 0, 0, 4, 3, 2, 1]
    mean = sum([x * p for x, p in zip(xs, p0s)])
    return mean


def means_puff(N, tau, j):
    fs = []
    cas = np.linspace(0.01, 1.00, 100)
    for ca in cas:
        f = -(ca - 0.33)/tau + j*N*mean_puff_single(ca)
        fs.append(f)
    return fs


def intensity_puff_single(x):
    m = 4
    n = 4
    r_opn = 0.13 * np.power(x / 0.33, 3) * ((1. + 0.33 ** 3) / (1. + x ** 3))
    r_ref = 1.3 * np.power(x / 0.33, 3) * ((1. + 0.33 ** 3) / (1. + x ** 3))
    r_cls = 50

    xs = [0, 0, 0, 0, 4, 3, 2, 1]
    idxs = [0, 1, 2, 3, 4, 5, 6, 7]
    p0s = steady_states_theory_invert_A(r_ref, r_opn, r_cls, m, n)

    D_theory = 0
    for k in idxs:
        sum_over_i = 0
        f_from_k_to = f_from_k_invert_A(k, r_ref, r_opn, r_cls, m, n)
        for i in idxs:
            sum_over_i += xs[i] * f_from_k_to[i]
        D_theory += xs[k] * p0s[k] * sum_over_i
    return D_theory


def d_func(x, j, N):
    if x == 0:
        return 0
    else:
        return np.power(j, 2)*N*intensity_puff_single(x)


def f_func(x, tau, j, N):
    if x == 0:
        return -(x - 0.33) / tau
    else:
        return -(x - 0.33) / tau + j * N * mean_puff_single(x)


def g_func(x, tau, j, N):
    return f_func(x, tau, j, N)/d_func(x, j, N)


def h_func(x, tau, j, N):
    h = quad(g_func, 0.33, x, args=(tau, j, N))[0]
    return h


def firing_rate_no_adap(tau, j, N):
    cas_theory = np.linspace(0.30, 1, 701)
    dca = cas_theory[1] - cas_theory[0]
    p0s_theo_ca = []
    integral = 0

    for ca in reversed(cas_theory[1:]):
        print(ca)
        h = h_func(ca, tau, j, N)
        d = d_func(ca, j, N)
        if ca == 1:
            integral += 0
        elif ca >= 0.33:
            integral += np.exp(-h)*dca
        p0s_theo_ca.append(integral * np.exp(h) / d)

    norm = np.sum(p0s_theo_ca) * dca
    r0 = 1 / norm
    return r0


def k_corr(data1, data2, k):
    # Get two arbitrary data set and calculate their correlation with lag k.
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    data1 = [x - mu1 for x in data1]
    data2 = [x - mu2 for x in data2]
    k_corr = 0
    if k == 0:
        for x, y in zip(data1, data2):
            k_corr += x * y
    else:
        for x, y in zip(data1[:-k], data2[k:]):
            k_corr += x * y
    return k_corr / len(data2[k:])


def fourier_transformation_isis(w, isis):
    t = 0
    f = 0
    for isi in isis:
        t += isi
        f += np.exp(1j*w*t)
    return f


def set_default_plot_style():
    rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Latin Modern'
    rc('text', usetex=True)


def remove_top_right_axis(axis):
    for ax in axis:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
