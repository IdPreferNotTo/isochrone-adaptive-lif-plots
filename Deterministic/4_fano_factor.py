import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os


def spike_count_in_T(data, T_fano):
    T = 0
    Tmax_trace = sum(data)
    n_max = int(Tmax_trace/T_fano)
    spike_counts = []
    spike_count = 0
    isi_idx = 0

    T += data[isi_idx]
    for n in range(n_max):
        T_start = n * T_fano
        T_stop = (n+1) * T_fano
        if T_start< T and T < T_stop:
            spike_count += 1
            isi_idx += 1
            T += data[isi_idx]
            while T < (n+1)*T_fano:
                spike_count += 1
                isi_idx += 1
                T += data[isi_idx]
        spike_counts.append(spike_count)
        spike_count = 0

    return spike_counts



if __name__ == "__main__":
    home = os.path.expanduser("~")
    phase = 2*np.pi/2 #1.57, 3.14, 4.71
    D = 0.01
    mu = 2.0
    #ISIs_thr = np.loadtxt(home + "/Data/isochrones/ISI_thr_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    #ISIs_iso = np.loadtxt(home + "/Data/isochrones/ISI_iso_D{:.1f}_phi{:.2f}.dat".format(D, phase))
    ISIss = []
    IPIss = []
    for run in range(10):
        ISIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/ISIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
        data_IPIs = np.loadtxt(home + "/CLionProjects/PhD/alif_pass_isochrone/out/IPIs_alif_mu{:.1f}_taua2.0_delta1.0_D{:.2f}_phase{:.2f}_run{:d}.dat".format(mu, D, phase, run))
        IPIs, v_pass, a_pass = np.transpose(data_IPIs)

        ISIss.append(ISIs)
        IPIss.append(IPIs)



    Ts = np.logspace(-1, 3, 5000)
    Fano_ISI = []
    Fano_IPI = []
    for T in Ts:
        print(T)
        spike_count_ISI = spike_count_in_T(ISIss[1], T)
        spike_count_IPI = spike_count_in_T(IPIss[1], T)

        print(np.var(spike_count_ISI), np.mean(spike_count_ISI))
        Fano_ISI.append(np.var(spike_count_ISI)/np.mean(spike_count_ISI))
        Fano_IPI.append(np.var(spike_count_IPI)/np.mean(spike_count_IPI))

    fig = plt.figure(tight_layout=True)
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:])
    ax.plot(Ts, Fano_ISI, c="#482677FF", label="Threshold")
    ax.plot(Ts, Fano_IPI, c="#29AF7FFF", label="Isochrone")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T")
    ax.set_ylabel("F(T)")
    ax.set_xlim([0.1, 1000])
    ax.legend(framealpha=1., edgecolor="k", fancybox = False)


    CV_IPI = np.var(IPIss[1])/np.power(np.mean(IPIss[1]),2)
    ax.axhline(CV_IPI, ls=":", c="k")

    plt.savefig(home + "/Data/isochrones/Fano_factor_mu{:.1f}_D{:.2f}_phi{:.2f}.pdf".format(mu, D, phase), transparent=True)
    plt.show()
