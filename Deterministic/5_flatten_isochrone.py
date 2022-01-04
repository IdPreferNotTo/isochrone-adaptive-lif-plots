import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os


if __name__ == "__main__":
    home  = os.path.expanduser("~")
    folder = "/CLionProjects/PhD/alif_deterministic_isochrone/out/"
    file = "isochrone_mu2.0_taua2.0_delta1.0_phase6.28.dat"
    isochrone = np.loadtxt(home + folder + file)

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

    v_iso_flat = []
    a_iso_flat = []

    v_lc = 1.0
    a_lc = 0.577359
    delta_a = 1.
    vs = np.linspace(-5, 1, 61)
    for i in range(3):
        for v in vs:
            v_iso_flat.append(v)
            a_iso_flat.append(a_lc + i*delta_a)

    fig = plt.figure(figsize=(4, 2*8/3), tight_layout=True)
    gs = gs.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    ax.plot(v_isos[0], a_isos[0])
    ax.plot(v_isos[1], a_isos[1])
    ax.plot(v_iso_flat, a_iso_flat)

    with open(home + folder + "isochrone_very_flat_mu2.0_taua2.0_delta1.0_phase6.28.dat", "w") as f:
        for v, a in zip(v_iso_flat, a_iso_flat):
            f.write(f"{v} {a} \n")
    plt.show()