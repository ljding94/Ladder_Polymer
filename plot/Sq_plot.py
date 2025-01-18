from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import os
from config_plot import *
from PIL import Image


def calc_Sq_discrete_infinite_thin_rod(q, L):
    # numereical calculation
    Sq = [1.0 / L for i in range(len(q))]
    for k in range(len(q)):
        Sqk = 0
        qk = q[k]
        for i in range(L - 1):
            for j in range(i + 1, L):
                Sqk += 2.0 * np.sin(qk * (i - j)) / (qk * (i - j)) / (L * L)
        Sq[k] += Sqk
    return np.array(Sq)


# consistent with ML analysis version
def read_Delta_Sq_data(folder, parameters, L0=200):
    # normalized againt L0
    all_features = []
    all_Sq = []
    all_Delta_Sq = []
    all_Delta_Sq_err = []
    q = []
    Sq_rod_discrete = []
    all_filename = []
    if len(parameters[0]) == 2:
        for segment_type, run_num in parameters:
            all_filename.append(f"{folder}/obs_{segment_type}_random_run{run_num}_SqB.csv")
    else:
        for segment_type, lnLmu, lnLsig, Kt, Kb, Rf in parameters:
            all_filename.append(f"{folder}/obs_MC_{segment_type}_lnLmu{lnLmu:.1f}_lnLsig{lnLsig:.2f}_Kt{Kt:.1f}_Kb{Kb:.1f}_Rf{Rf:.2f}_SqB.csv")
    for filename in all_filename:
        print("reading: ", filename)
        if os.path.exists(filename):
            Sqdata = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=1)
            if len(Sqdata) == 0:
                print(f"Warning: File {filename} is empty. Skiped")
                continue
            features = Sqdata[2:11]
            features = np.insert(features, 9, features[7] / (features[6] * features[6]))  # add L^2/(L)^2 = PDI
            # Sq, Sq_err, q = Sqdata[0, 8:], Sqdata[1, 8:], Sqdata[2, 8:]
            qdata = np.genfromtxt(filename, delimiter=",", skip_header=3, max_rows=1)
            Sq, q = Sqdata[11:], qdata[11:]
            print("Sq", Sq)
            # Sq_rod_discrete = Sqdata[3, 7:]
            if len(Sq_rod_discrete) == 0:
                Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(q, L0)
            # normalize Sq by Sq_rod_discrete with L0
            Delta_Sq = np.log(Sq / Sq_rod_discrete)
            # Delta_Sq = Sq
            # Delta_Sq_err = Sq_err/Sq_rod_discrete
            all_features.append(features)
            all_Sq.append(Sq)
            all_Delta_Sq.append(Delta_Sq)
            # all_Delta_Sq_err.append(Delta_Sq_err)
        else:
            print(f"Warning: File {filename} not found. Skiped")

    return segment_type, np.array(all_features), all_Sq, all_Delta_Sq, all_Delta_Sq_err, q


def read_Sq_data(folder, parameters):
    # normalized againt L0
    all_features = []
    all_Sq = []
    all_Sq_err = []
    q = []
    all_filename = []
    if len(parameters[0]) == 2:
        for segment_type, run_num in parameters:
            all_filename.append(f"{folder}/obs_{segment_type}_random_run{run_num}_SqB.csv")
    else:
        for segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha in parameters:
            all_filename.append(f"{folder}/obs_MC_{segment_type}_lnLmu{lnLmu:.2f}_lnLsig{lnLsig:.0f}_Kt{Kt:.0f}_Kb{Kb:.0f}_Rf{Rf:.1f}_alpha{alpha:.2f}_SqB.csv")

    for filename in all_filename:
        if os.path.exists(filename):
            Sqdata = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=1)
            if len(Sqdata) == 0:
                print(f"Warning: File {filename} is empty. Skiped")
                continue
            features = Sqdata[2:12]
            features = np.insert(features, 10, features[9] / (features[8] * features[8]))
            qdata = np.genfromtxt(filename, delimiter=",", skip_header=3, max_rows=1)
            Sq, q = Sqdata[12:], qdata[12:]

            all_features.append(features)
            all_Sq.append(Sq)
        else:
            print(f"Warning: File {filename} not found. Skiped")

    # print("all_Sq.shape", np.array(all_Sq).shape)
    all_feature_names = ["lnLmu", "lnLsig", "Kt", "Kb", "Rf", "alpha", "Rg2", "wRg2", "L", "L2", "PDI"]
    return segment_type, np.array(all_features), all_feature_names, all_Sq, all_Sq_err, q


def read_Delta_Sq_data_detail(folder, parameter):
    all_L = []
    all_Sq = []
    q = []
    segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameter
    filename = f"{folder}/obs_MC_{segment_type}_lnLmu{lnLmu:.1f}_lnLsig{lnLsig:.2f}_Kt{Kt:.1f}_Kb{Kb:.1f}_Rf{Rf:.2f}_SqB_detail.csv"

    print("reading: ", filename)
    if os.path.exists(filename):
        q = np.genfromtxt(filename, delimiter=",", skip_header=7, max_rows=1)[2:]
        Sqdata = np.genfromtxt(filename, delimiter=",", skip_header=9)
        all_Sq = Sqdata[:, 2:]
        all_L = Sqdata[:, 1]
    else:
        print(f"Warning: File {filename} not found. Skiped")

    return q, all_L, all_Sq


def plot_monodisperse_polymer_Sq(tex_lw=240.71031, ppi=72):

    # map Rf and L to line color and style
    # LS = {50: "-", 70: "--", 90: "-."}
    # LC = {0.4: "tomato", 0.5: "lawngreen", 0.6: "steelblue"}  # black for hard rod
    colors = ["tomato", "gold", "lawngreen", "steelblue"]
    segment_type_map = {"outofplane_twist": "2D CANAL", "inplane_twist": "3D CANAL"}

    folder = "../data/scratch_local/20241014"
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.45))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(133, sharex=ax1, sharey=ax1)

    # plot variation of L
    parameters = [
        ["outofplane_twist", 1.61, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 2.31, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.69, 0, 100, 100, 0.5, 0.93],
    ]

    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)
    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[i]
        L = all_features[i][8]
        ax1.loglog(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${L:.0f}$")
        # ax_inset00.semilogx(q, np.log(all_Sq[i])-np.log(Sq_rod_L40), "-", lw=1, color=colors[i], label=rf"${L:.0f}$")
    ax1.legend(title=r"$L$", loc="lower left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variation of Rf
    parameters = [
        ["outofplane_twist", 3.00, 0, 100, 100, 0.1, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.3, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.7, 0.93],
    ]

    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)
    # ax_inset01 = axs[0, 1].inset_axes([0.15, 0.15, 0.4, 0.4])
    # ax_inset01.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7)
    # ax_inset01.set_ylabel(r"$\Delta S(QB)$", fontsize=7)
    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[i]
        L = all_features[i][8]
        ax2.loglog(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${Rf:.1f}$")
        # ax_inset01.semilogx(q, np.log(all_Sq[i])-np.log(Sq_rod_L40), "-", lw=1, color=colors[i], label=rf"${L:.0f}$")

    ax2.legend(title=r"$R_a$", loc="lower left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variation of alpha

    parameters = [
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.79],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.86],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 1.00],
    ]

    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)

    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[i]
        L = all_features[i][8]
        ax3.loglog(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${alpha:.2f}$")
    ax3.legend(title=r"$\alpha$", loc="lower left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    ax1.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax2.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
    ax3.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
    ax1.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$S(QB)$", fontsize=9, labelpad=0)
    # ax1.tick_params(axis='y', labelrotation=45)

    ax2.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax3.set_xlabel(r"$QB$", fontsize=9, labelpad=0)

    ax1.text(0.75, 0.1, r"$(a)$", transform=ax1.transAxes, fontsize=9, ha="center")
    ax2.text(0.75, 0.1, r"$(b)$", transform=ax2.transAxes, fontsize=9, ha="center")
    ax3.text(0.75, 0.1, r"$(c)$", transform=ax3.transAxes, fontsize=9, ha="center")

    plt.tight_layout(pad=0.2)
    # plt.show()
    plt.savefig("figures/polymer_Sq.pdf", format="pdf")
    plt.savefig("figures/polymer_Sq.png", format="png", dpi=300)
    plt.show()
    plt.close()


def plot_monodisperse_polymer_Sq_config(tex_lw=240.71031, ppi=72):

    # map Rf and L to line color and style
    # LS = {50: "-", 70: "--", 90: "-."}
    # LC = {0.4: "tomato", 0.5: "lawngreen", 0.6: "steelblue"}  # black for hard rod
    colors = ["tomato", "gold", "lawngreen", "steelblue"]
    segment_type_map = {"outofplane_twist": "2D CANAL", "inplane_twist": "3D CANAL"}

    folder = "../data/scratch_local/20241014"
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1.5))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    axc1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
    axc2 = plt.subplot2grid((3, 3), (0, 1), rowspan=2, sharex=axc1, sharey=axc1)
    axc3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharex=axc1, sharey=axc1)

    ax1 = plt.subplot2grid((3, 3), (2, 0))
    ax2 = plt.subplot2grid((3, 3), (2, 1), sharex=ax1, sharey=ax1)
    ax3 = plt.subplot2grid((3, 3), (2, 2), sharex=ax1, sharey=ax1)

    # plot variation of L
    parameters = [
        ["outofplane_twist", 1.61, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 2.31, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.69, 0, 100, 100, 0.5, 0.93],
    ]

    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)
    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[i]
        filename = f"{folder}/config_{segment_type}_lnLmu{lnLmu:.2f}_lnLsig{lnLsig:.0f}_Kt{Kt:.0f}_Kb{Kb:.0f}_Rf{Rf:.1f}_alpha{alpha:.2f}.csv"
        ax_plot_sample_MC_config_2d(axc1, filename, [0, i * 30, 0], [0, 0, 1.7])

        L = all_features[i][8]

        ax1.loglog(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${L:.0f}$")
        # ax_inset00.semilogx(q, np.log(all_Sq[i])-np.log(Sq_rod_L40), "-", lw=1, color=colors[i], label=rf"${L:.0f}$")
    ax1.legend(title=r"$L$", loc="lower left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)
    axc1.set_axis_off()
    axc2.set_axis_off()
    axc3.set_axis_off()
    # plot variation of Rf
    parameters = [
        ["outofplane_twist", 3.00, 0, 100, 100, 0.1, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.3, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.7, 0.93],
    ]

    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)
    # ax_inset01 = axs[0, 1].inset_axes([0.15, 0.15, 0.4, 0.4])
    # ax_inset01.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7)
    # ax_inset01.set_ylabel(r"$\Delta S(QB)$", fontsize=7)
    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[i]
        L = all_features[i][8]
        ax2.loglog(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${Rf:.1f}$")
        # ax_inset01.semilogx(q, np.log(all_Sq[i])-np.log(Sq_rod_L40), "-", lw=1, color=colors[i], label=rf"${L:.0f}$")

    ax2.legend(title=r"$R_a$", loc="lower left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variation of alpha

    parameters = [
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.79],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.86],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93],
        ["outofplane_twist", 3.00, 0, 100, 100, 0.5, 1.00],
    ]

    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)

    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[i]
        L = all_features[i][8]
        ax3.loglog(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${alpha:.2f}$")
    ax3.legend(title=r"$\alpha$", loc="lower left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    ax1.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax2.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
    ax3.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
    ax1.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$S(QB)$", fontsize=9, labelpad=0)
    # ax1.tick_params(axis='y', labelrotation=45)

    ax2.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax3.set_xlabel(r"$QB$", fontsize=9, labelpad=0)

    ax1.text(0.75, 0.1, r"$(a)$", transform=ax1.transAxes, fontsize=9, ha="center")
    ax2.text(0.75, 0.1, r"$(b)$", transform=ax2.transAxes, fontsize=9, ha="center")
    ax3.text(0.75, 0.1, r"$(c)$", transform=ax3.transAxes, fontsize=9, ha="center")

    plt.tight_layout(pad=0.2)
    plt.savefig("figures/polymer_Sq_config.pdf", format="pdf")
    plt.savefig("figures/polymer_Sq_config.png", format="png", dpi=300)
    plt.show()
    plt.close()


def plot_polydisperse_polymer_Sq(tex_lw=240.71031, ppi=72):

    # map Rf and L to line color and style
    # LS = {50: "-", 70: "--", 90: "-."}
    # LC = {0.4: "tomato", 0.5: "lawngreen", 0.6: "steelblue"}  # black for hard rod
    colors = ["tomato", "gold", "lawngreen", "steelblue"]
    segment_type_map = {"outofplane_twist": "2D CANAL", "inplane_twist": "3D CANAL"}

    folder = "../data/scratch_local/20240916"
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    axs = fig.subplots(2, 2, sharex=True, sharey=True)

    # plot distribution of L
    parameters = [["outofplane_twist", 3.0, 0.50, 20.0, 20.0, 0.500]]
    segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameters[0]
    q, all_L, all_Sq = read_Delta_Sq_data_detail(folder, parameters[0])
    max_L = np.max(all_L)
    for i in range(len(all_L))[::100]:
        axs[0, 0].loglog(q, all_Sq[i], "-", lw=0.5, alpha=0.5, color="royalblue")  # , alpha=0.9*(all_L[i]/max_L)**2)
    axs[0, 0].loglog(q, all_Sq[0], "-", lw=1, alpha=0.5, color="royalblue", label=r"$S_L(QB)$")
    print("hello")
    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    print("np.shape(all_Sq)", np.shape(all_Sq))
    axs[0, 0].loglog(q, all_Sq[0], "-", lw=1, color="black", label=r"$I(QB)$")
    # Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(q, 200)
    # axs[0, 0].loglog(q, Sq_rod_discrete, ":", lw=1, color="black", label=r"rod; $L=200$")
    axs[0, 0].tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7)
    # axs[0, 0].set_xlabel("QB", fontsize=10)
    axs[0, 0].set_ylabel(r"$I(QB)$", fontsize=9, labelpad=0)

    axs[0, 0].legend(loc="center left", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variation of Rf
    parameters = [
        ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.400],
        ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.500],
        ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.600],
        ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.700],
    ]

    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    print("np.shape(all_Sq)", np.shape(all_Sq))

    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameters[i]
        axs[0, 1].semilogx(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${Rf}$")
    axs[0, 1].tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=False, labelsize=7)
    # axs[0, 1].set_ylabel(r"$\Delta S(QB)$", fontsize=10)
    axs[0, 1].legend(title=r"$R_f$", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variation of lnLmu

    parameters = [["outofplane_twist", 2.5, 0.5, 20.0, 20.0, 0.500], ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.500], ["outofplane_twist", 3.5, 0.5, 20.0, 20.0, 0.500]]

    # all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)
    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)

    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameters[i]
        axs[1, 0].semilogx(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${lnLmu}$")
    axs[1, 0].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    axs[1, 0].set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    axs[1, 0].set_ylabel(r"$I(QB)$", fontsize=9, labelpad=0)
    axs[1, 0].legend(title=r"$\mu_{(\ln{L})}$", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variaotion of siglnL
    parameters = [["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.500], ["outofplane_twist", 3.0, 0.7, 20.0, 20.0, 0.500], ["outofplane_twist", 3.0, 0.9, 20.0, 20.0, 0.500]]

    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameters[i]
        axs[1, 1].semilogx(q, all_Sq[i], "-", lw=1, color=colors[i], label=f"{lnLsig:.1f}")
    axs[1, 1].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
    axs[1, 1].set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    # axs[1, 1].set_ylabel(r"$S\Delta (QB)$", fontsize=10)
    axs[1, 1].legend(title=r"$\sigma_{(\ln{L})}$", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)
    # axs[1, 1].legend(ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.1, frameon=False, fontsize=10)

    plt.tight_layout(pad=0.15)
    # plt.show()
    plt.savefig("figures/Sq.pdf", format="pdf")
    plt.close()


# Sq fitted Rg vs. MC reference


def ax_fit(x, a):
    return a * x


def fit_Rg2(q, Sq):
    popt, pcov = curve_fit(ax_fit, q * q / 3, (1 - Sq))
    perr = np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def plot_Sq_fitted_Rg2(tex_lw=455.24408, ppi=72):

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 1))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    axs = fig.subplots(2, 1)

    folder = "../data/scratch_local/20240827"
    parameters = [["inplane_twist", 3.0, 0.8, 20.0, 20.0, 0.50], ["outofplane_twist", 3.0, 0.8, 20.0, 20.0, 0.50]]
    # ["flat", 3.0, 0.8, 20.0, 20.0, 0.50]]
    # parameters = [["inplane_twist", 3.0, 0, 20.0, 20.0, 0.50],
    #              ["outofplane_twist", 3.0, 0, 20.0, 20.0, 0.50],
    #              ["flat", 3.0, 0, 20.0, 20.0, 0.50]]
    # segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameters[0]

    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    print("np.shape(all_Sq)", np.shape(all_Sq))
    print("all_features", all_features)
    all_feature_names = ["lnLmu", "lnLsig", "Kt", "Kb", "Rf", "Rg2", "wRg2", "L", "L2", "PDI"]
    for i in range(len(parameters)):
        axs[0].plot(
            q[:], all_Sq[i][:], "o", ms=3, mfc="None", label=r"$S(QB)$" + f"MC Rg2={all_features[:, all_feature_names.index("Rg2")][i]}, wRg2={all_features[:, all_feature_names.index("wRg2")][i]}"
        )
        for qfn in [20, 40, 80]:
            Rg2, Rg2_err = fit_Rg2(q[:qfn], all_Sq[i][:qfn])
            axs[0].plot(q[:qfn], 1 - 1 / 3 * Rg2 * q[:qfn] ** 2, "-", lw=1, label=f"qf={q[qfn-1]}, Rg2={Rg2:.2f}" + r"$\pm$" + f"{Rg2_err:.2f}")

    axs[0].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    axs[0].set_xlabel("QB", fontsize=10)
    axs[0].set_ylabel(r"$S(QB)$", fontsize=10)
    axs[0].legend(ncol=1, columnspacing=0.5)
    plt.savefig("figures/Sq_fitted_Rg2.pdf", format="pdf")
    plt.show()


# following not used for manusctip


def plot_polymer_Sq_demo():
    # demo plot for slices
    folder = "../data/20240611"
    parameters = [["outofplane_twist", 98, 1.5, 1.5, 0.400], ["outofplane_twist", 98, 1.5, 1.5, 0.500], ["outofplane_twist", 98, 1.5, 1.5, 0.600]]
    colors = ["tomato", "lawngreen", "steelblue"]

    all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)
    # all_feature_names = ["L", "Kt", "Kb", "Rf", "Rg"] # for reference

    ppi = 72
    fig = plt.figure(figsize=(246 / ppi * 1, 246 / ppi * 0.8))
    ax = fig.subplots(1, 1)

    n = 20  # skip first few data
    ax.loglog(q[n:], all_Sq_rod[0][n:], "--", color="black", label="discrete rod")
    for i in range(len(parameters)):
        ax.loglog(q[n:], all_Sq[i][n:], color=colors[i], label=f"Rf={parameters[i][-1]}")
    ax.set_xlabel("q", fontsize=10)
    ax.set_ylabel("S(q)", fontsize=10)

    axin = ax.inset_axes([0.15, 0.4, 0.4, 0.4])
    for i in range(len(parameters)):
        axin.semilogx(q[n:], all_Delta_Sq[i][n:], color=colors[i])
    axin.set_xlabel("q", fontsize=5)
    axin.set_ylabel(r"$\Delta$S(q)= S(q)/S_rod(q)", fontsize=5)
    axin.tick_params(axis="both", labelsize=5)
    # axin.set_yticks([1, 2, 3])

    plt.legend(ncol=2, columnspacing=0.5, handlelength=0.5, handletextpad=0.1, frameon=False, fontsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig("figures/Sq_demo.png", dpi=300)
    plt.close()


# bonus plot, write it here just for convinence since it need to read Sq data
def plot_Rg_L_relation():
    folder = "../data/20240611"
    segment_type = "outofplane_twist"
    Ls = np.arange(50, 98.1, 1)
    logKts = [2.00]
    logKbs = [2.00]
    Rfs = np.arange(0.40, 0.601, 0.1)

    ppi = 72
    fig = plt.figure(figsize=(246 / ppi * 1, 246 / ppi * 0.8))
    ax = fig.subplots(1, 1)

    for Rf in Rfs:
        parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs]
        all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)
        ax.plot(all_features[:, 0], all_features[:, -1], "o-", markerfacecolor="none", markersize=3, label=f"Rf={Rf}")
        # ax.loglog(all_features[:, 0], all_features[:, -1], "o-", markerfacecolor="none", markersize=3, label=f"Rf={Rf}")
    ax.set_xlabel("L", fontsize=10)
    ax.set_ylabel("Rg", fontsize=10)
    plt.legend(ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.1, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/Rg_L.png", dpi=300)
    plt.close()


def plot_exp_ML_fitting(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.6))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    ax = fig.subplots(1, 1)

    # get exp data

    exp_filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq_subtracted.txt"
    # exp_filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_n65_patch_smooth_normalized_Iq.txt"

    QB, I_exp, I_exp_err = np.genfromtxt(exp_filename, delimiter=",", skip_header=1, unpack=True)
    QBi, QBf = 0.07, 3
    # Filter the experimental data to only include QB values within the specified range
    mask = (QB > QBi) & (QB < QBf)
    QB_filtered = QB[mask]
    I_exp_filtered = I_exp[mask]
    I_exp_err_filtered = I_exp_err[mask]

    # Guinier normalization
    A = 0.03736901
    # Plot the filtered experimental data
    ax.errorbar(QB_filtered, I_exp_filtered / A, yerr=I_exp_err_filtered / A, linestyle="None", label="SANS measured", capsize=2)
    # ax.errorbar(QB, I_exp / A, yerr=I_exp_err / A, linestyle="None", label="SANS measured", capsize=1)

    # MC calculated

    filename_up = "../data/scratch_local/20241012/obs_MC_outofplane_twist_lnLmu2.19722_lnLsig0.0_Kt100_Kb100_Rf0.07_alpha53.9_SqB.csv"
    filename_mid = "../data/scratch_local/20241012/obs_MC_outofplane_twist_lnLmu2.48491_lnLsig0.0_Kt100_Kb100_Rf0.14_alpha51.0_SqB.csv"
    filename_down = "../data/scratch_local/20241012/obs_MC_outofplane_twist_lnLmu2.63806_lnLsig0.0_Kt100_Kb100_Rf0.21_alpha48.1_SqB.csv"

    # print("reading: ", filename)
    Sqs = []
    filenames = [filename_down, filename_mid, filename_up]
    for i in range(len(filenames)):
        filename = filenames[i]
        if os.path.exists(filename_up):
            Sqdata = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=1)
            if len(Sqdata) == 0:
                print(f"Warning: File {filename} is empty. Skipped")
            features = Sqdata[2:12]
            features = np.insert(features, 10, features[9] / (features[8] * features[8]))  # add L^2/(L)^2 = PDI
            qdata = np.genfromtxt(filename, delimiter=",", skip_header=3, max_rows=1)
            Sq, q = Sqdata[12:], qdata[12:]
            Sqs.append(Sq)

    ax.loglog(q, Sqs[1], "-", color="black", alpha=0.5)
    ax.fill_between(q, Sqs[0], Sqs[2], color="gray", alpha=0.5, label="ML implied")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax.set_xlabel(r"$QB$", fontsize=9, labelpad=-2)
    ax.set_ylabel(r"$S(QB)$", fontsize=9, labelpad=-2)
    ax.legend(ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    ax_inset = ax.inset_axes([0.3, 0.25, 0.4, 0.4])
    arr_img = plt.imread("./figures/chem_structure.png")
    ax_inset.imshow(arr_img, zorder=-1)
    ax_inset.set_axis_off()

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/exp_ML_fitting.pdf", format="pdf", dpi=300)
    plt.savefig("figures/exp_ML_fitting.png", dpi=300)

    plt.show()
    plt.close()


def plot_exp_ML_fitting_TOC(tex_lw=240.71031, ppi=72):
    cm = 1 / 2.54
    fig = plt.figure(figsize=(8.25 * cm, 4.45 * cm))
    # fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.8))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=3)
    #axcfg = plt.subplot2grid((1, 3), (0, 2))

    # get exp data

    exp_filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq_subtracted.txt"
    # exp_filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_n65_patch_smooth_normalized_Iq.txt"

    QB, I_exp, I_exp_err = np.genfromtxt(exp_filename, delimiter=",", skip_header=1, unpack=True)
    QBi, QBf = 0.07, 3
    # Filter the experimental data to only include QB values within the specified range
    mask = (QB > QBi) & (QB < QBf)
    QB_filtered = QB[mask]
    I_exp_filtered = I_exp[mask]
    I_exp_err_filtered = I_exp_err[mask]

    # Guinier normalization
    A = 0.03736901
    # Plot the filtered experimental data
    ax.errorbar(QB_filtered, I_exp_filtered / A, yerr=I_exp_err_filtered / A, linestyle="None", label="SANS measured", capsize=2, elinewidth=1)
    # ax.errorbar(QB, I_exp / A, yerr=I_exp_err / A, linestyle="None", label="SANS measured", capsize=1)

    # MC calculated

    filename_up = "../data/scratch_local/20241012/obs_MC_outofplane_twist_lnLmu2.19722_lnLsig0.0_Kt100_Kb100_Rf0.07_alpha53.9_SqB.csv"
    filename_mid = "../data/scratch_local/20241012/obs_MC_outofplane_twist_lnLmu2.48491_lnLsig0.0_Kt100_Kb100_Rf0.14_alpha51.0_SqB.csv"
    filename_down = "../data/scratch_local/20241012/obs_MC_outofplane_twist_lnLmu2.63806_lnLsig0.0_Kt100_Kb100_Rf0.21_alpha48.1_SqB.csv"

    # print("reading: ", filename)
    Sqs = []
    filenames = [filename_down, filename_mid, filename_up]
    for i in range(len(filenames)):
        filename = filenames[i]
        if os.path.exists(filename_up):
            Sqdata = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=1)
            if len(Sqdata) == 0:
                print(f"Warning: File {filename} is empty. Skipped")
            features = Sqdata[2:12]
            features = np.insert(features, 10, features[9] / (features[8] * features[8]))  # add L^2/(L)^2 = PDI
            qdata = np.genfromtxt(filename, delimiter=",", skip_header=3, max_rows=1)
            Sq, q = Sqdata[12:], qdata[12:]
            Sqs.append(Sq)

    ax.loglog(q, Sqs[1], "-", color="black", lw=1, alpha=0.5)
    ax.fill_between(q, Sqs[0], Sqs[2], color="gray", alpha=0.5, label="ML implied")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax.set_xlabel(r"$QB$", fontsize=7, labelpad=-2)
    ax.set_ylabel(r"$S(QB)$", fontsize=7, labelpad=-2)
    ax.legend(ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot sample config
    # ax_inset = ax.inset_axes([0.1, 0.1, 0.5, 0.5])
    # filename_mid = "/Users/ldq/Work/Ladder_Polymer/data/scratch_local/20241012/config_outofplane_twist_lnLmu2.48491_lnLsig0.0_Kt100_Kb100_Rf0.14_alpha51.0.csv"
    filename_mid = "../data/scratch_local/20241014/outofplane_twist_lnLmu2.48491_lnLsig0.0_Kt100_Kb100_Rf0.14_alpha0.89/polymer_43.csv"

    # filename = "../data/scratch_local/20240913/config_outofplane_twist_lnLmu2.5_lnLsig0.0_Kt30.0_Kb30.0_Rf0.1.csv"

    # ax_plot_sample_MC_config_2d(ax_inset, filename_down, [0, 0, 0], [0, -0.5, 0])
    # ax_plot_sample_MC_config_2d(ax_inset, filename_mid, [20, 0, 0], [-0.4, -0.5, 0])
    axcfg= ax.inset_axes([0.27, 0.27, 0.4, 0.4])
    ax_plot_sample_MC_config_2d(axcfg, filename_mid, [0, 0, 0], [0, 0, 0.0])
    # ax_plot_sample_MC_config_2d(ax_inset, filename_up, [40, 0, 0], [0, 0.5, 0])
    axcfg.set_aspect("equal")
    axcfg.set_axis_off()

    ax_inset = ax.inset_axes([0.05, 0.27, 0.35, 0.35])
    #arr_image = Image.open("./figures/chem_structure.pdf")
    arr_img = plt.imread("./figures/chem_structure.png")
    ax_inset.imshow(arr_img, zorder=-10)
    ax_inset.set_axis_off()


    plt.tight_layout(pad=0.1)
    plt.savefig("figures/TOC.pdf", format="pdf", dpi=300)
    plt.savefig("figures/TOC.png", dpi=300)

    plt.show()
    plt.close()


def plot_exp_temp(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.6))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    ax = fig.subplots(1, 1)

    # get exp data

    #exp_background_filename = "../data/incoh_banjo_expdata/L2_temp/merged_incoh_L1_Iq.txt"
    # background is already substracted fro the following sent by Zhiqiang
    exp_25C_filename = "../data/incoh_banjo_expdata/L2_temp/merged_inelsub_L2 25C_Iq.txt"
    exp_75C_filename = "../data/incoh_banjo_expdata/L2_temp/merged_inelsub_L2 75C_Iq.txt"
    exp_125C_filename = "../data/incoh_banjo_expdata/L2_temp/merged_inelsub_L2 125C_Iq.txt"
    # exp_filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_n65_patch_smooth_normalized_Iq.txt"

    #Q, I_exp, I_exp_err, dQ = np.genfromtxt(exp_filename, delimiter=",", skip_header=2, unpack=True)
    #QBi, QBf = 0.07, 3
    # Filter the experimental data to only include QB values within the specified range
    #mask = (QB > QBi) & (QB < QBf)
    #QB_filtered = QB[mask]
    #I_exp_filtered = I_exp[mask]
    #I_exp_err_filtered = I_exp_err[mask]

    labels = [r"$25^\circ$C", r"$75^\circ$C", r"$125^\circ$C"]
    exp_filenames = [exp_25C_filename, exp_75C_filename, exp_125C_filename]
    for i in range(len(exp_filenames)):
        exp_filename = exp_filenames[i]
        Q, I_exp, I_exp_err, dQ = np.genfromtxt(exp_filename, delimiter="\t", skip_header=2, unpack=True)
        ax.errorbar(Q, I_exp, yerr=I_exp_err, linestyle="None", label=labels[i], capsize=1.5, elinewidth=0.75)

    #ax.errorbar(QB_filtered, I_exp_filtered / A, yerr=I_exp_err_filtered / A, linestyle="None", label="SANS measured", capsize=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax.set_xlabel(r"$Q (1/\rm{\AA})$", fontsize=9, labelpad=-0)
    ax.set_ylabel(r"$I(Q) (cm^{-1})$", fontsize=9, labelpad=-0)
    ax.legend(ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/exp_Sq_temp.pdf", format="pdf")
    plt.savefig("figures/exp_Sq_temp.png", dpi=300)

    plt.show()
    plt.close()
