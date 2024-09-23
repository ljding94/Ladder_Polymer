from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import os


def calc_Sq_discrete_infinite_thin_rod(q, L):
    # numereical calculation
    Sq = [1.0/L for i in range(len(q))]
    for k in range(len(q)):
        Sqk = 0
        qk = q[k]
        for i in range(L-1):
            for j in range(i+1, L):
                Sqk += 2.0*np.sin(qk*(i-j))/(qk*(i-j))/(L*L)
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
            Sqdata = np.genfromtxt(filename, delimiter=',', skip_header=1, max_rows=1)
            if len(Sqdata) == 0:
                print(f"Warning: File {filename} is empty. Skiped")
                continue
            features = Sqdata[2: 11]
            features = np.insert(features, 9, features[7]/(features[6]*features[6]))  # add L^2/(L)^2 = PDI
            # Sq, Sq_err, q = Sqdata[0, 8:], Sqdata[1, 8:], Sqdata[2, 8:]
            qdata = np.genfromtxt(filename, delimiter=',', skip_header=3, max_rows=1)
            Sq, q = Sqdata[11:], qdata[11:]
            print("Sq", Sq)
            # Sq_rod_discrete = Sqdata[3, 7:]
            if len(Sq_rod_discrete) == 0:
                Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(q, L0)
            # normalize Sq by Sq_rod_discrete with L0
            Delta_Sq = np.log(Sq/Sq_rod_discrete)
            # Delta_Sq = Sq
            # Delta_Sq_err = Sq_err/Sq_rod_discrete
            all_features.append(features)
            all_Sq.append(Sq)
            all_Delta_Sq.append(Delta_Sq)
            # all_Delta_Sq_err.append(Delta_Sq_err)
        else:
            print(f"Warning: File {filename} not found. Skiped")

    return segment_type, np.array(all_features), all_Sq, all_Delta_Sq, all_Delta_Sq_err, q


def read_Delta_Sq_data_detail(folder, parameter):
    all_L = []
    all_Sq = []
    q = []
    segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameter
    filename = f"{folder}/obs_MC_{segment_type}_lnLmu{lnLmu:.1f}_lnLsig{lnLsig:.2f}_Kt{Kt:.1f}_Kb{Kb:.1f}_Rf{Rf:.2f}_SqB_detail.csv"

    print("reading: ", filename)
    if os.path.exists(filename):
        q = np.genfromtxt(filename, delimiter=',', skip_header=7, max_rows=1)[2:]
        Sqdata = np.genfromtxt(filename, delimiter=',', skip_header=9)
        all_Sq = Sqdata[:, 2:]
        all_L = Sqdata[:, 1]
    else:
        print(f"Warning: File {filename} not found. Skiped")

    return q, all_L, all_Sq


def plot_polymer_Sq(tex_lw=240.71031, ppi=72):

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
    parameters = [["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.400],
                  ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.500],
                  ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.600],
                  ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.700]]

    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    print("np.shape(all_Sq)", np.shape(all_Sq))

    for i in range(len(parameters)):
        segment_type, lnLmu, lnLsig, Kt, Kb, Rf = parameters[i]
        axs[0, 1].semilogx(q, all_Sq[i], "-", lw=1, color=colors[i], label=rf"${Rf}$")
    axs[0, 1].tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=False, labelsize=7)
    # axs[0, 1].set_ylabel(r"$\Delta S(QB)$", fontsize=10)
    axs[0, 1].legend(title=r"$R_f$", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    # plot variation of lnLmu

    parameters = [["outofplane_twist", 2.5, 0.5, 20.0, 20.0, 0.500],
                  ["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.500],
                  ["outofplane_twist", 3.5, 0.5, 20.0, 20.0, 0.500]]

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
    parameters = [["outofplane_twist", 3.0, 0.5, 20.0, 20.0, 0.500],
                  ["outofplane_twist", 3.0, 0.7, 20.0, 20.0, 0.500],
                  ["outofplane_twist", 3.0, 0.9, 20.0, 20.0, 0.500]]

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
    return a*x


def fit_Rg2(q, Sq):
    popt, pcov = curve_fit(ax_fit, q*q/3, (1 - Sq))
    perr = np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def plot_Sq_fitted_Rg2(tex_lw=455.24408, ppi=72):

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 1))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    axs = fig.subplots(2, 1)

    folder = "../data/scratch_local/20240827"
    parameters = [["inplane_twist", 3.0, 0.8, 20.0, 20.0, 0.50],
                  ["outofplane_twist", 3.0, 0.8, 20.0, 20.0, 0.50]]
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
        axs[0].plot(q[:], all_Sq[i][:], "o", ms=3, mfc="None", label=r"$S(QB)$" + f"MC Rg2={all_features[:, all_feature_names.index("Rg2")][i]}, wRg2={all_features[:, all_feature_names.index("wRg2")][i]}")
        for qfn in [20, 40, 80]:
            Rg2, Rg2_err = fit_Rg2(q[:qfn], all_Sq[i][:qfn])
            axs[0].plot(q[:qfn], 1 - 1/3*Rg2*q[:qfn]**2, "-", lw=1, label=f"qf={q[qfn-1]}, Rg2={Rg2:.2f}"+r"$\pm$"+f"{Rg2_err:.2f}")

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
    parameters = [["outofplane_twist", 98, 1.5, 1.5, 0.400],
                  ["outofplane_twist", 98, 1.5, 1.5, 0.500],
                  ["outofplane_twist", 98, 1.5, 1.5, 0.600]]
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
    axin.tick_params(axis='both', labelsize=5)
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
