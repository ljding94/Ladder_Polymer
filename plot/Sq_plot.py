import matplotlib.pyplot as plt
import numpy as np
import os

# old version
def read_Sq_data(folder, parameters):
    all_features = []
    all_Sq = []
    all_Sq_rod = []
    all_Delta_Sq = []
    for segment_type, L, logKt, logKb, Rf in parameters:
        filename = f"{folder}/obs_{segment_type}_L{L:.0f}_logKt{logKt:.2f}_logKb{logKb:.2f}_Rf{Rf:.3f}_SqB.csv"
        print("reading: ", filename)
        Sqdata = np.genfromtxt(filename, delimiter=',', skip_header=1)
        features = Sqdata[0, 2: 7]
        Sq, q = Sqdata[0, 7:], Sqdata[2, 7:]
        Sq_rod_discrete = Sqdata[3, 7:]
        Delta_Sq = Sq/Sq_rod_discrete
        all_features.append(features)
        all_Sq.append(Sq)
        all_Sq_rod.append(Sq_rod_discrete)
        all_Delta_Sq.append(Delta_Sq)
    # all_Sq = np.transpose(all_Sq)
    return np.array(all_features), all_Sq, all_Sq_rod, all_Delta_Sq, q


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


# consisten with ML analysis version
def read_Delta_Sq_data(folder, parameters, L0=300):
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
        for segment_type, L, logKt, logKb, Rf in parameters:
            all_filename.append(f"{folder}/obs_{segment_type}_L{L:.0f}_logKt{logKt:.2f}_logKb{logKb:.2f}_Rf{Rf:.3f}_SqB.csv")

    for filename in all_filename:
        print("reading: ", filename)
        if os.path.exists(filename):
            Sqdata = np.genfromtxt(filename, delimiter=',', skip_header=1)
            features = Sqdata[0, 2: 8]
            Sq, Sq_err, q = Sqdata[0, 8:], Sqdata[1, 8:], Sqdata[2, 8:]
            # Sq_rod_discrete = Sqdata[3, 7:]
            if len(Sq_rod_discrete) == 0:
                Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(q, L0)
            # normalize Sq by Sq_rod_discrete with L0
            Delta_Sq = Sq/Sq_rod_discrete
            #Delta_Sq = Sq
            Delta_Sq_err = Sq_err/Sq_rod_discrete
            all_features.append(features)
            all_Sq.append(Sq)
            all_Delta_Sq.append(Delta_Sq)
            all_Delta_Sq_err.append(Delta_Sq_err)
        else:
            print(f"Warning: File {filename} not found. Skiped")

    return segment_type, np.array(all_features), all_Sq, all_Delta_Sq, all_Delta_Sq_err, q


def plot_polymer_Sq(tex_lw=455.24408, ppi=72):

    # map Rf and L to line color and style
    # LS = {50: "-", 70: "--", 90: "-."}
    # LC = {0.4: "tomato", 0.5: "lawngreen", 0.6: "steelblue"}  # black for hard rod
    colors = ["tomato", "lawngreen", "steelblue"]
    segment_type_map = {"outofplane_twist": "2D CANAL", "inplane_twist": "3D CANAL"}

    folder = "../data/20240613"
    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.6))
    axs = fig.subplots(2, 2, sharex=True)

    # plot variation of Rf
    parameters = [["outofplane_twist", 90, 1.5, 1.5, 0.400],
                  ["outofplane_twist", 90, 1.5, 1.5, 0.500],
                  ["outofplane_twist", 90, 1.5, 1.5, 0.600]]

    #all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)
    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(q, 300)

    n = 0  # skip first few data
    axs[0, 0].loglog(q[n:], Sq_rod_discrete[n:], "--", color="black", label="rod, L=300")
    for i in range(len(parameters)):
        segment_type, L, logKt, logKb, Rf = parameters[i]
        axs[0, 0].loglog(q[n:], all_Sq[i][n:], "-", color=colors[i], label=f"Rf={Rf}")
        axs[0, 1].semilogx(q[n:], all_Delta_Sq[i][n:], "-", color=colors[i], label=f"Rf={Rf}")

    axs[0, 0].tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7)
    axs[0, 1].tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7)

    # axs[0, 0].set_xlabel("QB", fontsize=10)
    axs[0, 0].set_ylabel("S(QB)", fontsize=10)
    axs[0, 0].legend(title=f"{segment_type_map[segment_type]}, L={L}", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=10)
    # axs[0, 1].set_xlabel("QB", fontsize=10)
    axs[0, 1].set_ylabel(r"$\Delta S(QB)$", fontsize=10)
    axs[0, 1].legend(title=f"{segment_type_map[segment_type]}, L={L}", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=10)


    # plot variation of L
    parameters = [["outofplane_twist", 50, 1.5, 1.5, 0.500],
                  ["outofplane_twist", 70, 1.5, 1.5, 0.500],
                  ["outofplane_twist", 90, 1.5, 1.5, 0.500]]

    #all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)
    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)

    for i in range(len(parameters)):
        segment_type, L, logKt, logKb, Rf = parameters[i]
        axs[1, 0].semilogx(q[n:], all_Delta_Sq[i][n:], "-", color=colors[i], label=f"L={L}")
    axs[1, 0].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    axs[1, 0].set_xlabel("QB", fontsize=10)
    axs[1, 0].set_ylabel(r"$S\Delta (QB)$", fontsize=10)
    axs[1, 0].legend(title=f"{segment_type_map[segment_type]}, Rf={Rf}", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=10)

    # plot variaotion of segment_type
    parameters = [["outofplane_twist", 70, 1.5, 1.5, 0.500],
                  ["inplane_twist", 70, 1.5, 1.5, 0.500]]
    #all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)
    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)

    for i in range(len(parameters)):
        segment_type, L, logKt, logKb, Rf = parameters[i]
        axs[1, 1].semilogx(q[n:], all_Delta_Sq[i][n:], "-", color=colors[i], label=f"{segment_type_map[segment_type]}")
    axs[1, 1].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    axs[1, 1].set_xlabel("QB", fontsize=10)
    axs[1, 1].set_ylabel(r"$S\Delta (QB)$", fontsize=10)
    axs[1, 1].legend(title=f"L={L}, Rf={Rf}", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=10)

    # axs[1, 1].legend(ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.1, frameon=False, fontsize=10)

    plt.tight_layout()
    #plt.show()
    plt.savefig("figures/Sq.pdf", format="pdf")
    plt.close()


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
