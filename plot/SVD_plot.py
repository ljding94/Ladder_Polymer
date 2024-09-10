from Sq_plot import read_Delta_Sq_data
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Qt5Agg')
# import pandas as pd


def read_SVD_data(folder, segment_type):

    L, Kt, Kb, Rf, Rg, sqv0, sqv1, sqv2 = np.loadtxt(f"{folder}/{segment_type}_svd_projection.csv", skiprows=1, delimiter=",", unpack=True)


def plot_SVD_data(tex_lw=455.24408, ppi=72):
    folder = "../data/20240813"
    segment_type = "inplane_twist"

    # TODO: this need to be updated to directly read from stored SVD data
    Ls = np.arange(50, 98.1, 1)
    logKts = [2.00]
    logKbs = [2.00]
    Rfs = np.arange(0.40, 0.601, 0.1)

    parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]

    segment_type, all_features, all_Sq, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)

    svd = np.linalg.svd(all_Delta_Sq)

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.8))
    axs00 = fig.add_subplot(111)

    axs00.plot(range(len(svd.S)), svd.S, "x", markersize=5, markerfacecolor='none', label=r"$\Sigma$")
    axs00.plot(range(len(svd.S))[:3], svd.S[:3], "ro", markersize=5, markerfacecolor='none')
    axs00.set_xlabel("SVR", fontsize=9)
    axs00.set_ylabel(r"$\Sigma$", fontsize=9)
    axs00.tick_params(labelsize=6)
    axs00.xaxis.set_major_locator(plt.MultipleLocator(20))
    axs00.xaxis.set_minor_locator(plt.MultipleLocator(10))
    plt.tight_layout(pad=1.5)
    # plt.show()
    plt.savefig("figures/SVD.png", dpi=300)


def plot_SVD_feature_data(tex_lw=455.24408, ppi=72):
    folder = "../data/20240819"

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.39))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    # for inplane_twist

    # for Lmu, Lr, Rg which are trainable
    ax11 = fig.add_subplot(241, projection='3d')
    ax12 = fig.add_subplot(242, projection='3d')
    ax13 = fig.add_subplot(243, projection='3d')
    ax14 = fig.add_subplot(244, projection='3d')

    ax21 = fig.add_subplot(245, projection='3d')
    ax22 = fig.add_subplot(246, projection='3d')
    ax23 = fig.add_subplot(247, projection='3d')
    ax24 = fig.add_subplot(248, projection='3d')

    # $cbar_major_locator = [100, 0.1, 5, 5, 20, 20]
    for segment_type in ["inplane_twist"]:
        if (segment_type == "inplane_twist"):
            axs = [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24]
        # elif (segment_type == "outofplane_twist"):
        #    axs = [ax21, ax22, ax23]

        # plot the svd distribution
        data = np.loadtxt(f"{folder}/data_{segment_type}_svd_projection.txt", skiprows=1, delimiter=",", unpack=True)
        lnLmu, lnLsig, Kt, Kb, Rf, Rg2, L, L2, PDI, sqv0, sqv1, sqv2 = data
        feature_data = {"lnLmu": lnLmu, "lnLsig": lnLsig,
                        "Kt": Kt, "Kb": Kb, "Rf": Rf,
                        "Rg2": Rg2, "L": L, "L2": L2, "PDI": PDI}

        features = ["lnLmu", "lnLsig", "Kt", "Kb", "L", "PDI", "Rf", "Rg2"]
        features_tex = [r"$\mu_{(\ln{L})}$", r"$\sigma_{(\ln{L})}$", r"$K_t$", r"$K_b$", r"$\left< L \right>$", r"$PDI$", r"$R_f$", r"$R_g^2$"]
        cbar_major_locator = [0.4, 0.03, 5, 5, 40, 0.2, 0.1, 40]
        for i in range(len(features)):
            mu = features[i]
            ax = axs[i]
            scatter = ax.scatter(sqv0, sqv1, sqv2, s=0.5, c=feature_data[mu], cmap="jet_r")
            ax.view_init(elev=23, azim=-102)

            ax.set_xlabel("V0", fontsize=9, labelpad=-12, rotation=0)
            ax.set_ylabel("V1", fontsize=9, labelpad=-5, rotation=0)
            ax.set_zlabel("V2", fontsize=9, labelpad=-5, rotation=0)
            # ax.tick_params(labelsize=7, pad=0)
            ax.tick_params("x", labelsize=7, pad=-7)
            ax.tick_params("y", labelsize=7, pad=-2)
            ax.tick_params("z", labelsize=7, pad=-0.5)

            # ax.set_title(features_tex[i])
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=-0.04)  # , location="top", orientation='horizontal')
            cbar.ax.tick_params(labelsize=7, pad=0)
            cbar.ax.yaxis.set_major_locator(plt.MultipleLocator(cbar_major_locator[i]))
            cbar.ax.yaxis.set_minor_locator(plt.MultipleLocator(cbar_major_locator[i]*0.5))
            #cbar.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            #cbar.ax.set_axes_locator(plt.MultipleLocator(cbar_major_locator[i]))
            #cbar.ax.xaxis.set_minor_locator(plt.MultipleLocator(cbar_major_locator[i]*0.5))
            cbar.ax.set_title(features_tex[i], fontsize=9)

            # cbar = .colorbar(axs.collections[0])
            # cbar.set_label(mu, fontsize=9)

            # ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
            # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
            # ax.zaxis.set_major_locator(plt.MultipleLocator(0.4))
            # ax.zaxis.set_minor_locator(plt.MultipleLocator(0.2))

            ax.grid(True, which='minor')

    # plt.tight_layout(pad=1.5)  # , h_pad=0) #, h_pad=-3, w_pad=2)
    plt.tight_layout(pad=0.01)
    plt.savefig("figures/SVD_feature.pdf", format="pdf", dpi=300)
    plt.show()
    plt.close()
