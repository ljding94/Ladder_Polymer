from Sq_plot import read_Sq_data
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def read_SVD_data(folder, segment_type):

    L, Kt, Kb, Rf, Rg, sqv0, sqv1, sqv2 = np.loadtxt(f"{folder}/{segment_type}_svd_projection.csv", skiprows=1, delimiter=",", unpack=True)


def plot_SVD_data(tex_lw=455.24408, ppi=72):
    folder = "../data/20240613"
    segment_type = "outofplane_twist"

    # TODO: this need to be updated to directly read from stored SVD data
    Ls = np.arange(50, 98.1, 1)
    logKts = [2.00]
    logKbs = [2.00]
    Rfs = np.arange(0.40, 0.601, 0.1)

    parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]

    all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)

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
    folder = "../data/20240630"

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.6))
    # for inplane_twist

    # for Lmu, Lr, Rg which are trainable
    ax11 = fig.add_subplot(231, projection='3d')
    ax12 = fig.add_subplot(232, projection='3d')
    ax13 = fig.add_subplot(233, projection='3d')

    ax21 = fig.add_subplot(234, projection='3d')
    ax22 = fig.add_subplot(235, projection='3d')
    ax23 = fig.add_subplot(236, projection='3d')



    # for outofplane twist
    # ax21 = fig.add_subplot(234, projection='3d')
    # ax22 = fig.add_subplot(235, projection='3d')
    # ax23 = fig.add_subplot(236, projection='3d')

    features = ["Lmu", "Rf", "Rg", "Lsig", "Kb", "Kt"]
    features_tex = [r"$L_{\mu}$", r"$R_f$", r"$R_g$", r"$L_{\sigma}$", r"$K_b$", r"$K_t$"]
    #$cbar_major_locator = [100, 0.1, 5, 5, 20, 20]
    for segment_type in ["outofplane_twist"]:
        if (segment_type == "outofplane_twist"):
            axs = [ax11, ax12, ax13, ax21, ax22, ax23]
        # elif (segment_type == "outofplane_twist"):
        #    axs = [ax21, ax22, ax23]

        # plot the svd distribution
        data = np.loadtxt(f"{folder}/data_{segment_type}_svd_projection.txt", skiprows=1, delimiter=",", unpack=True)
        Lmu, Lsig, Kt, Kb, Rf, Rg, sqv0, sqv1, sqv2 = data
        # feature_data = {"L": L, "Rf": Rf, "Rg": Rg}
        feature_data = {"Lmu": Lmu, "Rf": Rf, "Rg": Rg, "Lsig": Lsig, "Kb": Kb, "Kt": Kt}
        for i in range(len(features)):
            mu = features[i]
            ax = axs[i]
            scatter = ax.scatter(sqv0, sqv1, sqv2, s=1, c=feature_data[mu], cmap="jet_r")
            ax.view_init(elev=25., azim=70)

            ax.set_xlabel("V0", fontsize=9, labelpad=-12, rotation=0)
            ax.set_ylabel("V1", fontsize=9, labelpad=-7, rotation=0)
            ax.set_zlabel("V2", fontsize=9, labelpad=-5, rotation=0)
            # ax.tick_params(labelsize=7, pad=0)
            ax.tick_params("x", labelsize=7, pad=-7)
            ax.tick_params("y", labelsize=7, pad=-5)
            ax.tick_params("z", labelsize=7, pad=-0.5)

            # ax.set_title(features_tex[i])
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=-0.04)#, location="top", orientation='horizontal')
            cbar.ax.tick_params(labelsize=7, pad=0)
            #cbar.ax.set_axes_locator(plt.MultipleLocator(cbar_major_locator[i]))
            #cbar.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            cbar.ax.set_title(features_tex[i], fontsize=9)

            # cbar = .colorbar(axs.collections[0])
            # cbar.set_label(mu, fontsize=9)
            '''
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
            ax.zaxis.set_major_locator(plt.MultipleLocator(0.4))
            ax.zaxis.set_minor_locator(plt.MultipleLocator(0.2))
            '''
            ax.grid(True, which='minor')

            #ax.view_init(elev=16., azim=-110)

    plt.tight_layout(pad=1.5) #, h_pad=0) #, h_pad=-3, w_pad=2)
    # plt.tight_layout(pad=0.05)
    #plt.show()
    plt.savefig("figures/SVD_feature.pdf", format="pdf", dpi=300)
    plt.close()
