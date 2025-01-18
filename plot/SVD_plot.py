from Sq_plot import *
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Qt5Agg')
# import pandas as pd


def read_SVD_data(folder, segment_type):

    L, Kt, Kb, Rf, Rg, sqv0, sqv1, sqv2 = np.loadtxt(
        f"{folder}/{segment_type}_svd_projection.csv",
        skiprows=1,
        delimiter=",",
        unpack=True,
    )


def plot_SVD_data(tex_lw=240.71031, ppi=72):
    folder = "../data/20241011"
    segment_type = "outofplane_twist"

    data = np.loadtxt(
        f"{folder}/data_{segment_type}_svd.txt",
        skiprows=1,
        delimiter=",",
        unpack=True,
    )
    q, S, V0, V1, V2 = data

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.5))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.semilogx(range(1, len(S) + 1), S, "x", color="royalblue", mfc="None")
    ax1.semilogx(range(1, 4), S[:3], "o", color="red", mfc="None")
    ax1.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax1.set_xlabel(r"SVR", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$\Sigma$", fontsize=9, labelpad=0)

    ax2.semilogx(q, V0, label=r"$V0$")
    ax2.semilogx(q, V1, label=r"$V1$")
    ax2.semilogx(q, V2, label=r"$V2$")
    ax2.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax2.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"$V$", fontsize=9, labelpad=-5)
    # ax2.legend(loc="center right", bbox_to_anchor=(1.0,0.65),ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)
    ax2.legend(loc="upper left", ncol=2, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    ax1.text(0.7, 0.15, r"$(a)$", fontsize=9, transform=ax1.transAxes, color="black")
    ax2.text(0.7, 0.15, r"$(b)$", fontsize=9, transform=ax2.transAxes, color="black")

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/SVD.pdf", format="pdf")
    plt.savefig("figures/SVD.png", dpi=300)
    plt.show()
    plt.close()


def plot_SVD_data_with_Sq(tex_lw=240.71031, ppi=72):
    folder = "../data/20241011"
    segment_type = "outofplane_twist"

    data = np.loadtxt(
        f"{folder}/data_{segment_type}_svd.txt",
        skiprows=1,
        delimiter=",",
        unpack=True,
    )
    q, S, V0, V1, V2 = data

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 1.0))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(122)

    ax1.semilogx(range(1, len(S) + 1), S, "x", color="royalblue", mfc="None")
    ax1.semilogx(range(1, 4), S[:3], "o", color="red", mfc="None")
    ax1.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax1.set_xlabel(r"SVR", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$\Sigma$", fontsize=9, labelpad=0)

    # ax2 = ax1.inset_axes([0.2, 0.2, 0.9, 0.9])

    ax2.semilogx(q, V0, lw=1, label=r"$V0$")
    ax2.semilogx(q, V1, lw=1, label=r"$V1$")
    ax2.semilogx(q, V2, lw=1, label=r"$V2$")
    ax2.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax2.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"$V$", fontsize=9, labelpad=-5)
    # ax2.legend(loc="center right", bbox_to_anchor=(1.0,0.65),ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)
    ax2.legend(loc="upper left", ncol=2, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)

    ax1.text(0.7, 0.15, r"$(a)$", fontsize=9, transform=ax1.transAxes, color="black")
    ax2.text(0.7, 0.15, r"$(b)$", fontsize=9, transform=ax2.transAxes, color="black")

    parameters = [["outofplane_twist", 3.00, 0, 100, 100, 0.5, 0.93]]
    folder = "../data/scratch_local/20241014"
    segment_type, all_features, all_feature_names, all_Sq, all_Sq_err, q = read_Sq_data(folder, parameters)
    segment_type, lnLmu, lnLsig, Kt, Kb, Rf, alpha = parameters[0]
    L = all_features[0][8]
    Sq = all_Sq[0]
    lnSq0 = np.inner(np.log(Sq), V0) * V0
    lnSq1 = np.inner(np.log(Sq), V1) * V1
    lnSq2 = np.inner(np.log(Sq), V2) * V2
    ax3.loglog(q, Sq, linestyle=(0, (2, 2)), color="red", lw=1.5, label=r"$S(QB)$")
    ax3.loglog(q, np.exp(lnSq0 + lnSq1 + lnSq2), linestyle=(2, (2, 2)), lw=1.5, color="blue", label=r"$S0\odot S1\odot S2$")
    ax3.loglog(q, np.exp(lnSq0), "-", lw=0.75, label=r"$S0$")
    ax3.loglog(q, np.exp(lnSq1), "-", lw=0.75, label=r"$S1$")
    ax3.loglog(q, np.exp(lnSq2), "-", lw=0.75, label=r"$S2$")

    ax3.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7)
    ax3.set_xlabel(r"$QB$", fontsize=9, labelpad=0)
    ax3.set_ylabel(r"$S(QB)$", fontsize=9, labelpad=-6)
    ax3.legend(loc="lower left", ncol=2, columnspacing=0.5, handlelength=1.5, handletextpad=0.5, frameon=False, fontsize=9)

    ax1.text(0.7, 0.15, r"$(a)$", fontsize=9, transform=ax1.transAxes, color="black")
    ax2.text(0.7, 0.15, r"$(b)$", fontsize=9, transform=ax2.transAxes, color="black")
    ax3.text(0.9, 0.2, r"$(c)$", fontsize=9, transform=ax3.transAxes, color="black")

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/SVD_Sq.pdf", format="pdf")
    plt.savefig("figures/SVD_Sq.png", dpi=300)
    plt.show()
    plt.close()


def plot_SVD_feature_data(tex_lw=240.71031, ppi=72):
    folder = "../data/20241011"

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 1.15))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # outofplane_twist
    # Rf, alpha, L, Rg_2 inversible
    # Kb, Kt, not inversible
    ax1 = fig.add_subplot(321, projection="3d")
    ax2 = fig.add_subplot(322, projection="3d")
    ax3 = fig.add_subplot(323, projection="3d")
    ax4 = fig.add_subplot(324, projection="3d")
    ax5 = fig.add_subplot(325, projection="3d")
    ax6 = fig.add_subplot(326, projection="3d")

    # $cbar_major_locator = [100, 0.1, 5, 5, 20, 20]
    for segment_type in ["outofplane_twist"]:
        if segment_type == "inplane_twist":
            axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        elif segment_type == "outofplane_twist":
            axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        # plot the svd distribution

        data = np.loadtxt(
            f"{folder}/data_{segment_type}_svd_projection.txt",
            skiprows=1,
            delimiter=",",
            unpack=True,
        )
        lnLmu, lnLsig, Kt, Kb, Rf, alpha, Rg2, wRg2, L, L2, PDI, sqv0, sqv1, sqv2 = data
        feature_data = {
            "lnLmu": lnLmu,
            "lnLsig": lnLsig,
            "Kt": Kt,
            "Kb": Kb,
            "Rf": Rf,
            "alpha": alpha,
            "Rg2": Rg2,
            "wRg2": wRg2,
            "L": L,
            "L2": L2,
            "PDI": PDI,
        }

        feature_to_tex = {
            # "lnLmu": r"$\mu_{(\ln{L})}$",
            # "lnLsig": r"$\sigma_{(\ln{L})}$",
            "Kt": r"$K_t$",
            "Kb": r"$K_b$",
            "Rf": r"$R_a$",
            "alpha": r"$\alpha$",
            "L": r"$L$",
            # "PDI": r"$PDI$",
            "Rg2": r"$R_g^2$",
            # "wRg2": r"$\widetilde{R_g^2}$"
        }
        features = [
            "Rf",
            "alpha",
            "L",
            "Rg2",
            "Kt",
            "Kb",
        ]
        cbar_major_locator = [0.3, 0.1, 20, 40, 20, 20]
        for i in range(len(features)):
            mu = features[i]
            ax = axs[i]
            scatter = ax.scatter(sqv0, sqv1, sqv2, s=0.5, c=feature_data[mu], cmap="rainbow", rasterized=True)
            ax.view_init(elev=30, azim=70)

            ax.set_xlabel(r"$FV0$", fontsize=9, labelpad=-12, rotation=0)
            ax.set_ylabel(r"$FV1$", fontsize=9, labelpad=-7, rotation=0)
            ax.set_zlabel(r"$FV2$", fontsize=9, labelpad=-7, rotation=0)
            # ax.tick_params(labelsize=7, pad=0)
            ax.tick_params("x", direction="in", which="both", labelsize=7, pad=-7)
            ax.tick_params("y", direction="in", which="both", labelsize=7, pad=-2)
            ax.tick_params("z", direction="in", which="both", labelsize=7, pad=-2)

            # ax.zaxis.set_major_locator(plt.MultipleLocator(1))
            # ax.zaxis.set_minor_locator(plt.MultipleLocator(0.5))

            # ax.set_title(features_tex[i])
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=-0.0)  # , location="top", orientation='horizontal')
            cbar.ax.tick_params(direction="in", which="both", labelsize=7, pad=0.8)
            cbar.ax.yaxis.set_major_locator(plt.MultipleLocator(cbar_major_locator[i]))
            cbar.ax.yaxis.set_minor_locator(plt.MultipleLocator(cbar_major_locator[i] * 0.5))
            # cbar.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
            # cbar.ax.set_axes_locator(plt.MultipleLocator(cbar_major_locator[i]))
            # cbar.ax.xaxis.set_minor_locator(plt.MultipleLocator(cbar_major_locator[i]*0.5))
            cbar.ax.set_title(feature_to_tex[mu], fontsize=9)

            # cbar = .colorbar(axs.collections[0])
            # cbar.set_label(mu, fontsize=9)

            # ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
            # ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
            # ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            # ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
            # ax.zaxis.set_major_locator(plt.MultipleLocator(0.4))
            # ax.zaxis.set_minor_locator(plt.MultipleLocator(0.2))

            ax.grid(True, which="minor")

    textlabel = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i in range(len(axs)):
        axs[i].text2D(0.2, 0.85, textlabel[i], fontsize=9, transform=axs[i].transAxes, color="black")

    # plt.tight_layout(pad=1.5)  # , h_pad=0) #, h_pad=-3, w_pad=2)
    plt.tight_layout(pad=0.4, w_pad=2.4, h_pad=0.5)
    plt.savefig("figures/SVD_feature.pdf", format="pdf", dpi=300)
    plt.savefig("figures/SVD_feature.png", dpi=300)
    plt.show()

    # save animation
    for ii in range(40):
        if ii < 15:
            for ax in axs:
                ax.view_init(elev=30.0, azim=18 * ii)
        else:
            for ax in axs:
                ax.view_init(elev=30.0, azim=18 * ii)
        plt.tight_layout()
        plt.savefig(f"figures/SVD_gif/demo_ani%d" % ii + ".png", dpi=300)

    plt.close()


def plot_animate_polymer_config(filename, tag):
    plt.figure()
    ax = plt.axes(projection="3d")
    ax_plot_polymer_config_rectangle(ax, filename)

    for ii in range(40):
        if ii < 15:
            for ax in axs:
                ax.view_init(elev=18.0, azim=18 * ii)
        else:
            ax.view_init(elev=18.0 * (ii - 14), azim=18 * 14)
        plt.tight_layout()
        plt.savefig(f"figures/config_gif/{tag}_demo_ani%d" % ii + ".png", dpi=300)
    plt.close()
