from Sq_plot import read_Sq_data
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def read_SVD_data(folder, segment_type):

    L, Kt, Kb, Rf, Rg, sqv0, sqv1, sqv2 = np.loadtxt(f"{folder}/{segment_type}_svd_projection.csv", skiprows=1, delimiter=",", unpack=True)


def plot_SVD_data():
    folder = "../data/20240611"
    segment_type = "outofplane_twist"
    Ls = np.arange(50, 98.1, 1)
    logKts = [2.00]
    logKbs = [2.00]
    Rfs = np.arange(0.40, 0.601, 0.1)
    parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]

    all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)

    svd = np.linalg.svd(all_Delta_Sq)

    ppi = 72
    fig = plt.figure(figsize=(246 / ppi * 1, 246 / ppi * 0.8))
    axs00 = fig.add_subplot(111)

    axs00.plot(range(len(svd.S)), svd.S, "x", markersize=5, markerfacecolor='none', label=r"$\Sigma$")
    axs00.plot(range(len(svd.S))[:3], svd.S[:3], "ro", markersize=5, markerfacecolor='none')
    axs00.set_xlabel("SVR", fontsize=9)
    axs00.set_ylabel(r"$\Sigma$", fontsize=9)
    axs00.tick_params(labelsize=6)
    axs00.xaxis.set_major_locator(plt.MultipleLocator(20))
    axs00.xaxis.set_minor_locator(plt.MultipleLocator(10))
    plt.tight_layout(pad=1.5)
    #plt.show()
    plt.savefig("figures/SVD.png", dpi=300)

def plot_SVD_feature_data():
    folder = "../data/20240611"
    segment_type = "outofplane_twist"
    Ls = np.arange(50, 98.1, 1)
    logKts = [2.00]
    logKbs = [2.00]
    Rfs = np.arange(0.40, 0.601, 0.1)
    parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]

    all_features, all_Sq, all_Sq_rod, all_Delta_Sq, q = read_Sq_data(folder, parameters)

    svd = np.linalg.svd(all_Delta_Sq)

    ppi = 72
    fig = plt.figure(figsize=(246 / ppi * 1.7, 246 / ppi * 0.5))

    #axs00 = fig.add_subplot(221)
    axs01 = fig.add_subplot(131, projection='3d')
    axs10 = fig.add_subplot(132, projection='3d')
    axs11 = fig.add_subplot(133, projection='3d')
    '''
    axs00.plot(range(len(svd.S)), svd.S, "x", markersize=5, markerfacecolor='none', label=r"$\Sigma$")
    axs00.plot(range(len(svd.S))[:3], svd.S[:3], "ro", markersize=5, markerfacecolor='none')
    axs00.set_xlabel("SVR", fontsize=9)
    axs00.set_ylabel(r"$\Sigma$", fontsize=9)
    axs00.tick_params(labelsize=6)
    axs00.xaxis.set_major_locator(plt.MultipleLocator(20))
    axs00.xaxis.set_minor_locator(plt.MultipleLocator(10))

    '''

    L, Kt, Kb, Rf, Rg, sqv0, sqv1, sqv2 = np.loadtxt(f"{folder}/data_{segment_type}_svd_projection.txt", skiprows=1, delimiter=",", unpack=True)

    scatter01 = axs01.scatter(sqv0, sqv1, sqv2, s=1, c=L, cmap="jet_r")
    axs01.tick_params(labelsize=5)

    colorbar01 = fig.colorbar(scatter01, ax=axs01, fraction=0.04, pad=0.01)
    colorbar01.ax.set_title('L',fontsize=8)
    colorbar01.ax.tick_params(labelsize=5)

    scatter10 = axs10.scatter(sqv0, sqv1, sqv2, s=1, c=Rf, cmap="jet_r")
    colorbar10 = fig.colorbar(scatter10, ax=axs10, ticks=[0.4, 0.45, 0.5, 0.55,0.6], fraction=0.04, pad=0.01)
    colorbar10.ax.set_title('Rf',fontsize=8)
    colorbar10.ax.tick_params(labelsize=5)

    scatter11 = axs11.scatter(sqv0, sqv1, sqv2, s=1, c=Rg, cmap="jet_r")
    colorbar11 = fig.colorbar(scatter11, ax=axs11, fraction=0.04, pad=0.01)
    colorbar11.ax.set_title('Rg',fontsize=8)
    colorbar11.ax.tick_params(labelsize=5)

    for ax in [axs01, axs10, axs11]:
        ax.view_init(elev=16., azim=-110)
        ax.set_xlabel("V0", fontsize=8, labelpad=-12)
        ax.set_ylabel("V1", fontsize=8, labelpad=-7)
        ax.zaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.set_zlabel("V2", fontsize=8, labelpad=-5)
        ax.tick_params("x",labelsize=6,pad=-7)
        ax.tick_params("y",labelsize=6,pad=-5)
        ax.tick_params("z",labelsize=6,pad=-0.5)
        ax.set_box_aspect([1, 1, 0.8])

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        ax.zaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.zaxis.set_minor_locator(plt.MultipleLocator(0.025))
        ax.grid(True, which='minor')

    plt.tight_layout(pad=1.5)
    #plt.show()
    plt.savefig("figures/SVD_feature.png", dpi=300)

    plt.close()
