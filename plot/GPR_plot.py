from Sq_plot import read_Sq_data
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def plot_GPR_data():

    folder = "../data/20240611"
    segment_type = "outofplane_twist"

    ppi = 72
    fig = plt.figure(figsize=(246 / ppi * 1.8, 246 / ppi * 0.6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    L, L_predict, L_predict_err = np.loadtxt(f"{folder}/data_{segment_type}_L_prediction.txt", skiprows=1, delimiter=",", unpack=True)
    ax1.errorbar(L, L_predict, L_predict_err, marker="x", markersize=1,linestyle="None")
    ax1.plot(L, L, "r-", linewidth=0.5)
    ax1.set_xlabel("L", fontsize=9)
    ax1.set_ylabel("ML inversed L", fontsize=9)
    ax1.set_xlim([np.min(L), np.max(L)])
    ax1.set_ylim([np.min(L), np.max(L)])
    ax1.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(10))

    Rf, Rf_predict, Rf_predict_err = np.loadtxt(f"{folder}/data_{segment_type}_Rf_prediction.txt", skiprows=1, delimiter=",", unpack=True)
    ax2.errorbar(Rf, Rf_predict, yerr=Rf_predict_err, marker="x", markersize=1,linestyle="None")
    ax2.plot(Rf, Rf, "r-", linewidth=0.5)
    ax2.set_xlabel("Rf", fontsize=9)
    ax2.set_ylabel("ML inversed Rf", fontsize=9)
    ax2.set_xlim([np.min(Rf), np.max(Rf)])
    ax2.set_ylim([np.min(Rf), np.max(Rf)])
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    Rg, Rg_predict, Rg_predict_err = np.loadtxt(f"{folder}/data_{segment_type}_Rg_prediction.txt", skiprows=1, delimiter=",", unpack=True)
    ax3.errorbar(Rg, Rg_predict, yerr=Rg_predict_err, marker="x", markersize=1,linestyle="None")
    ax3.plot(Rg, Rg, "r-", linewidth=0.5)
    ax3.set_xlabel("Rg", fontsize=9)
    ax3.set_ylabel("ML inversed Rg", fontsize=9)
    ax3.set_xlim([np.min(Rg), np.max(Rg)])
    ax3.set_ylim([np.min(Rg), np.max(Rg)])
    ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.tight_layout()

    plt.savefig("figures/GPR.png", dpi=300)

    plt.close()