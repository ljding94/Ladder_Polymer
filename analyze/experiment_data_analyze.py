import matplotlib.pyplot as plt
import numpy as np
import scipy
from ML_analysis import *
from scipy.signal import savgol_filter


def Guinier_fit(q, A, Rg2):
    return A * np.exp(-(q**2) * Rg2 / 3)

    # (1 - q**2*Rg2/3)


def fit_Guinier(Q, Sq, Sqerr):
    popt, pcov = scipy.optimize.curve_fit(Guinier_fit, Q, Sq, sigma=Sqerr, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def Gaussian_chain_fit(q, A, Rg2):
    ans = 2 * (q * q * Rg2 - 1 + np.exp(-q * q * Rg2)) / (q * q * Rg2) ** 2
    return A * ans


def fit_Gaussian_chain(Q, Sq, Sqerr):
    popt, pcov = scipy.optimize.curve_fit(Gaussian_chain_fit, Q, Sq, sigma=Sqerr, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def flexible_cyliner_Sq(q,)

def fit_flexible_cylinder(Q, Sq, Sqerr):
    pass

def fit_largeQ_L(Q, Sq, Sqerr):
    popt, pcov = scipy.optimize.curve_fit(L_fit, Q, Sq, sigma=Sqerr, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def plot_experimental_Sq():
    lb = 8.12  # segment length in Angstrom

    folder = "../data/incoh_banjo_expdata"
    # solvent
    filename = "../data/incoh_banjo_expdata/merged_incoh_L1_Iq.txt"
    data = np.genfromtxt(filename, delimiter="\t", skip_header=2)
    Q_sol, I_sol, dI_sol = data[:, 0], data[:, 1], data[:, 2]

    # low-mw 2d canal in DCB
    filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq.txt"
    data = np.genfromtxt(filename, delimiter="\t", skip_header=2)
    Q_L2, I_L2, dI_L2 = data[:-1, 0], data[:-1, 1], data[:-1, 2]

    plt.figure()
    plt.loglog(Q_L2, I_L2, label="Sample", marker="None", linestyle="-", linewidth=2)
    plt.loglog(
        Q_sol,
        I_sol,
        label="Solvent",
        marker="None",
        linestyle="--",
        linewidth=2,
    )
    plt.xlabel("q (1/Å)", fontsize=14)
    plt.ylabel("Intensity (cm⁻¹)", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{folder}/merged_incoh_L2_Iq.png")
    # plt.show()

    f = 0.96
    I_sub = I_L2 - I_sol * f
    dI_sub = np.sqrt(dI_L2**2 + (f * dI_sol) ** 2)
    plt.figure()
    plt.loglog(
        Q_L2 * lb,
        I_sub,
        label="Subtracted",
        marker="None",
        color="blue",
        linestyle="-",
        linewidth=2,
    )
    plt.errorbar(Q_L2 * lb, I_sub, yerr=dI_sub, color="blue")
    QB = Q_L2 * lb

    QBi, QBf = 0.07, 3
    plt.plot([QBi, QBi], [1e-4, 1e-1], "k--")
    plt.plot([QBf, QBf], [1e-4, 1e-1], "k--")
    print("QB[:10]", QB[:10])
    print("QB[:10]", QB[-10:])
    plt.xlabel("QB", fontsize=14)
    plt.ylabel("Intensity (cm⁻¹)", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{folder}/merged_incoh_L2_Iq_subtracted.png")
    output_filename = f"{folder}/merged_incoh_L2_Iq_subtracted.txt"
    np.savetxt(
        output_filename,
        np.column_stack((QB, I_sub, dI_sub)),
        header="QB,I_sub,dI_sub",
        delimiter=",",
        comments="",
    )
    # plt.show()

    bin_bum = 100
    print("np.arange(bin_bum)/(bin_bum-1)", np.arange(bin_bum) / (bin_bum - 1))
    QBnew = QBi * np.power(QBf / QBi, np.arange(bin_bum) / (bin_bum - 1))
    I_interp = scipy.interpolate.griddata(QB, I_sub, QBnew, method="cubic")
    dI_interp = scipy.interpolate.griddata(QB, dI_sub, QBnew, method="cubic")

    plt.figure()
    plt.loglog(
        QBnew,
        I_interp,
        label="Subtracted, interpolated",
        marker="None",
        color="blue",
        linestyle="-",
        linewidth=2,
    )
    plt.errorbar(QBnew, I_interp, yerr=dI_interp, color="blue", capsize=2)
    plt.xlabel("QB", fontsize=14)
    plt.ylabel("Intensity (cm⁻¹)", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated.png")
    # plt.show()

    """
    for n in [50, 55, 60, 65, 70, 75, 80]:
        popt, perr = fit_Guinier(QBnew[:n], I_interp[:n], dI_interp[:n])
        print("fit Guinier with n=", n)
        print("popt", popt)
        print("perr", perr)
        plt.figure()
        plt.errorbar(QBnew, I_interp, yerr=dI_interp, color="blue", linestyle='None')
        plt.loglog(QBnew[:n], Guinier_fit(QBnew[:n], *popt), label='Subtracted, interpolated, Guinier fitted', marker='None', color="tomato", linestyle='-', linewidth=2)
        plt.title(f"Rg2={(popt[1]):.4f}±{perr[1]:.4f}, A = {(popt[0]):.4f}±{perr[0]:.4f}")
        plt.xlabel('QB', fontsize=14)
        plt.ylabel('Intensity (cm⁻¹)', fontsize=14)
        plt.legend()
        plt.grid(True, which='both', ls='--', linewidth=0.5)
        plt.tight_layout()
        output_filename = f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_fit_n{n}_normalized_Iq.txt"
        np.savetxt(output_filename, np.column_stack((QBnew, I_interp/popt[0], dI_interp/popt[0])),
                   header="QBnew,I_interp,dI_interp", delimiter=",", comments='')
        plt.savefig(f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_fit_n{n}.png")
        plt.show()
    """

    """
    for n in [50, 55, 60, 65, 70, 75, 80, 90, 95]:
        popt, perr = fit_Gaussian_chain(QBnew[:n], I_interp[:n], dI_interp[:n])
        print("popt", popt)
        print("perr", perr)
        plt.figure()
        plt.errorbar(QBnew, I_interp, yerr=dI_interp, color="blue", linestyle='None')
        plt.loglog(QBnew[:n], Gaussian_chain_fit(QBnew[:n], *popt), label='Subtracted, interpolated, Gaussian_chain fitted', marker='None', color="tomato", linestyle='-', linewidth=2)
        plt.title(f"Rg2={(popt[1]):.4f}±{perr[1]:.4f}, A = {(popt[0]):.4f}±{perr[0]:.4f}")
        plt.xlabel('QB', fontsize=14)
        plt.ylabel('Intensity (cm⁻¹)', fontsize=14)
        plt.legend()
        plt.grid(True, which='both', ls='--', linewidth=0.5)
        plt.tight_layout()
        output_filename = f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated_Gaussian_Chain_fit_n{n}_normalized_Iq.txt"
        np.savetxt(output_filename, np.column_stack((QBnew, I_interp/popt[0], dI_interp/popt[0])),
                   header="QBnew,I_interp,dI_interp", delimiter=",", comments='')
        plt.savefig(f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated_Gaussian_chain_fit_n{n}.png")
        plt.show()
    """

    # filename = "../data/scratch_local/20241010/obs_MC_outofplane_twist_lnLmu2.56495_lnLsig0.0_Kt50.0_Kb50.0_Rf0.20_alpha52.45_SqB.csv"
    filename = "/Users/ldq/Work/Ladder_Polymer/data/scratch_local/data_pool/obs_MC_outofplane_twist_lnLmu2.56495_lnLsig0.0_Kt50.0_Kb50.0_Rf0.15_alpha50.4_SqB.csv"
    # print("reading: ", filename)
    if os.path.exists(filename):
        Sqdata = np.genfromtxt(filename, delimiter=",", skip_header=1, max_rows=1)
        if len(Sqdata) == 0:
            print(f"Warning: File {filename} is empty. Skipped")
        features = Sqdata[2:12]
        features = np.insert(features, 10, features[9] / (features[8] * features[8]))  # add L^2/(L)^2 = PDI
        qdata = np.genfromtxt(filename, delimiter=",", skip_header=3, max_rows=1)
        Sq, q = Sqdata[12:], qdata[12:]

    from scipy.interpolate import splrep, BSpline

    # find normalization factor
    n = 65
    popt, perr = fit_Guinier(QBnew[:n], I_interp[:n], dI_interp[:n])
    I_Guinier = Guinier_fit(QBnew[:n], *popt)
    print("fit Guinier with n=", n)
    print("popt", popt)
    print("perr", perr)
    A = popt[0]
    I_interp = I_interp / A
    I_Guinier = I_Guinier / A
    I_patch = np.concatenate((I_Guinier[:n], I_interp[n:]))
    I_smooth = savgol_filter(I_patch, 15, 2)

    plt.figure()
    # plt.errorbar(QBnew, I_interp/0.04, yerr=dI_interp/0.04, color="blue", linestyle='None')
    plt.loglog(
        QBnew[:n],
        Guinier_fit(QBnew[:n], *popt) / A,
        label="Subtracted, interpolated, Guinier fitted",
        marker="None",
        color="royalblue",
        linestyle="-",
        linewidth=2,
    )
    plt.loglog(QBnew, I_interp, "x", color="blue", linestyle="None")
    plt.loglog(QBnew, I_patch, "-", color="gray", alpha=0.2)
    plt.loglog(QBnew, I_smooth, "-", color="red", label="patch smoothed", linewidth=2)
    plt.loglog(
        q,
        Sq,
        marker="None",
        color="black",
        linestyle="-",
        linewidth=2,
        label="MC GPR",
    )

    n=85
    popt, perr = fit_largeQ_L(QBnew[n:], I_interp[n:]/A, dI_interp[n:]/A)
    #plt.loglog(QBnew[n:], L_fit(QBnew[n:], *popt), ":", color="red", label=f"q^-1,L={popt[0]:.4f}")
    plt.loglog(QBnew[n:], L_fit(QBnew[n:], 11), ":", color="red", label=f"q^-1,L={popt[0]:.4f}")
    print("popt", popt)
    print("perr", perr)


    plt.xlabel("QB", fontsize=14)
    plt.ylabel("Intensity (cm⁻¹)", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    output_filename = f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_n{n}_patch_smooth_normalized_Iq.txt"
    np.savetxt(
        output_filename,
        np.column_stack((QBnew, I_smooth, dI_interp / A)),
        header="QBnew,I_smooth,dI_smooth_ish",
        delimiter=",",
        comments="",
    )
    plt.savefig(f"{folder}/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_n{n}_patch_smooth_normalized_Iq.png")
    plt.show()
    plt.close()
