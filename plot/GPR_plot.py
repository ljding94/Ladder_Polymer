from Sq_plot import read_Sq_data
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def plot_GPR_data(tex_lw=455.24408, ppi=72):

    folder = "../data/20240713_partial"
    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.33)) #0.65))
    ax11 = fig.add_subplot(131)
    ax12 = fig.add_subplot(132)
    ax13 = fig.add_subplot(133)

    # plot both segment type in same ax
    # ax21 = fig.add_subplot(234)
    # ax22 = fig.add_subplot(235)
    # ax23 = fig.add_subplot(236)

    features = ["Lmu", "Rf", "Rg"]
    features_tex = [r"$L_\mu$", r"$R_f$", r"$R_g$"]
    segment_type_map = {"outofplane_twist": "2D CANAL", "inplane_twist": "3D CANAL"}
    axs = [ax11, ax12, ax13]
    major_locator = [100, 0.1, 4]
    minor_locator = [25, 0.05, 2]
    for segment_type in ["inplane_twist", "outofplane_twist"]:
        if (segment_type == "inplane_twist"):
            # axs = [ax11, ax12, ax13]
            color = "royalblue"
            marker = "x"
        elif (segment_type == "outofplane_twist"):
            # axs = [ax21, ax22, ax23]
            color = "tomato"
            marker = "+"
        for i in range(len(features)):
            mu = features[i]
            ax = axs[i]
            mu, mu_predict, my_predict_err = np.loadtxt(f"{folder}/data_{segment_type}_{mu}_prediction.txt", skiprows=1, delimiter=",", unpack=True)
            ax.scatter(mu, mu_predict, color=color, marker=marker, s=4, alpha=0.8, lw=0.5, label=segment_type_map[segment_type])
            ax.legend(title=features_tex[i], fontsize=7, frameon=False, loc="upper left", handlelength=0.5, handletextpad=0.5)
            max_min = (np.max(mu)-np.min(mu))
            xlim = [np.min(mu)-0.1*max_min, np.max(mu)+0.1*max_min]
            ax.plot(xlim, xlim, "k-", linewidth=0.5, alpha=0.5)
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
            ax.xaxis.set_major_locator(plt.MultipleLocator(major_locator[i]))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_locator[i]))
            ax.yaxis.set_major_locator(plt.MultipleLocator(major_locator[i]))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_locator[i]))
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    axall = fig.add_subplot(111, frameon=False)
    axall.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    axall.set_xlabel("MC References", fontsize=9)
    axall.set_ylabel("ML Inversion", fontsize=9)

    # add annotation
    annotation = [r"$\mathbf{(%s)}$" % s for s in ["a", "b", "c"]]  # , "d", "e", "f"]]
    for ax in [ax11, ax12, ax13]:  # , ax21, ax22, ax23]:
        ax.text(0.8, 0.1, annotation.pop(0), fontsize=10, transform=ax.transAxes)

    plt.tight_layout(pad=0.05)
    plt.savefig("figures/GPR.pdf", format="pdf", dpi=300)

    plt.close()


# for suplemmentary material
def plot_PDDF_ACF_LML_data(tex_lw=455.24408, ppi=72):

    folder = "../data/20240713_partial"
    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.5))
    # for inplane twist
    ax11 = fig.add_subplot(241)
    ax12 = fig.add_subplot(242)
    ax13 = fig.add_subplot(243)
    ax14 = fig.add_subplot(244)

    # for outofplane twist
    ax21 = fig.add_subplot(245, sharex=ax11, sharey=ax11)
    ax22 = fig.add_subplot(246, sharex=ax12, sharey=ax12)
    ax23 = fig.add_subplot(247, sharex=ax13, sharey=ax13)
    ax24 = fig.add_subplot(248, sharex=ax14, sharey=ax14)

    features = ["Lmu", "Rf", "Rg"]
    features_tex = [r"$L\mu$", r"$R_f$", r"$R_g$"]
    for segment_type in ["inplane_twist", "outofplane_twist"]:
        if (segment_type == "inplane_twist"):
            axp = ax11
            axs = [ax12, ax13, ax14]
        elif (segment_type == "outofplane_twist"):
            axp = ax21
            axs = [ax22, ax23, ax24]

        # plot pddf
        data = np.loadtxt(f"{folder}/data_{segment_type}_pddf_acf.txt", skiprows=1, delimiter=",", unpack=True)
        z, p_z, acf_Lmu, acf_Lsig, acf_Lsig_per_Lmu, acf_Kt, acf_Kb, acf_Rf, acf_Rg = data
        max_z = 50
        nz = np.argmax(z >= max_z)

        p_z_max = np.max(p_z[:nz])
        axp.plot(z[:nz], p_z[:nz]/p_z_max, linewidth=1, label=r"$\frac{p(z)}{p_{max}(z)}$")
        axp.plot(z[:nz], acf_Lmu[:nz], linewidth=1, label=r"$C_{L_\mu}(z)$")
        axp.plot(z[:nz], acf_Rf[:nz], linewidth=1, label=r"$C_{R_f}(z)$")
        axp.plot(z[:nz], acf_Rg[:nz], linewidth=1, label=r"$C_{R_g}(z)$")
        axp.set_xlabel(r"$z$", fontsize=10, labelpad=0)
        axp.set_ylabel(r"$C(z)$", fontsize=10, labelpad=-3)
        axp.tick_params(labelsize=7, which="both", direction="in", top="on", right="on")
        axp.set_ylim([-1, 1.5])
        # if (segment_type == "inplane_twist"):
        # axp.set_xlabel(None)
        # axp.tick_params(labelbottom=False, labelleft=True)
        axp.legend(fontsize=7, frameon=False, ncol=2, columnspacing=0.2, handlelength=0.5, handletextpad=0.5)

        # plot log marginal likelihood contour
        for i in range(len(features)):
            mu = features[i]
            ax = axs[i]
            data = np.loadtxt(f"{folder}/data_{segment_type}_{mu}_LML.txt", skiprows=1, delimiter=",", unpack=True)
            gp_theta0, gp_theta1, theta0, theta1, LML = data[0][0], data[1][0], data[2], data[3], data[4:]
            print(segment_type, mu, gp_theta0, gp_theta1)
            print(f"gp_theta0={gp_theta0}, gp_theta1={gp_theta1}")
            Theta0, Theta1 = np.meshgrid(theta0, theta1)

            ax.contour(Theta0, Theta1, LML, levels=300)
            # ax.imshow(LML,extent=[Theta0.min(), Theta0.max(), Theta1.min(), Theta1.max()])
            ax.plot([gp_theta0], [gp_theta1], 'x', color='red', markersize=5, markeredgewidth=2)  # , label=r"l=%.2e, $\sigma$=%.2e" % (gp_theta0, gp_theta1))
            ax.set_xlabel(r"$l$", fontsize=10, labelpad=0)
            ax.set_ylabel(r"$\sigma$", fontsize=10, labelpad=-3)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=7, loc="upper left", title=features_tex[i], frameon=False)
            ax.tick_params(labelsize=7, which="both", direction="in", top="on", right="on")
            # if (segment_type == "inplane_twist"):
            # ax.set_xlabel(None)
            # ax.tick_params(labelbottom=False, labelleft=True)
            # if (i > 0):
            # ax.set_ylabel(None)
            # ax.tick_params(labelleft=False)

            # TODO: consider these later
            '''
            ax.xaxis.set_major_locator(plt.MultipleLocator(20))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
            ax.yaxis.set_major_locator(plt.MultipleLocator(20))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
            '''
    # add annotation
    annotation = [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$", r"$\mathbf{(c)}$", r"$\mathbf{(d)}$", r"$\mathbf{(e)}$", r"$\mathbf{(f)}$", r"$\mathbf{(g)}$", r"$\mathbf{(h)}$"]
    for ax in [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24]:
        ax.text(0.1, 0.1, annotation.pop(0), fontsize=10, transform=ax.transAxes)

    # plt.tight_layout(h_pad=0.01, w_pad=0.0)
    plt.tight_layout(pad=0.05, h_pad=-0.1, w_pad=-0.1)
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("figures/PDDF_ACF_LML.pdf", format="pdf", dpi=300)

    plt.close()
