import matplotlib.pyplot as plt
import numpy as np
import scipy


def ax_plot_polymer_config_line(ax, filename):
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    x, y, z = data[:, 2] - np.mean(data[:, 2]), data[:, 3] - \
        np.mean(data[:, 3]), data[:, 4] - np.mean(data[:, 4])
    ux, uy, uz = data[:, 5], data[:, 6], data[:, 7]
    vx, vy, vz = data[:, 8], data[:, 9], data[:, 10]

    d = 0.8
    for i in range(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color="royalblue", linewidth=1, alpha=0.7) # u: tangent
        ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i]]) # v

        ax.plot([x[i]-0.5*d*vx[i], x[i]-0.5*d*vx[i]+ux[i]], [y[i]-0.5*d*vy[i], y[i]-0.5*d*vy[i]+uy[i]], [z[i]-0.5*d*vz[i], z[i]-0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=3, alpha=1, solid_capstyle='round')

        # ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i] + uy[i], y[i]-0.5*d*vy[i]+uy[i], y[i]-0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]], color="royalblue", linewidth=1, alpha=0.7)

        # thicker side lines
        ax.plot([x[i]-0.5*d*vx[i], x[i]-0.5*d*vx[i]+ux[i]], [y[i]-0.5*d*vy[i], y[i]-0.5*d*vy[i]+uy[i]], [z[i]-0.5*d*vz[i], z[i]-0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=3, alpha=1, solid_capstyle='round')
        ax.plot([x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i]], [y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i]+uy[i]], [z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=3, alpha=1, solid_capstyle='round')

    # Set the same limits for x, y, and z axes
    max_range = max(max(x), max(y), max(z))
    min_range = min(min(x), min(y), min(z))
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])
    ax.set_aspect('equal')  # Set equal aspect ratio for all axes
    ax.set_axis_off()
    ax.grid(False)


def ax_plot_polymer_config_rectangle(ax, filename):
    # rectangle style representation of monomer
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    print(data)
    x, y, z = data[:, 2] - np.mean(data[:, 2]), data[:, 3] - \
        np.mean(data[:, 3]), data[:, 4] - np.mean(data[:, 4])
    ux, uy, uz = data[:, 5], data[:, 6], data[:, 7]
    vx, vy, vz = data[:, 8], data[:, 9], data[:, 10]
    print(x)

    # ax.scatter(x,y,z,color="black",s=3)
    d = 0.8
    for i in range(len(x)-1):
        ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i] + uy[i], y[i]-0.5*d*vy[i]+uy[i], y[i]-0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]], color="royalblue", linewidth=1, alpha=0.7)

        # thicker side lines
        ax.plot([x[i]-0.5*d*vx[i], x[i]-0.5*d*vx[i]+ux[i]], [y[i]-0.5*d*vy[i], y[i]-0.5*d*vy[i]+uy[i]], [z[i]-0.5*d*vz[i], z[i]-0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=3, alpha=1, solid_capstyle='round')
        ax.plot([x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i]], [y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i]+uy[i]], [z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=3, alpha=1, solid_capstyle='round')

    # Set the same limits for x, y, and z axes
    max_range = max(max(x), max(y), max(z))
    min_range = min(min(x), min(y), min(z))
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])
    ax.set_aspect('equal')  # Set equal aspect ratio for all axes
    ax.set_axis_off()
    ax.grid(False)


def plot_polymer_configs():
    ppi = 72
    fig = plt.figure(figsize=(246 / ppi * 1, 246 / ppi * 1))
    axs = fig.subplots(2, 2, subplot_kw={'projection': '3d'})
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # Call the ax_plot_polymer_config function for each subplot
    ax_plot_polymer_config(axs[0, 0], "../data/scratch_local/20240612/config_inplane_twist_L20_logKt2.0_logKb2.0_Rf0.050.csv")
    ax_plot_polymer_config(axs[0, 1], "../data/scratch_local/20240612/config_inplane_twist_L20_logKt2.0_logKb2.0_Rf1.000.csv")
    ax_plot_polymer_config(axs[1, 0], "../data/scratch_local/20240612/config_outofplane_twist_L20_logKt2.0_logKb2.0_Rf0.050.csv")
    ax_plot_polymer_config(axs[1, 1], "../data/scratch_local/20240612/config_outofplane_twist_L20_logKt2.0_logKb2.0_Rf1.000.csv")

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()


def plot_animate_polymer_config(filename, tag):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax_plot_polymer_config_rectangle(ax, filename)

    for ii in range(40):
        if (ii < 15):
            ax.view_init(elev=18., azim=18*ii)
        else:
            ax.view_init(elev=18.*(ii-14), azim=18*14)
        plt.tight_layout()
        plt.savefig(f"figures/config_gif/{tag}_demo_ani%d" % ii+".png", dpi=300)
    plt.close()


def plot_animate_polymer_configs():
    plot_animate_polymer_config("../data/scratch_local/20240612/config_inplane_twist_L20_logKt2.0_logKb2.0_Rf0.050.csv", "inplane_lowRf")
    plot_animate_polymer_config("../data/scratch_local/20240612/config_inplane_twist_L20_logKt2.0_logKb2.0_Rf1.000.csv", "inplane_highRf")
    plot_animate_polymer_config("../data/scratch_local/20240612/config_outofplane_twist_L20_logKt2.0_logKb2.0_Rf0.050.csv", "outofplane_lowRf")
    plot_animate_polymer_config("../data/scratch_local/20240612/config_outofplane_twist_L20_logKt2.0_logKb2.0_Rf1.000.csv", "outofplane_highRf")
