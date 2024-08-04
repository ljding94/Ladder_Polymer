import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.patches as mpatches


def ax_plot_polymer_config_line(ax, filename):
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    x, y, z = data[:, 2] - np.mean(data[:, 2]), data[:, 3] - \
        np.mean(data[:, 3]), data[:, 4] - np.mean(data[:, 4])
    ux, uy, uz = data[:, 5], data[:, 6], data[:, 7]
    vx, vy, vz = data[:, 8], data[:, 9], data[:, 10]

    d = 0.8
    for i in range(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color="royalblue", linewidth=0.6, alpha=0.7)  # u: tangent
        ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i]])  # v

        ax.plot([x[i]-0.5*d*vx[i], x[i]-0.5*d*vx[i]+ux[i]], [y[i]-0.5*d*vy[i], y[i]-0.5*d*vy[i]+uy[i]], [z[i]-0.5*d*vz[i], z[i]-0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')

        # ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i] + uy[i], y[i]-0.5*d*vy[i]+uy[i], y[i]-0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]], color="royalblue", linewidth=0.6, alpha=0.7)

        # thicker side lines
        ax.plot([x[i]-0.5*d*vx[i], x[i]-0.5*d*vx[i]+ux[i]], [y[i]-0.5*d*vy[i], y[i]-0.5*d*vy[i]+uy[i]], [z[i]-0.5*d*vz[i], z[i]-0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')
        ax.plot([x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i]], [y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i]+uy[i]], [z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')

    # Set the same limits for x, y, and z axes
    max_range = max(max(x), max(y), max(z))
    min_range = min(min(x), min(y), min(z))
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])
    ax.set_aspect('equal')  # Set equal aspect ratio for all axes
    ax.set_axis_off()
    ax.grid(False)


def ax_plot_polymer_config_rectangle(ax, filename, elev=30, zaim=30):
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
        ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i] + uy[i], y[i]-0.5*d*vy[i]+uy[i], y[i]-0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]], color="royalblue", linewidth=0.6, alpha=0.7)

        # thicker side lines
        ax.plot([x[i]-0.5*d*vx[i], x[i]-0.5*d*vx[i]+ux[i]], [y[i]-0.5*d*vy[i], y[i]-0.5*d*vy[i]+uy[i]], [z[i]-0.5*d*vz[i], z[i]-0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')
        ax.plot([x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i]], [y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i]+uy[i]], [z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i]], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')

    # Set the same limits for x, y, and z axes
    max_range = max(max(x), max(y), max(z))
    min_range = min(min(x), min(y), min(z))
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])
    ax.view_init(elev=elev, azim=zaim)
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


def ax_plot_outofplane_twist_monomer(ax, theta, dv, dt):
    du = 1 - dt*np.cos(theta)
    # v line
    ax.plot([0, 0], [0, 1*dv], [0, 0], color="royalblue", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 1
    # u line
    ax.plot([0, du], [0, 0], [0, 0], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    ax.plot([0, du], [1*dv, 1*dv], [0, 0], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    # another v line
    ax.plot([du, du], [0, 1*dv], [0, 0], color="royalblue", linestyle="--", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 1
    # ut line
    ax.plot([du, 1], [0, 0], [0, dt*np.sin(theta)], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    ax.plot([du, 1], [1*dv, 1*dv], [0, dt*np.sin(theta)], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    # vt line
    ax.plot([1, 1], [0, 1*dv], [dt*np.sin(theta), dt*np.sin(theta)], color="royalblue", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 2

    d_text = 0.05
    # u and v
    qd = 0.4*dv  # quiver length
    ax.quiver(0, 0.5*dv, 0, qd, 0, 0, color="black", arrow_length_ratio=0.5)
    ax.text(qd, 0.5*dv, 0+d_text, r"$\hat{\mathrm{u}}$", fontsize=9, color="black")
    ax.quiver(0, 0.5*dv, 0, 0, qd, 0, color="black", arrow_length_ratio=0.5)
    ax.text(0, 0.5*dv+qd, 0+d_text, r"$\hat{\mathrm{v}}$", fontsize=9, color="black")

    # ut and vt
    ax.quiver(1, 0.5*dv, dt*np.sin(theta), qd*np.cos(theta), 0, qd*np.sin(theta), color="black", arrow_length_ratio=0.5)
    ax.text(1+qd*np.cos(theta), 0.5*dv, dt*np.sin(theta)+qd*np.sin(theta)+d_text, r"$\hat{\mathrm{u}}_t$", fontsize=9, color="black")
    ax.quiver(1, 0.5*dv, dt*np.sin(theta), 0, qd, 0, color="black", arrow_length_ratio=0.5)
    ax.text(1, 0.5*dv+qd, dt*np.sin(theta)+d_text, r"$\hat{\mathrm{v}}_t$", fontsize=9, color="black")

    ax.view_init(elev=30., azim=-140)
    ax.set_aspect('equal')
    ax.set_axis_off()


def ax_plot_inplane_twist_monomer(ax, theta, dv, dt):
    dtot = 1.5
    du = dtot - dt*np.cos(theta)
    dud = dtot - (dv-dt*np.sin(theta))*np.tan(theta)  # du of the lower edge
    # v line
    ax.plot([0, 0], [0, 1*dv], [0, 0], color="royalblue", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 1
    # u line
    ax.plot([0, dud], [0, 0], [0, 0], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    ax.plot([0, du], [1*dv, 1*dv], [0, 0], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    # another v line
    ax.plot([dud, du], [0, 1*dv], [0, 0], color="royalblue", linestyle="--", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 1
    # ut line
    ax.plot([du, dtot], [1*dv, 1*dv-dt*np.sin(theta)], [0, 0], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    # vt line
    ax.plot([dud, dtot], [0, 1*dv-dt*np.sin(theta)], [0, 0], color="royalblue", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 2

    d_text = 0.05
    # u and v
    qd = 0.4*dv  # quiver length
    ax.quiver(0, 0.5*dv, 0, qd, 0, 0, color="black", arrow_length_ratio=0.5)
    ax.text(qd, 0.5*dv, 0+d_text, r"$\hat{\mathrm{u}}$", fontsize=9, color="black")
    ax.quiver(0, 0.5*dv, 0, 0, qd, 0, color="black", arrow_length_ratio=0.5)
    ax.text(0, 0.5*dv+qd, 0+d_text, r"$\hat{\mathrm{v}}$", fontsize=9, color="black")

    # ut and vt
    dumid = 0.5*(dud+dtot)
    dvmid = 0.5*(dv-dt*np.sin(theta))
    ax.quiver(dumid, dvmid, 0, qd*np.cos(theta), -qd*np.sin(theta), 0, color="black", arrow_length_ratio=0.5)
    ax.text(dumid+qd*np.cos(theta), dvmid-qd*np.sin(theta), 0+d_text, r"$\hat{\mathrm{u}}_t$", fontsize=9, color="black")
    ax.quiver(dumid, dvmid, 0, qd*np.sin(theta), qd*np.cos(theta), 0, color="black", arrow_length_ratio=0.5)
    ax.text(dumid+qd*np.sin(theta), dvmid+qd*np.cos(theta), 0+d_text, r"$\hat{\mathrm{v}}_t$", fontsize=9, color="black")

    ax.view_init(elev=60., azim=-125)
    ax.set_aspect('equal')
    ax.set_axis_off()


def plot_illustrative_config(tex_lw=455.24408, ppi=72):

    fig = plt.figure(figsize=(tex_lw / ppi * 0.5, tex_lw / ppi * 0.3))
    # plt.rc("text", usetex=True)
    # plt.rc("text.latex", preamble=r"\usepackage{physics}")
    # for outofplane twist (2D CANAL)
    ax11 = fig.add_subplot(231, projection='3d')
    ax12 = fig.add_subplot(232, projection='3d')
    ax13 = fig.add_subplot(233, projection='3d')

    # for inplane twist (3D CANAL)
    ax21 = fig.add_subplot(234, projection='3d')
    ax22 = fig.add_subplot(235, projection='3d')
    ax23 = fig.add_subplot(236, projection='3d')

    # outogplane twist but just single monomer for illustration
    theta = 51/180*np.pi
    dv = 0.7
    dt = 0.1
    ax_plot_outofplane_twist_monomer(ax11, theta, dv, dt)
    ax_plot_inplane_twist_monomer(ax21, theta, dv, dt)

    ax_plot_polymer_config_rectangle(ax12, "../data/scratch_local/20240612/config_outofplane_twist_L20_logKt2.0_logKb2.0_Rf0.050.csv")
    ax_plot_polymer_config_rectangle(ax13, "../data/scratch_local/20240612/config_outofplane_twist_L20_logKt2.0_logKb2.0_Rf0.900.csv", 160, 30)
    ax_plot_polymer_config_rectangle(ax22, "../data/scratch_local/20240612/config_inplane_twist_L20_logKt2.0_logKb2.0_Rf0.050.csv", -13, 123)
    ax_plot_polymer_config_rectangle(ax23, "../data/scratch_local/20240612/config_inplane_twist_L20_logKt2.0_logKb2.0_Rf0.900.csv", -30, 85)

    # Add annotations to each ax in global figure coordinate
    axs = [ax11, ax12, ax13, ax21, ax22, ax23]
    anno = [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$", r"$\mathbf{(c)}$", r"$\mathbf{(d)}$", r"$\mathbf{(e)}$", r"$\mathbf{(f)}$"]
    for i in range(len(axs)):
        ax = axs[i]
        bbox = ax.get_position()
        x = bbox.x0 + bbox.width*(0.0)
        y = bbox.y0 + bbox.height*(0.9)
        fig.text(x, y, anno[i], ha='center', va='center', fontsize=9)

    plt.tight_layout(pad=0.1, h_pad=-1.4, w_pad=-1)
    # plt.tight_layout(h_pad=-2, w_pad=-6)
    #plt.show()
    plt.savefig("figures/config_demo.pdf", format="pdf")
    plt.close()
