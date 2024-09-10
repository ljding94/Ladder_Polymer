import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.patches as mpatches
import pandas as pd


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
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
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
    # plt.show()
    plt.savefig("figures/config_demo.pdf", format="pdf")
    plt.close()


def ax_plot_outofplane_twist_segment(ax, r, u, v, ut, vt, dv, dt, flip="None", label=""):

    # u line
    ax.plot([r[0]+u[0]*dt, r[0]+u[0]], [r[1]+u[1]*dt, r[1]+u[1]], [r[2]+u[2]*dt, r[2]+u[2]], color="gray", linewidth=1.5, alpha=1, solid_capstyle='round')
    ax.plot([r[0]+u[0]*dt+v[0]*dv, r[0]+u[0]+v[0]*dv], [r[1]+u[1]*dt+v[1]*dv, r[1]+u[1]+v[1]*dv], [r[2]+u[2]*dt+v[2]*dv, r[2]+u[2]+v[2]*dv], color="gray", linewidth=1.5, alpha=1, solid_capstyle='round')
    # v line
    color = "gray"
    ls = "-"
    if (flip == 0):
        color = "blue"
    if (flip == 1):
        color = "red"

    if (label != ""):
        ax.plot([r[0]+u[0]*dt, r[0]+u[0]*dt+v[0]*dv], [r[1]+u[1]*dt, r[1]+u[1]*dt+v[1]*dv], [r[2]+u[2]*dt, r[2]+u[2]*dt+v[2]*dv], color=color, linewidth=2, alpha=1, solid_capstyle='round', ls=ls, label=label)
    else:
        ax.plot([r[0]+u[0]*dt, r[0]+u[0]*dt+v[0]*dv], [r[1]+u[1]*dt, r[1]+u[1]*dt+v[1]*dv], [r[2]+u[2]*dt, r[2]+u[2]*dt+v[2]*dv], color=color, linewidth=2, alpha=1, solid_capstyle='round', ls=ls)
    # bend v line
    ax.plot([r[0]+u[0], r[0]+u[0]+v[0]*dv], [r[1]+u[1], r[1]+u[1]+v[1]*dv], [r[2]+u[2], r[2]+u[2]+v[2]*dv], "--", color="gray", linewidth=0.5, alpha=0.5, solid_capstyle='round')

    # ut line
    ax.plot([r[0]+u[0], r[0]+u[0]+dt*ut[0]], [r[1]+u[1], r[1]+u[1]+dt*ut[1]], [r[2]+u[2], r[2]+u[2]+dt*ut[2]], color="gray", linewidth=1.5, alpha=1, solid_capstyle='round')
    # ut line
    ax.plot([r[0]+u[0]+v[0]*dv, r[0]+u[0]+dt*ut[0]+v[0]*dv], [r[1]+u[1]+v[1]*dv, r[1]+u[1]+dt*ut[1]+v[1]*dv], [r[2]+u[2]+v[2]*dv, r[2]+u[2]+dt*ut[2]+v[2]*dv], color="gray", linewidth=1.5, alpha=1, solid_capstyle='round')
    # vt line
    ax.plot([r[0]+u[0]+dt*ut[0], r[0]+u[0]+dt*ut[0]+v[0]*dv], [r[1]+u[1]+dt*ut[1], r[1]+u[1]+dt*ut[1]+v[1]*dv], [r[2]+u[2]+dt*ut[2], r[2]+u[2]+dt*ut[2]+v[2]*dv], color="gray", linewidth=1.5, alpha=1, solid_capstyle='round')

    # annotation
    # if (flip != "None" ):
    #    ax.text(r[0]+u[0]+dv*v[0]-0.1, r[1]+u[1]+dv*v[1], r[2]+u[2]+dv*v[2]+0.1, fr"{flip}", fontsize=9, color="black")

    # ax.plot([dud, du], [0, 1*dv], [0, 0], color="royalblue", linestyle="--", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 1
    # ut line
    # ax.plot([du, dtot], [1*dv, 1*dv-dt*np.sin(theta)], [0, 0], color="royalblue", linewidth=1.5, alpha=1, solid_capstyle='round')  # edge 2
    # vt line
    # ax.plot([dud, dtot], [0, 1*dv-dt*np.sin(theta)], [0, 0], color="royalblue", linewidth=0.6, alpha=0.7, solid_capstyle='round')  # edge 2

    ax.set_aspect('equal')
    ax.set_axis_off()


def generate_outofplane_twist_polymer(flips):
    alpha = 51/180*np.pi
    r, u, v, ut, vt = [], [], [], [], []
    r = [np.array([0, 0, 0])]
    u = [np.array([1, 0, 0])]
    v = [np.array([0, 1, 0])]
    w = np.cross(u[-1], v[-1])
    ut.append(u[-1]*np.cos(alpha)+w*np.sin(alpha))
    vt.append(v[-1])
    for i in range(len(flips)):
        r.append(r[-1]+u[-1])
        u.append(ut[-1])
        v.append(vt[-1])
        alpha = -alpha if flips[i] == 1 else alpha
        w = np.cross(u[-1], v[-1])
        ut.append(u[-1]*np.cos(alpha)+w*np.sin(alpha))
        vt.append(v[-1])
    return r, u, v, ut, vt


def plot_flipping_demo(tex_lw=455.24408, ppi=72):

    flip = [1, 1, 1, 0, 0, 0, 1]
    r, u, v, ut, vt = generate_outofplane_twist_polymer(flip)
    print("len(r)", len(r))
    print("len(flip)", len(flip))
    print("r", r)
    print("u", u)
    print("v", v)
    print("ut", ut)
    print("vt", vt)

    fig = plt.figure(figsize=(tex_lw / ppi * 0.5, tex_lw / ppi * 0.4))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    # for outofplane twist (2D CANAL)
    ax = fig.add_subplot(111, projection='3d')
    all_flip = ["None"] + flip
    first_blue = 0
    first_red = 0

    for i in range(len(r)):
        label = ""
        if (all_flip[i] == 0):
            color = "blue"
            if (first_blue == 0):
                first_blue = 1
                label = "0"
        if (all_flip[i] == 1):
            color = "red"
            if (first_red == 0):
                first_red = 1
                label = "1"
        ax_plot_outofplane_twist_segment(ax, r[i], u[i], v[i], ut[i], vt[i], 0.8, 0.1, all_flip[i], label)
    ax.legend(loc="center left", title=r"flip", ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)
    ax.view_init(elev=30., azim=-140)
    plt.tight_layout(pad=-2)
    plt.savefig("figures/flipping_demo.pdf", format="pdf")

    plt.show()

    plt.close()


def sample_plot_2d_canel(tex_lw=240.71031, ppi=72):
    print("sample plotting 2d canal")
    atom_file = "./2D_CANEL_corrd/2d_canal2_atom.txt"
    bond_file = "./2D_CANEL_corrd/2d_canal2_bond.txt"

    atom_data = pd.read_csv(atom_file, sep=" ", skiprows=1)
    atom_data["atom_color"] = "orange"
    atom_data.loc[atom_data["atom_name"] == "H", "atom_color"] = "lime"

    bond_data = pd.read_csv(bond_file, sep=" ", skiprows=1)

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.4))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    for i in range(len(bond_data)):
        bond_x = [atom_data["x"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["x"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        bond_y = [atom_data["y"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["y"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        bond_z = [atom_data["z"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["z"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        ax.plot(bond_x, bond_y, bond_z, color="gray", lw=1, alpha=0.75)
    ax.scatter(atom_data["x"][atom_data["atom_name"] == "H"], atom_data["y"][atom_data["atom_name"] == "H"], atom_data["z"][atom_data["atom_name"] == "H"], marker="o", facecolors="lime", s=20, label="Hydrogen")
    ax.scatter(atom_data["x"][atom_data["atom_name"] == "C"], atom_data["y"][atom_data["atom_name"] == "C"], atom_data["z"][atom_data["atom_name"] == "C"], marker="o", facecolors="orange", s=20, label="Carbon")
    # for i in range(len(atom_data)):
    # ax.text(atom_data["x"][i], atom_data["y"][i], atom_data["z"][i], atom_data["atom_id"][i], fontsize=9)

    ax.set_aspect('equal')
    ax.view_init(elev=30., azim=110)
    ax.set_axis_off()

    r0 = np.array([(atom_data["x"][atom_data["atom_id"] == 11].values[0] + atom_data["x"][atom_data["atom_id"] == 12].values[0]) * 0.5,
                   (atom_data["y"][atom_data["atom_id"] == 11].values[0] + atom_data["y"][atom_data["atom_id"] == 12].values[0]) * 0.5,
                   (atom_data["z"][atom_data["atom_id"] == 11].values[0] + atom_data["z"][atom_data["atom_id"] == 12].values[0]) * 0.5])

    v0 = np.array([atom_data["x"][atom_data["atom_id"] == 12].values[0] - atom_data["x"][atom_data["atom_id"] == 11].values[0],
                   atom_data["y"][atom_data["atom_id"] == 12].values[0] - atom_data["y"][atom_data["atom_id"] == 11].values[0],
                   atom_data["z"][atom_data["atom_id"] == 12].values[0] - atom_data["z"][atom_data["atom_id"] == 11].values[0]])

    r1 = np.array([(atom_data["x"][atom_data["atom_id"] == 32].values[0] + atom_data["x"][atom_data["atom_id"] == 33].values[0]) * 0.5,
                   (atom_data["y"][atom_data["atom_id"] == 32].values[0] + atom_data["y"][atom_data["atom_id"] == 33].values[0]) * 0.5,
                   (atom_data["z"][atom_data["atom_id"] == 32].values[0] + atom_data["z"][atom_data["atom_id"] == 33].values[0]) * 0.5])
    v1 = np.array([atom_data["x"][atom_data["atom_id"] == 33].values[0] - atom_data["x"][atom_data["atom_id"] == 32].values[0],
                   atom_data["y"][atom_data["atom_id"] == 33].values[0] - atom_data["y"][atom_data["atom_id"] == 32].values[0],
                   atom_data["z"][atom_data["atom_id"] == 33].values[0] - atom_data["z"][atom_data["atom_id"] == 32].values[0]])

    r2 = np.array([(atom_data["x"][atom_data["atom_id"] == 53].values[0] + atom_data["x"][atom_data["atom_id"] == 54].values[0]) * 0.5,
                   (atom_data["y"][atom_data["atom_id"] == 53].values[0] + atom_data["y"][atom_data["atom_id"] == 54].values[0]) * 0.5,
                   (atom_data["z"][atom_data["atom_id"] == 53].values[0] + atom_data["z"][atom_data["atom_id"] == 54].values[0]) * 0.5])
    v2 = np.array([atom_data["x"][atom_data["atom_id"] == 54].values[0] - atom_data["x"][atom_data["atom_id"] == 53].values[0],
                   atom_data["y"][atom_data["atom_id"] == 54].values[0] - atom_data["y"][atom_data["atom_id"] == 53].values[0],
                   atom_data["z"][atom_data["atom_id"] == 54].values[0] - atom_data["z"][atom_data["atom_id"] == 53].values[0]])
    r3 = np.array([(atom_data["x"][atom_data["atom_id"] == 74].values[0] + atom_data["x"][atom_data["atom_id"] == 75].values[0]) * 0.5,
                   (atom_data["y"][atom_data["atom_id"] == 74].values[0] + atom_data["y"][atom_data["atom_id"] == 75].values[0]) * 0.5,
                   (atom_data["z"][atom_data["atom_id"] == 74].values[0] + atom_data["z"][atom_data["atom_id"] == 75].values[0]) * 0.5])

    d = 2
    e1 = r0-v0*0.5*d
    e2 = r1-v0*0.5*d
    e3 = r1+v0*0.5*d
    e4 = r0+v0*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=200)

    e1 = r1-v1*0.5*d
    e2 = r2-v1*0.5*d
    e3 = r2+v1*0.5*d
    e4 = r1+v1*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="blue", linewidth=4, alpha=1, solid_capstyle='round', zorder=200, label="flip")

    e1 = r2-v2*0.5*d
    e2 = r3-v2*0.5*d
    e3 = r3+v2*0.5*d
    e4 = r2+v2*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="blue", linewidth=4, alpha=1, solid_capstyle='round', zorder=200)
    ax.plot([e2[0], e3[0]], [e2[1], e3[1]], [e2[2], e3[2]], color="black", linewidth=2, alpha=1, solid_capstyle='round', zorder=200)
    ax.legend(frameon=False, fontsize=9)
    print("r0", r0)
    print("r1", r1)
    print("r2", r2)

    u = r1-r0
    lb = np.sqrt(np.sum(u**2))
    u = u/np.sqrt(np.sum(u**2))
    ut = r2-r1
    ut = ut/np.sqrt(np.sum(ut**2))
    print("lb", lb)
    print("u", u)
    print("ut", ut)
    print("np.dot(u,ut)", np.dot(u, ut))
    print("np.arccos(np.dot(u,ut))", np.arccos(np.dot(u, ut)))
    print("np.arccos(np.dot(u,ut))/np.pi*180", np.arccos(np.dot(u, ut))/np.pi*180)  # 48.61354128560861

    plt.tight_layout()
    plt.show()
    plt.close()


def ax_plot_2d_canel_nonflip(ax, xshift=0, yshift=0, zshift=0):
    print("sample plotting 2d canal, flip")
    atom_file = "./2D_CANEL_corrd/2d_canal2_atom.txt"
    bond_file = "./2D_CANEL_corrd/2d_canal2_bond.txt"
    atom_data = pd.read_csv(atom_file, sep=" ", skiprows=1)
    bond_data = pd.read_csv(bond_file, sep=" ", skiprows=1)
    atom_data["x"] = atom_data["x"] - np.mean(atom_data["x"])
    atom_data["y"] = atom_data["y"] - np.mean(atom_data["y"])
    atom_data["z"] = atom_data["z"] - np.mean(atom_data["z"])

    atom_data["x"] = atom_data["x"] + xshift
    atom_data["y"] = atom_data["y"] + yshift
    atom_data["z"] = atom_data["z"] + zshift

    for i in range(len(bond_data)):
        bond_x = [atom_data["x"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["x"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        bond_y = [atom_data["y"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["y"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        bond_z = [atom_data["z"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["z"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        ax.plot(bond_x, bond_y, bond_z, color="gray", lw=0.5, solid_capstyle='round', alpha=0.75)
    ax.scatter(atom_data["x"][atom_data["atom_name"] == "H"], atom_data["y"][atom_data["atom_name"] == "H"], atom_data["z"][atom_data["atom_name"] == "H"], marker="o", facecolors="lime", s=10, label="Hydrogen")
    ax.scatter(atom_data["x"][atom_data["atom_name"] == "C"], atom_data["y"][atom_data["atom_name"] == "C"], atom_data["z"][atom_data["atom_name"] == "C"], marker="o", facecolors="orange", s=10, label="Carbon")
    #for i in range(len(atom_data)):
        #ax.text(atom_data["x"][i], atom_data["y"][i], atom_data["z"][i], atom_data["atom_id"][i], fontsize=9)

    r_t = get_atom_pos(atom_data, 9, 10)
    r0_h = get_atom_pos(atom_data, 13, 15)
    r0_t = get_atom_pos(atom_data, 30, 31)
    u0_ht = r0_t - r0_h
    lb0 = np.sqrt(np.sum(u0_ht**2))
    u0 = u0_ht/lb0
    v_th = r0_h - r_t
    l_th = np.sqrt(np.sum(v_th**2))
    v_th = v_th/l_th
    dth = l_th*0.5/np.dot(v_th, u0)
    print("dth", dth)
    r0 = r0_h-dth*u0
    u0 = u0*(2*dth+lb0)
    v0 = get_atom_pos_diff(atom_data, 13, 15)

    r1_h = get_atom_pos(atom_data, 34, 36)
    r1_t = get_atom_pos(atom_data, 51, 52)
    u1_ht = r1_t - r1_h
    lb1 = np.sqrt(np.sum(u1_ht**2))
    u1 = u1_ht/lb1
    r1 = r1_h-dth*u1
    u1 = u1*(2*dth+lb1)
    v1 = get_atom_pos_diff(atom_data, 34, 36)

    r2_h = get_atom_pos(atom_data, 55, 57)
    r2_t = get_atom_pos(atom_data, 72, 73)
    u2_ht = r2_t - r2_h
    lb2 = np.sqrt(np.sum(u2_ht**2))
    u2 = u2_ht/lb2
    r2 = r2_h-dth*u2
    u2 = u2*(2*dth+lb2)
    v2 = get_atom_pos_diff(atom_data, 55, 57)

    d = 2
    e1 = r0-v0*0.5*d
    e2 = r0-v0*0.5*d+u0
    e3 = r0+v0*0.5*d+u0
    e4 = r0+v0*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=200)
    ax.plot([e2[0], e3[0]], [e2[1], e3[1]], [e2[2], e3[2]], color="blue", linewidth=2, alpha=1, solid_capstyle='round', zorder=200, label="non-flip")

    e1 = r1-v1*0.5*d
    e2 = r1-v1*0.5*d+u1
    e3 = r1+v1*0.5*d+u1
    e4 = r1+v1*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="blue", linewidth=2, alpha=1, solid_capstyle='round', zorder=200)

    e1 = r2-v2*0.5*d
    e2 = r2-v2*0.5*d+u2
    e3 = r2+v2*0.5*d+u2
    e4 = r2+v2*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="blue", linewidth=2, alpha=1, solid_capstyle='round', zorder=200)
    ax.plot([e2[0], e3[0]], [e2[1], e3[1]], [e2[2], e3[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=200)

    print("lb", lb0+2*dth)
    u0 = u0/np.sqrt(np.sum(u0**2))
    u1 = u1/np.sqrt(np.sum(u1**2))
    print("u0", u0)
    print("u1", u1)
    print("np.dot(u0,u1)", np.dot(u0, u1))
    print("np.arccos(np.dot(u0,u1))", np.arccos(np.dot(u0, u1)))
    print("np.arccos(np.dot(u0,u1))/np.pi*180", np.arccos(np.dot(u0, u1))/np.pi*180)  # 48.61354128560861

def ax_plot_2d_canel_flip(ax, xshift=0, yshift=0, zshift=0):
    print("sample plotting 2d canal, flip")
    atom_file = "./2D_CANEL_corrd/2d_canal2f_atom.txt"
    bond_file = "./2D_CANEL_corrd/2d_canal2f_bond.txt"
    atom_data = pd.read_csv(atom_file, sep=" ", skiprows=1)
    atom_data["x"] = atom_data["x"] - np.mean(atom_data["x"])
    atom_data["y"] = atom_data["y"] - np.mean(atom_data["y"])
    atom_data["z"] = atom_data["z"] - np.mean(atom_data["z"])
    bond_data = pd.read_csv(bond_file, sep=" ", skiprows=1)
    atom_data["x"] = atom_data["x"] + xshift
    atom_data["y"] = atom_data["y"] + yshift
    atom_data["z"] = atom_data["z"] + zshift

    for i in range(len(bond_data)):
        bond_x = [atom_data["x"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["x"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        bond_y = [atom_data["y"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["y"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        bond_z = [atom_data["z"][atom_data["atom_id"] == bond_data["origin_atom_id"][i]], atom_data["z"][atom_data["atom_id"] == bond_data["target_atom_id"][i]]]
        ax.plot(bond_x, bond_y, bond_z, color="gray", lw=0.5, solid_capstyle='round', alpha=0.75)
    ax.scatter(atom_data["x"][atom_data["atom_name"] == "H"], atom_data["y"][atom_data["atom_name"] == "H"], atom_data["z"][atom_data["atom_name"] == "H"], marker="o", facecolors="lime", s=10)
    ax.scatter(atom_data["x"][atom_data["atom_name"] == "C"], atom_data["y"][atom_data["atom_name"] == "C"], atom_data["z"][atom_data["atom_name"] == "C"], marker="o", facecolors="orange", s=10)
    # for i in range(len(atom_data)):
    # ax.text(atom_data["x"][i], atom_data["y"][i], atom_data["z"][i], atom_data["atom_id"][i], fontsize=9)

    r_t = get_atom_pos(atom_data, 9, 10)
    r0_h = get_atom_pos(atom_data, 13, 15)
    r0_t = get_atom_pos(atom_data, 31, 30)
    u0_ht = r0_t - r0_h
    lb0 = np.sqrt(np.sum(u0_ht**2))
    u0 = u0_ht/lb0
    v_th = r0_h - r_t
    l_th = np.sqrt(np.sum(v_th**2))
    v_th = v_th/l_th
    dth = l_th*0.5/np.dot(v_th, u0)
    print("dth", dth)
    r0 = r0_h-dth*u0
    u0 = u0*(2*dth+lb0)
    v0 = get_atom_pos_diff(atom_data, 13, 15)

    r1_h = get_atom_pos(atom_data, 36, 34)
    r1_t = get_atom_pos(atom_data, 55, 56)
    u1_ht = r1_t - r1_h
    lb1 = np.sqrt(np.sum(u1_ht**2))
    u1 = u1_ht/lb1
    r1 = r1_h-dth*u1
    u1 = u1*(2*dth+lb1)
    v1 = get_atom_pos_diff(atom_data, 36, 34)

    r2_h = get_atom_pos(atom_data, 59, 61)
    r2_t = get_atom_pos(atom_data, 77, 76)
    u2_ht = r2_t - r2_h
    lb2 = np.sqrt(np.sum(u2_ht**2))
    u2 = u2_ht/lb2
    r2 = r2_h-dth*u2
    u2 = u2*(2*dth+lb2)
    v2 = get_atom_pos_diff(atom_data, 59, 61)

    d = 2
    e1 = r0-v0*0.5*d
    e2 = r0-v0*0.5*d+u0
    e3 = r0+v0*0.5*d+u0
    e4 = r0+v0*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=200)
    ax.plot([e2[0], e3[0]], [e2[1], e3[1]], [e2[2], e3[2]], color="red", linewidth=2, alpha=1, solid_capstyle='round', zorder=200, label="flip")

    e1 = r1-v1*0.5*d
    e2 = r1-v1*0.5*d+u1
    e3 = r1+v1*0.5*d+u1
    e4 = r1+v1*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="red", linewidth=2, alpha=1, solid_capstyle='round', zorder=200)

    e1 = r2-v2*0.5*d
    e2 = r2-v2*0.5*d+u2
    e3 = r2+v2*0.5*d+u2
    e4 = r2+v2*0.5*d
    ax.plot([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e3[0], e4[0]], [e3[1], e4[1]], [e3[2], e4[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=100)
    ax.plot([e1[0], e4[0]], [e1[1], e4[1]], [e1[2], e4[2]], color="red", linewidth=2, alpha=1, solid_capstyle='round', zorder=200)
    ax.plot([e2[0], e3[0]], [e2[1], e3[1]], [e2[2], e3[2]], color="black", linewidth=1, alpha=1, solid_capstyle='round', zorder=200)

    print("lb", lb0+2*dth)
    u0 = u0/np.sqrt(np.sum(u0**2))
    u1 = u1/np.sqrt(np.sum(u1**2))
    print("u0", u0)
    print("u1", u1)
    print("np.dot(u0,u1)", np.dot(u0, u1))
    print("np.arccos(np.dot(u0,u1))", np.arccos(np.dot(u0, u1)))
    print("np.arccos(np.dot(u0,u1))/np.pi*180", np.arccos(np.dot(u0, u1))/np.pi*180)  # 48.61354128560861


def get_atom_pos(atom_data, atom_1, atom_2):
    return np.array([(atom_data["x"][atom_data["atom_id"] == atom_1].values[0] + atom_data["x"][atom_data["atom_id"] == atom_2].values[0]) * 0.5,
                     (atom_data["y"][atom_data["atom_id"] == atom_1].values[0] + atom_data["y"][atom_data["atom_id"] == atom_2].values[0]) * 0.5,
                     (atom_data["z"][atom_data["atom_id"] == atom_1].values[0] + atom_data["z"][atom_data["atom_id"] == atom_2].values[0]) * 0.5])


def get_atom_pos_diff(atom_data, atom_1, atom_2):
    return np.array([(atom_data["x"][atom_data["atom_id"] == atom_1].values[0] - atom_data["x"][atom_data["atom_id"] == atom_2].values[0]),
                     (atom_data["y"][atom_data["atom_id"] == atom_1].values[0] - atom_data["y"][atom_data["atom_id"] == atom_2].values[0]),
                     (atom_data["z"][atom_data["atom_id"] == atom_1].values[0] - atom_data["z"][atom_data["atom_id"] == atom_2].values[0])])


def plot_molecule_structure(tex_lw=240.71031, ppi=72):
    # plot molecule structure of 2d canal
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.7))
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    # for outofplane twist (2D CANAL)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax_plot_2d_canel_nonflip(ax)
    ax_plot_2d_canel_flip(ax, xshift=0,zshift=-10)
    ax.set_aspect('equal')
    ax.view_init(elev=35., azim=110)
    ax.set_axis_off()
    ax.legend(loc="right",bbox_to_anchor=(0.95,0.5), ncol=1, columnspacing=0.5, handlelength=0.5, handletextpad=0.5, frameon=False, fontsize=9)
    plt.tight_layout(pad=-5)
    plt.savefig("./figures/molecule_structure.pdf", format="pdf")
    plt.show()
    plt.close()
