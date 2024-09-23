import matplotlib.pyplot as plt
import numpy as np
import scipy


def plot_polymer_config(filename):
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    print(data)
    x, y, z = data[:, 2] - np.mean(data[:, 2]), data[:, 3] - \
        np.mean(data[:, 3]), data[:, 4] - np.mean(data[:, 4])
    ux, uy, uz = data[:, 5], data[:, 6], data[:, 7]
    vx, vy, vz = data[:, 8], data[:, 9], data[:, 10]
    print(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the polymer path
    # ax.plot(x, y, z, "o", markersize=3)
    ax.scatter(x, y, z, color='blue', alpha=0.2, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    d = 1
    for i in range(len(x)-1):
        ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]],
                [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i] +
                    uy[i], y[i]-0.5*d*vy[i]+uy[i], y[i]-0.5*d*vy[i]],
                # plot the segment frame
                [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]], linewidth=1, alpha=1)

    # Set the same limits for x, y, and z axes
    max_range = max(max(x), max(y), max(z))
    min_range = min(min(x), min(y), min(z))
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])

    ax.set_aspect('equal')  # Set equal aspect ratio for all axes
    # plt.savefig(filename[:-4]+'.png')
    plt.show()


def ax_plot_polymer_config(ax, filename, polymer_width=1, smooth_link=False):
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    x, y, z = data[:, 2] - np.mean(data[:, 2]), data[:, 3] - np.mean(data[:, 3]), data[:, 4] - np.mean(data[:, 4])
    ux, uy, uz = data[:, 5], data[:, 6], data[:, 7]
    vx, vy, vz = data[:, 8], data[:, 9], data[:, 10]

    d = polymer_width

    # plot transparent sphere at each x,y,z
    ax.scatter(x, y, z, color='blue', alpha=0.2, s=1)

    if smooth_link:
        # plot single loop representing the entire polymer
        pass
        # TODO: tobe implemented, but not prioritize as of 3Jun24
    else:
        # plot each segment by themselves
        for i in range(len(x)-1):
            ax.plot([x[i]-0.5*d*vx[i], x[i]+0.5*d*vx[i], x[i]+0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]+ux[i], x[i]-0.5*d*vx[i]], [y[i]-0.5*d*vy[i], y[i]+0.5*d*vy[i], y[i]+0.5*d*vy[i] + uy[i], y[i]-0.5*d*vy[i]+uy[i], y[i]-0.5*d*vy[i]], [z[i]-0.5*d*vz[i], z[i]+0.5*d*vz[i], z[i]+0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]+uz[i], z[i]-0.5*d*vz[i]], linewidth=1, alpha=1)

    # Set the same limits for x, y, and z axes
    max_range = max(max(x), max(y), max(z))
    min_range = min(min(x), min(y), min(z))
    ax.set_xlim([min_range, max_range])
    ax.set_ylim([min_range, max_range])
    ax.set_zlim([min_range, max_range])

    ax.set_aspect('equal')  # Set equal aspect ratio for all axes
    # plt.savefig(filename[:-4]+'.png')
    # plt.show(


def plot_polymer_config_with_multiple_parameters(folder, segment_type, L, Kts, Kbs, smooth_link=False):
    all_filenames = []
    for Kt in Kts:
        all_filenames.append([])
        for Kb in Kbs:
            filename = f"{folder}/config_{segment_type}_L{L}_Kt{Kt:.2f}_Kb{Kb:.2f}.csv"
            print("reading: ", filename)
            all_filenames[-1].append(filename)
    # Create subplots
    fig = plt.figure(figsize=(2*len(Kbs), 2*len(Kts)))
    axs = fig.subplots(len(Kts), len(Kbs), subplot_kw={'projection': '3d'}, sharex=True, sharey=True)
    # Loop through each subplot
    for i in range(len(Kts)):
        for j in range(len(Kbs)):
            axs[i, j].set_title(f'L={L}, Kt={Kts[i]:.1f}, Kb={Kbs[j]}')
            ax_plot_polymer_config(axs[i, j], all_filenames[i][j], smooth_link)
    plt.tight_layout()
    plt.savefig(f'{folder}/{segment_type}_config_for_multiple_parameters.png', dpi=300)


def plot_observable_distribution(filename):
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    E, Tot_phi, Tot_theta = data[:, 1], data[:, 2], data[:, 3]

    # Plot histogram of E
    plt.figure()
    plt.hist(E, bins=50, histtype='step')
    plt.xlabel('E')
    plt.ylabel('Frequency')
    plt.title('Energy Distribution')
    plt.savefig(filename[:-4]+'energy_histogram.png')

    # Plot histogram of Tot_phi
    plt.figure()
    plt.hist(Tot_phi, bins=50, histtype='step')
    plt.xlabel('Tot_phi')
    plt.ylabel('Frequency')
    plt.title('Tot_phi Distribution')
    plt.savefig(filename[:-4]+'tot_phi_histogram.png')

    # Plot histogram of Tot_theta
    plt.figure()
    plt.hist(Tot_theta, bins=50, histtype='step')
    plt.xlabel('Tot_theta')
    plt.ylabel('Frequency')
    plt.title('Tot_theta Distribution')
    plt.savefig(filename[:-4]+'tot_theta_histogram.png')


def plot_average_pair_distribution_function(filename):
    data = np.loadtxt(filename, skiprows=5, delimiter=",")
    gr = data[:, 4:]
    average_gr = np.mean(gr, axis=0)

    # Plot pair distribution function (gr)
    r = np.arange(len(average_gr))
    plt.figure()
    for i in range(0, len(gr), int(len(gr)/20)):
        plt.plot(r*0.1, gr[i], color="orange", linestyle="None",
                 marker="o", markersize=3, alpha=0.1, linewidth=0.5)

    plt.plot(r*0.1, average_gr, color="blue", linestyle="-",
             linewidth=1.5, label="average gr")

    plt.xlabel('Distance')
    plt.ylabel('g(r)')
    plt.legend()
    plt.title('Pair Distribution Function')
    plt.savefig(filename[:-4]+'pair_distribution_function.png')


def plot_stats_distribution_function(filename, functions):
    # data = np.loadtxt(filename, skiprows=5, delimiter=",")
    all_avg_func = []
    all_std_dev_func = []
    all_t = []
    for func in functions:
        data = np.genfromtxt(
            filename[:-4]+func+".csv", delimiter=',', skip_header=5)
        if (func == "gr"):
            Sq = scipy.fft.dct(data[0, 1:])
            q = np.array(range(len(Sq)))*np.pi/len(Sq)
            all_avg_func.append(Sq)
            all_std_dev_func.append(np.zeros(len(Sq)))
            all_t.append(q)
            func = "gr implied Sq"
        else:
            all_avg_func.append(data[0, 1:])
            all_std_dev_func.append(data[1, 1:])
            all_t.append(data[2, 1:])

    plt.figure()
    for i in range(len(functions)):
        plt.plot(all_t[i], all_avg_func[i],
                 linestyle="-", linewidth=1, label=functions[i])
    # plt.fill_between(r*0.1, avg_gr - err_gr, avg_gr + err_gr,color="gray", alpha=0.5, label=r"$1\sigma$ error")

    plt.xlabel('q')
    plt.ylabel('S(q)')
    plt.legend()
    plt.title('Structure Factor')
    plt.savefig(filename[:-4]+'distribution_function_stats.png')


def calc_Sq_continous_infinite_thin_rod(qL):
    # ref Pedersen 1996 equ(6) https://doi.org/10.1021/ma9607630
    Sq = []
    for qkL in qL:
        si, ci = scipy.special.sici(qkL)
        Sqk = 2*si/(qkL) - 4*(np.sin(qkL/2.0)/(qkL))**2
        Sq.append(Sqk)
    #print(Sq)
    return np.array(Sq)


def calc_Sq_discrete_infinite_thin_rod(q, L):
    # numereical calculation
    Sq = [1.0/L for i in range(len(q))]
    for k in range(len(q)):
        Sqk = 0
        qk = q[k]
        for i in range(L-1):
            for j in range(i+1,L):
                Sqk += 2.0*np.sin(qk*(i-j))/(qk*(i-j))/(L*L)
        Sq[k]+=Sqk
    return np.array(Sq)


def plot_stats_structure_factor_for_multiple_parameters(folder, segment_type, L, Kts, Kbs, use_delta_Sq=False):
    all_avg_Sq = []
    all_qL = []
    for Kt in Kts:
        all_avg_Sq.append([])
        all_qL.append([])
        for Kb in Kbs:
            filename = f"{folder}/obs_{segment_type}_L{L}_Kt{Kt:.2f}_Kb{Kb:.2f}_SqL.csv"
            print("reading: ", filename)
            data = np.genfromtxt(filename, delimiter=',', skip_header=5)
            all_avg_Sq[-1].append(data[0, 1:])
            all_qL[-1].append(data[2, 1:])
    Sq_rod = calc_Sq_continous_infinite_thin_rod(all_qL[0][0])
    print("Sq_rod", Sq_rod)
    Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(all_qL[0][0]/L, L)
    print("Sq_rod_discrete", Sq_rod_discrete)

    # Create subplots
    fig, axs = plt.subplots(len(Kts)+1, len(Kbs)+1,
                            figsize=(2*len(Kbs), 2*len(Kts)+2), sharex=True, sharey=True)

    # Loop through each subplot
    for i in range(len(Kts)):
        for j in range(len(Kbs)):
            # Plot the pair distribution function
            if use_delta_Sq:
                axs[i, j].loglog(all_qL[i][j], all_avg_Sq[i][j]/Sq_rod_discrete, linestyle="-", linewidth=1, label=r"$\Delta Sq(QL)$")
                axs[len(Kts), j].loglog(all_qL[i][j], all_avg_Sq[i][j]/Sq_rod_discrete, linestyle="-", linewidth=1, label=f"Kt={Kts[i]:.1f}")
                axs[i, len(Kbs)].loglog(all_qL[i][j], all_avg_Sq[i][j]/Sq_rod_discrete, linestyle="-", linewidth=1, label=f"Kb={Kbs[j]:.1f}")
            else:
                '''
                axs[i, j].loglog(all_q[i][j]*L, all_avg_Sq[i][j], linestyle="-", linewidth=1, label="Sq")
                axs[i, j].loglog(all_q[i][j]*L, Sq_rod_discrete, linestyle="--", linewidth=1, color="black", label="Sq_rod_discrete")

                axs[len(Kts), j].loglog(all_q[i][j]*L, all_avg_Sq[i][j], linestyle="-", linewidth=1, label=f"Kt={Kts[i]:.1f}")
                if (i == 0):
                    axs[len(Kts), j].loglog(all_q[i][j]*L, Sq_rod_discrete, linestyle="--", linewidth=1, color="black", label="Sq_rod")

                axs[i, len(Kbs)].loglog(all_q[i][j]*L, all_avg_Sq[i][j], linestyle="-", linewidth=1, label=f"Kb={Kbs[j]:.1f}")
                if (j == 0):
                    axs[i, len(Kbs)].loglog(all_q[i][j]*L, Sq_rod_discrete, linestyle="--", linewidth=1, color="black", label="Sq_rod")
                '''
                axs[i, j].loglog(all_qL[i][j], all_avg_Sq[i][j]/Sq_rod_discrete*Sq_rod, linestyle="-", linewidth=1, label="Sq")
                axs[i, j].loglog(all_qL[i][j], Sq_rod, linestyle="--", linewidth=1, color="black", label="Sq_rod")

                axs[len(Kts), j].loglog(all_qL[i][j], all_avg_Sq[i][j]/Sq_rod_discrete*Sq_rod, linestyle="-", linewidth=1, label=f"Kt={Kts[i]:.1f}")
                if (i == 0):
                    axs[len(Kts), j].loglog(all_qL[i][j], Sq_rod, linestyle="--", linewidth=1, color="black", label="Sq_rod")

                axs[i, len(Kbs)].loglog(all_qL[i][j], all_avg_Sq[i][j]/Sq_rod_discrete*Sq_rod, linestyle="-", linewidth=1, label=f"Kb={Kbs[j]:.1f}")
                if (j == 0):
                    axs[i, len(Kbs)].loglog(all_qL[i][j], Sq_rod, linestyle="--", linewidth=1, color="black", label="Sq_rod")


            # axs[i, j].set_ylim([axs[i, j].get_ylim()[0], 1.1])
            axs[i, j].grid(True, which='both', linewidth=0.2, alpha=0.5)
            axs[i, j].legend()
            axs[i, j].set_title(f'L={L}, Kt={Kts[i]:.1f}, Kb={Kbs[j]:.1f}', fontsize='small')

            axs[len(Kts), j].legend(fontsize='x-small')
            axs[len(Kts), j].set_title(f'L={L},Kb={Kbs[j]:.1f}')
            axs[i, len(Kbs)].legend(fontsize='x-small')
            axs[i, len(Kbs)].set_title(f'L={L}, Kt={Kts[i]:.1f}', fontsize='x-small')

    axs[len(Kts), len(Kbs)//2].set_xlabel('QL')
    if use_delta_Sq:
        axs[len(Kts)//2, 0].set_ylabel(r'$\DeltaS(QL) = S(QL)$/S_{rod}(QL)$')
    else:
        axs[len(Kts)//2, 0].set_ylabel('S(QL)')

    # Adjust spacing between subplots
    plt.tight_layout()
    # Save the figure
    if use_delta_Sq:
        plt.savefig(f'{folder}/{segment_type}_stats_delta_sq_for_multiple_parameters.png', dpi=300)
    else:
        plt.savefig(f'{folder}/{segment_type}_stats_sq_for_multiple_parameters.png', dpi=300)


def plot_stats_obervable_for_multiple_parameters(Kts, Kbs):
    L = 20
    all_avg_gr = []
    all_err_gr = []
    for Kt in Kts:
        all_avg_gr.append([])
        all_err_gr.append([])
        for Kb in Kbs:
            filename = f"../data/scratch_local/obs_L{L}_Kt{Kt}_Kb{Kb}.csv"
            print("reading: ", filename)
            data = np.genfromtxt(filename, delimiter=',', skip_header=5)
            avg_gr = data[0, 4:]
            std_dev_gr = data[1, 4:]
            err_gr = std_dev_gr/np.sqrt(1e4)
            all_avg_gr[-1].append(avg_gr)
            all_err_gr[-1].append(err_gr)

    # Create subplots
    fig, axs = plt.subplots(len(Kts), len(
        Kbs), figsize=(10, 10), sharex=True, sharey=True)

    # Loop through each subplot
    for i in range(len(Kts)):
        for j in range(len(Kbs)):
            # Get the average gr and error for the current Kt and Kb
            avg_gr = all_avg_gr[i][j]
            err_gr = all_err_gr[i][j]
            # Plot the pair distribution function
            r = np.arange(len(avg_gr)) * 0.1
            axs[i, j].plot(r, avg_gr, color="blue", linestyle="-",
                           linewidth=1.5, label="average gr")
            axs[i, j].fill_between(r, avg_gr - err_gr, avg_gr + err_gr,
                                   color="gray", alpha=0.5, label=r"$1\sigma$ error")
            # axs[i, j].set_xlabel('Distance')
            # axs[i, j].set_ylabel('g(r)')
            # axs[i, j].set_aspect("equal")
            axs[i, j].grid(True, which='both', linewidth=0.2, alpha=0.5)

            axs[i, j].xaxis.set_minor_locator(plt.MultipleLocator(1))
            axs[i, j].xaxis.set_major_locator(plt.MultipleLocator(5))
            axs[i, j].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
            axs[i, j].yaxis.set_major_locator(plt.MultipleLocator(1))

            axs[i, j].legend()
            axs[i, j].set_title(f'L=20, Kt={Kts[i]}, Kb={Kbs[j]}')

    axs[4, 2].set_xlabel('r')
    axs[2, 0].set_ylabel('g(r)')

    # Adjust spacing between subplots
    plt.tight_layout()
    # Save the figure
    plt.savefig(
        '../data/scratch_local/pair_distribution_function_stats_multi_parameters.png')


