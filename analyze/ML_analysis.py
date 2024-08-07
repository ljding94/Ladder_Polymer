import numpy as np
from plot_analyze import *
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import random
import os


def calc_Sq_discrete_infinite_thin_rod(q, L):
    # numereical calculation
    Sq = [1.0/L for i in range(len(q))]
    for k in range(len(q)):
        Sqk = 0
        qk = q[k]
        for i in range(L-1):
            for j in range(i+1, L):
                Sqk += 2.0*np.sin(qk*(i-j))/(qk*(i-j))/(L*L)
        Sq[k] += Sqk
    return np.array(Sq)


def read_Delta_Sq_data(folder, parameters, L0=300):
    # normalized againt L0
    all_features = []
    all_Delta_Sq = []
    all_Delta_Sq_err = []
    q = []
    Sq_rod_discrete = []
    all_filename = []
    if len(parameters[0]) == 2:
        for segment_type, run_num in parameters:
            all_filename.append(f"{folder}/obs_{segment_type}_random_run{run_num}_SqB.csv")
    else:
        for segment_type, L, logKt, logKb, Rf in parameters:
            all_filename.append(f"{folder}/obs_{segment_type}_L{L:.0f}_logKt{logKt:.2f}_logKb{logKb:.2f}_Rf{Rf:.3f}_SqB.csv")

    for filename in all_filename:
        print("reading: ", filename)
        if os.path.exists(filename):
            Sqdata = np.genfromtxt(filename, delimiter=',', skip_header=1, max_rows=1)
            if len(Sqdata) == 0:
                print(f"Warning: File {filename} is empty. Skiped")
                continue
            features = Sqdata[2: 10]
            features = np.insert(features, 8, features[7]/(features[6]*features[6]))  # add L^2/(L)^2 = PDI
            print(["lnLmu", "lnLsig", "Kt", "Kb", "Rf", "Rg2", "L", r"$L^2$", "PDI"])
            print("features", features)
            # Sq, Sq_err, q = Sqdata[0, 8:], Sqdata[1, 8:], Sqdata[2, 8:]
            qdata = np.genfromtxt(filename, delimiter=',', skip_header=3, max_rows=1)
            Sq, q = Sqdata[10:]*1e-6, qdata[10:]
            # Sq_rod_discrete = Sqdata[3, 7:]
            if len(Sq_rod_discrete) == 0:
                Sq_rod_discrete = calc_Sq_discrete_infinite_thin_rod(q, L0)
            # normalize Sq by Sq_rod_discrete with L0
            Delta_Sq = Sq/Sq_rod_discrete
            # Delta_Sq = Sq
            # Delta_Sq_err = Sq_err/Sq_rod_discrete
            all_features.append(features)
            all_Delta_Sq.append(Delta_Sq)
            # all_Delta_Sq_err.append(Delta_Sq_err)
        else:
            print(f"Warning: File {filename} not found. Skiped")
    all_feature_names = ["lnLmu", "lnLsig", "Kt", "Kb", "Rf", "Rg2", "L", "L2", "PDI"]
    return segment_type, np.array(all_features), all_feature_names, all_Delta_Sq, all_Delta_Sq_err, q


def plot_svd(folder, parameters):
    segment_type, all_features, all_feature_names, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    print("all_features shape:", np.array(all_features).shape)

    print("np.array(all_Delta_Sq).shape", np.array(all_Delta_Sq).shape)
    svd = np.linalg.svd(all_Delta_Sq)
    print(svd.S)
    print("np.array(svd.U).shape", np.array(svd.U).shape)
    print("np.array(svd.S).shape", np.array(svd.S).shape)
    print("np.array(svd.Vh).shape", np.array(svd.Vh).shape)
    # print(np.linalg.svd(all_Delta_Sq))

    plt.figure(figsize=(6, 3))
    # Subplot for svd.S
    plt.subplot(1, 2, 1)
    plt.plot(range(len(svd.S)), svd.S, "o--", markerfacecolor='none', label="svd.S")
    plt.title("Singular Values (svd.S)")

    # Subplot for svd.U
    plt.subplot(1, 2, 2)
    # plt.plot(range(len(svd.Vh)), svd.Vh[0], label="svd.Vh[0]")
    plt.semilogx(q, svd.Vh[0], label=r"$V^T[0]$")
    plt.semilogx(q, svd.Vh[1], label=r"$V^T[1]$")
    plt.semilogx(q, svd.Vh[2], label=r"$V^T[2]$")
    plt.xlabel("qB")
    plt.title("Left Singular Vectors (svd.Vh)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{folder}/{segment_type}_svd.png", dpi=300)
    # plt.show()
    plt.close()

    all_Delta_SqV = np.inner(all_Delta_Sq, np.transpose(svd.Vh))
    plt.figure()
    fig = plt.figure(figsize=(2*len(all_feature_names), 8))
    axs = [fig.add_subplot(2, len(all_feature_names)//2 +1, i+1, projection='3d') for i in range(len(all_feature_names))]
    for i in range(len(all_feature_names)):
        scatter = axs[i].scatter(all_Delta_SqV[:, 0], all_Delta_SqV[:, 1], all_Delta_SqV[:, 2], c=all_features[:, i], cmap="jet_r", s=2)
        axs[i].set_xlabel("V[0]")
        axs[i].set_ylabel("V[1]")
        axs[i].set_zlabel("V[2]")
        axs[i].set_title(all_feature_names[i])
        axs[i].set_box_aspect([1, 1, 1])  # Set the aspect ratio of the plot
        # Set the same range for each axis
        max_range = np.array([all_Delta_SqV[:, 0].max()-all_Delta_SqV[:, 0].min(), all_Delta_SqV[:, 1].max()-all_Delta_SqV[:, 1].min(), all_Delta_SqV[:, 2].max()-all_Delta_SqV[:, 2].min()]).max() / 2.0
        mid_x = (all_Delta_SqV[:, 0].max()+all_Delta_SqV[:, 0].min()) * 0.5
        mid_y = (all_Delta_SqV[:, 1].max()+all_Delta_SqV[:, 1].min()) * 0.5
        mid_z = (all_Delta_SqV[:, 2].max()+all_Delta_SqV[:, 2].min()) * 0.5
        axs[i].set_xlim(mid_x - max_range, mid_x + max_range)
        axs[i].set_ylim(mid_y - max_range, mid_y + max_range)
        axs[i].set_zlim(mid_z - max_range, mid_z + max_range)
        fig.colorbar(scatter, ax=axs[i], fraction=0.02)
        axs[i].view_init(elev=60., azim=18)

    plt.tight_layout()
    plt.savefig(f"{folder}/{segment_type}_svd_projection_scatter_plot.png", dpi=300)
    plt.show()
    plt.close()

    # save these analyzed data for further easy plotting
    # svd data
    data = np.column_stack((q, svd.S, svd.Vh[0], svd.Vh[1], svd.Vh[2]))
    column_names = ['q', 'svd.S', 'svd.Vh[0]', 'svd.Vh[1]', 'svd.Vh[2]']
    np.savetxt(f"{folder}/data_{segment_type}_svd.txt", data, delimiter=',', header=','.join(column_names), comments='')

    #  svd projection data
    # save svd projection data
    data = np.column_stack((all_features, all_Delta_SqV[:, 0], all_Delta_SqV[:, 1], all_Delta_SqV[:, 2]))
    column_names = all_feature_names + ['sqv[0]', 'sqv[1]', 'sqv[2]']
    np.savetxt(f"{folder}/data_{segment_type}_svd_projection.txt", data, delimiter=',', header=','.join(column_names), comments='')


def calc_Sq_pair_distance_distribution(all_Delta_Sq, max_z, bin_num):
    all_z = np.linspace(0, max_z, bin_num)
    all_Delta_Sq_dis = np.zeros(bin_num)
    all_Delta_Sq_dis[0] = 1/len(all_Delta_Sq)  # for self distance

    for i in range(len(all_Delta_Sq)-1):
        for j in range(i+1, len(all_Delta_Sq)):
            Delta_Sq_dis = np.sqrt(np.sum(np.square(all_Delta_Sq[i]-all_Delta_Sq[j])))
            bin_index = int(Delta_Sq_dis/(max_z/bin_num))
            if (bin_index >= bin_num):
                raise ValueError(f"bin_index >= bin_num, z={bin_index/bin_num*max_z}")
            all_Delta_Sq_dis[bin_index] += 2.0/(len(all_Delta_Sq))**2/(max_z/bin_num)  # 2.0 for i,j and j,i symmetry, normalize to 1

    return all_Delta_Sq_dis, all_z


def calc_Sq_autocorrelation(mu, all_Delta_Sq, max_z, bin_num):
    # measure the autocorrelation of mu(Delta_Sq)
    all_z = np.linspace(0, max_z, bin_num)
    avg_mu = np.mean(mu)
    avg_mu2 = np.mean(np.square(mu))
    print("np.shape(mu)", np.shape(mu))
    print("np.shape(all_Delta_Sq)", np.shape(all_Delta_Sq))

    print("avg_mu:", avg_mu)
    print("avg_mu2:", avg_mu2)
    print("avg_mu2-avg_mu**2:", avg_mu2-avg_mu**2)

    avg_mumuz = np.zeros(bin_num)
    avg_mu2z = np.zeros(bin_num)
    avg_muz = np.zeros(bin_num)

    # avg_mumuz[0] = avg_mu2  # for self distance
    # for i in range(len(all_Delta_Sq)-1):
    #    for j in range(i+1, len(all_Delta_Sq)):
    bin_count = np.zeros(bin_num)
    for i in range(len(mu)):
        for j in range(len(mu)):
            Delta_Sq_dis = np.sqrt(np.sum(np.square(all_Delta_Sq[i]-all_Delta_Sq[j])))
            bin_index = int(Delta_Sq_dis/(max_z/bin_num))
            if (bin_index >= bin_num):
                raise ValueError(f"bin_index >= bin_num, z={bin_index/bin_num*max_z}")
            avg_muz[bin_index] += mu[i]
            avg_mu2z[bin_index] += mu[i]*mu[i]
            avg_mumuz[bin_index] += mu[i]*mu[j]
            bin_count[bin_index] += 1
    for i in range(bin_num):
        avg_muz[i] /= bin_count[i]
        avg_mu2z[i] /= bin_count[i]
        avg_mumuz[i] /= bin_count[i]

    ac_mu = np.ones(bin_num)
    for i in range(0, bin_num):
        if ((avg_mu2-avg_mu**2 == 0)):
            ac_mu[i] = 1
        else:
            ac_mu[i] = (avg_mumuz[i]-avg_muz[i]**2)/(avg_mu2z[i]-avg_muz[i]**2)

    if (ac_mu[0] != 1):
        print("ac_mu[0]!=1: ")
        print("avg_mumuz[0]-avg_muz[0]**2,", avg_mumuz[0]-avg_muz[0]**2)
        print("(avg_mu2z[0]-avg_muz[0]**2)", (avg_mu2z[0]-avg_muz[0]**2))
        print(bin_count)
    ac_mu[0] = 1
    print("ac_mu", ac_mu)
    return ac_mu, all_z


def plot_pddf_acf(folder, parameters, max_z=2, n_bin=100):
    # plot pair distance distribution of Delta Sq
    segment_type, all_features, all_feature_names, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters)
    p_z, z = calc_Sq_pair_distance_distribution(all_Delta_Sq, max_z, n_bin)

    plt.figure(figsize=(8, 6))
    plt.plot(z, p_z, label="p_z")

    acf_data = []
    for i in range(len(all_feature_names)):
        # pass
        acf_mu, z = calc_Sq_autocorrelation(all_features[:, i], all_Delta_Sq, max_z, n_bin)
        plt.plot(z, acf_mu, label=f"acf_{all_feature_names[i]}")
        acf_data.append(acf_mu)

    plt.xlabel("z")
    plt.ylabel("Value")
    plt.title("Pair Distance Distribution and Autocorrelation")
    plt.legend()
    plt.savefig(f"{folder}/{segment_type}pddf_acf.png", dpi=300)
    plt.close()

    # save these data to file for futher easy plotting

    data = np.column_stack((z, p_z, *acf_data))
    column_names = ['z', 'p_z', *['acf_' + feature_name for feature_name in all_feature_names]]
    np.savetxt(f"{folder}/data_{segment_type}_pddf_acf.txt", data, delimiter=',', header=','.join(column_names), comments='')


def GaussianProcess_optimization(folder, parameters_train, all_feature_names):
    segment_type, all_features, all_feature_names, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters_train)
    grid_size = 40

    # ["lnLmu", "lnLsig", "Kt", "Kb", "Rf", "Rg2", "L", "L2", "PDI"]
    inplane_theta_per_feature = {  # "L": (np.logspace(-1, 1, grid_size), np.logspace(-11, -8, grid_size)),
        # "Rf": (np.logspace(-1, 2, grid_size), np.logspace(-4, -1, grid_size)), # old Rf
        # "Rg": (np.logspace(2, 1, grid_size), np.logspace(-6, -4, grid_size)),  # under Delta Sq space
        #"Lmu": (np.logspace(-2, 2, grid_size), np.logspace(-4, 0, grid_size)),
        #"L": (np.logspace(0, 2, grid_size), np.logspace(-3, -1, grid_size)),
        #"Rf": (np.logspace(0, 2, grid_size), np.logspace(-4, 1, grid_size)),
        "Rg2": (np.logspace(0, 3, grid_size), np.logspace(-3, -1, grid_size)),


        # "Lsig": (np.logspace(-10, -2, grid_size), np.logspace(-10, -1, grid_size)),
        # "Kt": (np.logspace(-5, 1, grid_size), np.logspace(-10, 1, grid_size)),
        # "Kb": (np.logspace(-1, 1, grid_size), np.logspace(-11, -8, grid_size)),
    }
    outofplane_theta_per_feature = {#"Lmu": (np.logspace(-2, 2, grid_size), np.logspace(-4, 0, grid_size)),
                                    "L": (np.logspace(-2, 2, grid_size), np.logspace(-4, 0, grid_size)),
                                    "Rf": (np.logspace(-1, 2, grid_size), np.logspace(-2, 1, grid_size)),
                                    "Rg2": (np.logspace(0, 2, grid_size), np.logspace(-6, -4, grid_size)),  # under Delta Sq space
                                    # "Rg": (np.logspace(-2, 2, grid_size), np.logspace(-4, -1, grid_size)),
                                    }

    if (segment_type == "inplane_twist"):
        theta_per_feature = inplane_theta_per_feature
    elif (segment_type == "outofplane_twist"):
        theta_per_feature = outofplane_theta_per_feature
    else:
        print("segment_type not recognized\n")

    # feature normalization
    all_feature_mean = np.mean(all_features, axis=0)
    all_feature_std = np.std(all_features, axis=0)
    all_features = (all_features - all_feature_mean) / all_feature_std
    all_gp_per_feature = {}
    plt.figure()
    fig, axs = plt.subplots(1, len(all_feature_names), figsize=(6*len(all_feature_names), 6))
    for feature_name, (theta0, theta1) in theta_per_feature.items():
        if feature_name not in all_feature_names:
            continue
        print("training: ", feature_name)
        feature_index = all_feature_names.index(feature_name)

        F_learn = all_Delta_Sq

        # if( feature_name == "Rg"):
        #    F_learn = np.delete(all_features, feature_index, axis=1)
        #    print("F_learn(Rg)", F_learn)

        # witout theta optimization
        kernel = RBF(1) + WhiteKernel(1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None).fit(F_learn, all_features[:, feature_index])
        print(" all_features[:, feature_index]", all_features[:, feature_index])

        print("GPML kernel: %s" % gp.kernel_)
        gp_theta = np.exp(gp.kernel_.theta)
        # kernel_params_array = np.array(list(kernel_params.values()))
        print("Kernel parameters:", gp_theta)
        print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

        # calc Log likelihood
        ax = axs[all_feature_names.index(feature_name)]
        Theta0, Theta1 = np.meshgrid(theta0, theta1)
        LML = [[gp.log_marginal_likelihood(np.log([Theta0[i, j], Theta1[i, j]])) for i in range(Theta0.shape[0])] for j in range(Theta0.shape[1])]
        # reason for np.log here is the theta is log-transformed hyperparameters (https://github.com/scikit-learn/scikit-learn/blob/5491dc695/sklearn/gaussian_process/kernels.py#L1531) line (289)
        LML = np.array(LML).T

        ax.contour(Theta0, Theta1, LML, levels=1000)
        # find optimized theta0, theta1, using the above contour as guidanve
        kernel = RBF(theta0[grid_size//2], (theta0[0], theta0[-1])) + WhiteKernel(theta1[grid_size//2], (theta1[0], theta1[-1]))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10).fit(F_learn, all_features[:, feature_index])
        all_gp_per_feature[feature_name] = gp

        print("GPML kernel: %s" % gp.kernel_)
        gp_theta = np.exp(gp.kernel_.theta)
        # kernel_params_array = np.array(list(kernel_params.values()))
        print("Kernel parameters:", gp_theta)
        print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

        ax.plot([gp_theta[0]], [gp_theta[1]], 'x', color='red', markersize=10, markeredgewidth=2, label=r"l=%.2e, $\sigma$=%.2e" % (gp_theta[0], gp_theta[1]))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'theta0: l')
        ax.set_ylabel(r'theta1: $\sigma$')
        feature_name_legend = feature_name
        ax.set_title(f'Log Marginal Likelihood for {feature_name_legend}')
        ax.legend()

        data = np.column_stack(([gp_theta[0]] * len(theta0), [gp_theta[1]] * len(theta1), theta0, theta1, np.array(LML).T))
        column_names = ['gp_theta0', 'gp_theta1', 'theta0', 'theta1', 'LML']
        np.savetxt(f"{folder}/data_{segment_type}_{feature_name}_LML.txt", data, delimiter=',', header=','.join(column_names), comments='')

    plt.tight_layout()
    plt.savefig(f"{folder}/{segment_type}LML_subplots.png", dpi=300)
    # plt.show()
    plt.close()

    # return trained GPR
    return all_feature_mean, all_feature_std, all_gp_per_feature


def GaussianProcess_prediction(folder, parameters_test, all_feature_names, all_feature_mean, all_feature_std, all_gp_per_feature):
    segment_type, all_features, all_feature_names, all_Delta_Sq, all_Delta_Sq_err, q = read_Delta_Sq_data(folder, parameters_test)

    # normalize test data features using the save scaling as the training data
    # all_features = (all_features - all_feature_mean) / all_feature_std

    '''
    grid_size = 40
    theta_per_feature = {"Rf": (np.logspace(0, 3, grid_size), np.logspace(-6, -3, grid_size)),
                         "Rg": (np.logspace(-1.5, 1.2, grid_size), np.logspace(-5.5, -1, grid_size)), # under Delta Sq space
                         #"Rg": (np.logspace(-2, 2, grid_size), np.logspace(-4, -1, grid_size)), # learn Rg using L and Rf
                         "L": (np.logspace(0, 1, grid_size), np.logspace(-12, -9, grid_size))
                         }
    '''
    plt.figure()

    fig, axs = plt.subplots(1, len(all_feature_names), figsize=(6*len(all_feature_names), 6))
    for feature_name, gp in all_gp_per_feature.items():
        feature_index = all_feature_names.index(feature_name)
        Y = all_features[:, feature_index]
        '''
        if (feature_name == "L"):
            all_features[:, feature_index] = np.power(all_features[:, feature_index], 1/3)
        '''
        # F_learn = all_Delta_Sq

        # if( feature_name == "Rg"):
        #    F_learn = np.delete(all_features, feature_index, axis=1)
        #    print("F_learn(Rg)", F_learn)

        '''
        concate_data = np.concatenate((all_features[:, feature_index].reshape(-1, 1), F_learn), axis=1)
        print("all_features[:, feature_index]", all_features[:, feature_index])
        print("concate_data", concate_data)
        random.shuffle(concate_data)
        train_data = concate_data[:int(len(concate_data)*0.7)]
        print("train_data", train_data)
        test_data = concate_data[int(len(concate_data)*0.3):]
        '''

        # find optimized theta0, theta1, using the above contour as guidanve
        # kernel = RBF(theta0[grid_size//2], (theta0[0], theta0[-1])) + WhiteKernel(theta1[grid_size//2], (theta1[0], theta1[-1]))
        # kernel = RBF(all_gp_theta[feature_index][0], length_scale_bounds="fixed") + WhiteKernel(all_gp_theta[feature_index][1], noise_level_bounds="fixed")
        # use optimized theta0, theta1 from training data

        # gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10).fit(train_data[:, 1:], train_data[:, 0])
        # gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)
        # gp = all_gp_per_feature[feature_name]

        print("GPML kernel: %s" % gp.kernel_)
        gp_theta = np.exp(gp.kernel_.theta)  # gp.kernel_.theta return log transformed theta
        # kernel_params_array = np.array(list(kernel_params.values()))
        print("Kernel parameters:", gp_theta)
        print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

        Y_predict, Y_predict_err = gp.predict(all_Delta_Sq, return_std=True)
        # print("np.shape(test_data[:, 0])", np.shape(test_data[:, 0]))
        print("np.shape(all_Delta_Sq)", np.shape(all_Delta_Sq))
        print("np.shape(Y_predict)", np.shape(Y_predict))

        Y_predict = Y_predict * all_feature_std[feature_index] + all_feature_mean[feature_index]
        Y_predict_err = Y_predict_err * all_feature_std[feature_index]
        # convert L back to L from L^1/3
        '''
        if (feature_name == "L"):
            test_data[:, 0] = np.power(test_data[:, 0], 3)
            Y_predict_err = 3* Y_predict_err * np.power(Y_predict, 2)
            Y_predict = np.power(Y_predict, 3)
        '''

        axs[feature_index].errorbar(Y, Y_predict, yerr=Y_predict_err, marker="o", markerfacecolor="none", markersize=3, linestyle="none")
        axs[feature_index].plot(Y, Y, "--")
        min_val = min(np.min(Y), np.min(Y_predict - Y_predict_err))
        max_val = max(np.max(Y), np.max(Y_predict + Y_predict_err))
        axs[feature_index].set_xlim(min_val, max_val)
        axs[feature_index].set_ylim(min_val, max_val)
        axs[feature_index].set_xlabel(f"{feature_name}")
        axs[feature_index].set_ylabel(f"{feature_name} Prediction")

        # save data to file
        data = np.column_stack((Y, Y_predict, Y_predict_err))
        column_names = [feature_name, "ML predicted", "ML predicted uncertainty"]
        np.savetxt(f"{folder}/data_{segment_type}_{feature_name}_prediction.txt", data, delimiter=',', header=','.join(column_names), comments='')

    plt.savefig(f"{folder}/{segment_type}_prediction.png", dpi=300)
    plt.close()
