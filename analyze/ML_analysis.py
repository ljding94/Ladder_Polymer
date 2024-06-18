import numpy as np
from plot_analyze import *
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import random


def read_Delta_Sq_data(folder, parameters):
    all_features = []
    all_Delta_Sq = []
    for segment_type, L, logKt, logKb, Rf in parameters:
        filename = f"{folder}/obs_{segment_type}_L{L:.0f}_logKt{logKt:.2f}_logKb{logKb:.2f}_Rf{Rf:.3f}_SqB.csv"
        print("reading: ", filename)
        if os.path.exists(filename):
            Sqdata = np.genfromtxt(filename, delimiter=',', skip_header=1)
            features = Sqdata[0, 2: 7]
            Sq, q = Sqdata[0, 7:], Sqdata[2, 7:]
            Sq_rod_discrete = Sqdata[3, 7:]
            Delta_Sq = Sq/Sq_rod_discrete
            all_features.append(features)
            all_Delta_Sq.append(Delta_Sq)
        else:
            print(f"Warning: File {filename} not found.")
    return np.array(all_features), all_Delta_Sq, q


def plot_svd(folder, parameters, all_feature_names):

    Ls, logKts, logKbs, Rfs = [], [], [], []
    for segment_type, L, logKt, logKb, Rf in parameters:
        Ls.append(L)
        logKts.append(logKt)
        logKbs.append(logKb)
        Rfs.append(Rf)

    all_features, all_Delta_Sq, q = read_Delta_Sq_data(folder, parameters)

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
    #plt.show()
    plt.close()

    all_Delta_SqV = np.inner(all_Delta_Sq, np.transpose(svd.Vh))
    plt.figure()
    fig = plt.figure(figsize=(20, 4))
    axs = [fig.add_subplot(1, len(all_feature_names), i+1, projection='3d') for i in range(len(all_feature_names))]
    for i in range(len(all_feature_names)):
        scatter = axs[i].scatter(all_Delta_SqV[:, 0], all_Delta_SqV[:, 1], all_Delta_SqV[:, 2], c=all_features[:, i], cmap="jet_r")
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
        fig.colorbar(scatter, ax=axs[i])
        axs[i].view_init(elev=18., azim=0)

    plt.tight_layout()
    plt.show()
    #plt.savefig(f"{folder}/{segment_type}_svd_projection_scatter_plot.png", dpi=300)
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
    print("ac_mu", ac_mu)
    return ac_mu, all_z


def plot_pddf_acf(folder, parameters, all_feature_names):
    segment_type = parameters[0][0]
    # plot pair distance distribution of Delta Sq
    all_features, all_Delta_Sq, q = read_Delta_Sq_data(folder, parameters)

    max_z = 15
    p_z, z = calc_Sq_pair_distance_distribution(all_Delta_Sq, max_z, 100)

    plt.figure(figsize=(8, 6))
    plt.plot(z, p_z, label="p_z")

    acf_data = []
    for i in range(len(all_feature_names)):
        # pass
        acf_mu, z = calc_Sq_autocorrelation(all_features[:, i], all_Delta_Sq, max_z, 100)
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
    np.savetxt(f"{folder}/data_{segment_type}pddf_acf.txt", data, delimiter=',', header=','.join(column_names), comments='')


def GaussianProcess_optimization(folder, parameters, all_feature_names):
    Ls, logIts, logIbs, Rfs = [], [], [], []
    for segment_type, L, logIt, logIb, Rf in parameters:
        Ls.append(L)
        logIts.append(logIt)
        logIbs.append(logIb)
        Rfs.append(Rf)

    all_features, all_Delta_Sq, q = read_Delta_Sq_data(folder, parameters)

    grid_size = 40
    theta_per_feature = {#"Rf": (np.logspace(0, 3, grid_size), np.logspace(-6, -3, grid_size)),
                         #"Rg": (np.logspace(-1.5, 1.2, grid_size), np.logspace(-5.5, -1, grid_size)), # under Delta Sq space
                         "Rg": (np.logspace(-2, 2, grid_size), np.logspace(-4, -1, grid_size)),
                         #"L": (np.logspace(0, 1, grid_size), np.logspace(-12, -9, grid_size))
                         }

    plt.figure()

    fig, axs = plt.subplots(1, len(all_feature_names), figsize=(6*len(all_feature_names), 6))
    for feature_name, (theta0, theta1) in theta_per_feature.items():
        feature_index = all_feature_names.index(feature_name)
        if (feature_name == "L"):
            all_features[:, feature_index] = np.power(all_features[:, feature_index], 1/3)
            # feature_name = r"$L^{1/3}$"
        F_learn = all_Delta_Sq
        if( feature_name == "Rg"):
            F_learn = np.delete(all_features, feature_index, axis=1)
            print("F_learn(Rg)", F_learn)

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
        if (feature_name == "L"):
            feature_name = r"$L^{1/3}$"
        ax.set_title(f'Log Marginal Likelihood for {feature_name}')
        ax.legend()

    plt.tight_layout()
    #plt.savefig(f"{folder}/{parameters[0][0]}LML_subplots.png", dpi=300)
    plt.show()
    plt.close()

    # save these data for future easy plotting
    # TODO: let's figure out how to save this later


def GaussianProcess_prediction(folder, parameters, all_feature_names):
    all_features, all_Delta_Sq, q = read_Delta_Sq_data(folder, parameters)
    segment_type = parameters[0][0]

    grid_size = 40
    theta_per_feature = {"Rf": (np.logspace(0, 3, grid_size), np.logspace(-6, -3, grid_size)),
                         #"Rg": (np.logspace(-1.5, 1.2, grid_size), np.logspace(-5.5, -1, grid_size)), # under Delta Sq space
                         "Rg": (np.logspace(-2, 2, grid_size), np.logspace(-4, -1, grid_size)), # learn Rg using L and Rf
                         "L": (np.logspace(0, 1, grid_size), np.logspace(-12, -9, grid_size))
                         }

    plt.figure()

    fig, axs = plt.subplots(1, len(all_feature_names), figsize=(6*len(all_feature_names), 6))
    for feature_name, (theta0, theta1) in theta_per_feature.items():
        feature_index = all_feature_names.index(feature_name)
        if (feature_name == "L"):
            all_features[:, feature_index] = np.power(all_features[:, feature_index], 1/3)
        F_learn = all_Delta_Sq

        if( feature_name == "Rg"):
            F_learn = np.delete(all_features, feature_index, axis=1)
            print("F_learn(Rg)", F_learn)


        concate_data = np.concatenate((all_features[:, feature_index].reshape(-1, 1), F_learn), axis=1)
        print("all_features[:, feature_index]", all_features[:, feature_index])
        print("concate_data", concate_data)
        random.shuffle(concate_data)
        train_data = concate_data[:int(len(concate_data)*0.7)]
        print("train_data", train_data)
        test_data = concate_data[int(len(concate_data)*0.3):]

        # find optimized theta0, theta1, using the above contour as guidanve
        kernel = RBF(theta0[grid_size//2], (theta0[0], theta0[-1])) + WhiteKernel(theta1[grid_size//2], (theta1[0], theta1[-1]))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10).fit(train_data[:, 1:], train_data[:, 0])

        print("GPML kernel: %s" % gp.kernel_)
        gp_theta = np.exp(gp.kernel_.theta)
        # kernel_params_array = np.array(list(kernel_params.values()))
        print("Kernel parameters:", gp_theta)
        print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

        Y_predict, Y_predict_err = gp.predict(test_data[:, 1:], return_std=True)
        print("np.shape(test_data[:, 0])", np.shape(test_data[:, 0]))
        print("np.shape(Y_predict)", np.shape(Y_predict))

        # convert L back to L from L^1/3
        if (feature_name == "L"):
            test_data[:, 0] = np.power(test_data[:, 0], 3)
            Y_predict_err = 3* Y_predict_err * np.power(Y_predict, 2)
            Y_predict = np.power(Y_predict, 3)

        axs[feature_index].errorbar(test_data[:, 0].T, Y_predict, yerr=Y_predict_err, marker="o", markerfacecolor="none", markersize=3, linestyle="none")
        axs[feature_index].plot(test_data[:, 0], test_data[:, 0], "--")
        min_val = min(np.min(test_data[:, 0]), np.min(Y_predict - Y_predict_err))
        max_val = max(np.max(test_data[:, 0]), np.max(Y_predict + Y_predict_err))
        axs[feature_index].set_xlim(min_val, max_val)
        axs[feature_index].set_ylim(min_val, max_val)
        axs[feature_index].set_xlabel(f"{feature_name}")
        axs[feature_index].set_ylabel(f"{feature_name} Prediction")

        # save data to file
        data = np.column_stack((test_data[:, 0], Y_predict, Y_predict_err))
        column_names = [feature_name, "ML predicted", "ML predicted uncertainty"]
        np.savetxt(f"{folder}/data_{segment_type}_{feature_name}_prediction.txt", data, delimiter=',', header=','.join(column_names), comments='')

    plt.savefig(f"{folder}/{segment_type}_prediction.png", dpi=300)
    plt.close()

