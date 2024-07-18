#!/opt/homebrew/bin/python3
from plot_analyze import *
from process_analyze import *
import numpy as np
from ML_analysis import *
import sys
import random
import time


def main():
    if len(sys.argv) > 1:
        date = sys.argv[1]
        segment_type = sys.argv[2]
        rand_max = int(sys.argv[3])  # max number of random samples to read
        folder = "/home/dingl1/ladder_polymer/data_Cades/"+date
        print(f"analyzing folder: {folder} segment type {segment_type}")
        Ls = np.arange(50, 209.1, 1)
        logKts = [1.50]
        logKbs = [1.50]
        Rfs = np.arange(0.41, 0.6001, 0.005)
        if (rand_max > 0):
            parameters = [[segment_type, rand_num] for rand_num in range(rand_max)]
        else:
            parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]
        all_feature_names = ["Lmu", "Lsig", "Kt", "Kb", "Rf", "Rg"]  # "Kt", "Kb",
        print("analyzing parameters", parameters)
        plot_svd(folder, parameters, all_feature_names)
        plot_pddf_acf(folder, parameters, all_feature_names)

        random.shuffle(parameters)
        parameters_train = parameters[:int(0.7*len(parameters))]
        parameters_test = parameters[int(0.7*len(parameters)):]

        all_feature_mean, all_feature_std, all_gp_per_feature = GaussianProcess_optimization(folder, parameters_train, all_feature_names)
        GaussianProcess_prediction(folder, parameters_test, all_feature_names, all_feature_mean, all_feature_std, all_gp_per_feature)

    else:
        print("hello!")
        # folder = "../data/scratch_local/20240610"
        folder = "../data/20240713_partial"
        segment_type = "flat"
        segment_type = "none"
        #segment_type = "inplane_twist"
        segment_type = "outofplane_twist"
        rand_max = 4000

        Ls = np.arange(50, 99.1, 1)
        logKts = [1.50]  # np.arange(0.00, 3.001, 0.10)
        logKbs = [1.50]
        Rfs = np.arange(0.41, 0.6001, 0.010)

        if (rand_max > 0):
            parameters = [[segment_type, rand_num] for rand_num in range(rand_max)]
        else:
            parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]
        # parameteres = [[segment_type, L, Kt, Kb, Rf] for Kt in Kts for Kb in Kbs for Rf in Rfs]
        print("parameters", parameters)
        print("total number of parameters", len(parameters))

        all_feature_names = ["Lmu", "Lsig", "Lsig/Lmu","Kt", "Kb", "Rf", "Rg"] # already embedded in read sq funciton

        #plot_svd(folder, parameters)
        plot_pddf_acf(folder, parameters, max_z=80, n_bin=500)

        random.shuffle(parameters)
        parameters_train = parameters[:int(0.7*len(parameters))]
        parameters_test = parameters[int(0.7*len(parameters)):]

        #all_feature_mean, all_feature_std, all_gp_per_feature = GaussianProcess_optimization(folder, parameters_train, all_feature_names)
        #GaussianProcess_prediction(folder, parameters_test, all_feature_names, all_feature_mean, all_feature_std, all_gp_per_feature)




if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
