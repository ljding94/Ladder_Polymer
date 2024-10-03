#!/opt/homebrew/bin/python3
from plot_analyze import *
from process_analyze import *
import numpy as np
from ML_analysis import *
import sys
import random
import time
import os

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
        # folder = "../data/20240713_partial"
        folder = "../data/20241002"
        #segment_type = "inplane_twist"
        segment_type = "outofplane_twist"
        rand_max = 4000
        rand_read = 4000

        Ls = np.arange(50, 99.1, 1)
        logKts = [1.50]  # np.arange(0.00, 3.001, 0.10)
        logKbs = [1.50]
        Rfs = np.arange(0.41, 0.6001, 0.010)

        if (rand_max > 0):
            parameters = []
            for run_num in range(rand_max):
                filename = f"{folder}/obs_{segment_type}_random_run{run_num}_SqB.csv"
                if os.path.exists(filename):
                    parameters.append([segment_type, run_num])
                if len(parameters) >= rand_read:
                    break
                #[[segment_type, rand_num] for rand_num in range(rand_max)]
        else:
            parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]
        # parameteres = [[segment_type, L, Kt, Kb, Rf] for Kt in Kts for Kb in Kbs for Rf in Rfs]
        print("parameters", parameters)
        print("total number of parameters", len(parameters))

        #plot_svd(folder, parameters)
        #plot_pddf_acf(folder, parameters, max_z=6, n_bin=100)
        #return 0
        #random.shuffle(parameters) # already random input, no need to shuffle
        parameters_train = parameters[:int(0.7*len(parameters))]
        parameters_test = parameters[int(0.7*len(parameters)):]

        all_feature_mean, all_feature_std, all_gp_per_feature = GaussianProcess_optimization(folder, parameters_train)

        all_feature_names, all_feature_mean, all_feature_std, all_gp_per_feature = read_gp_and_feature_stats(folder, segment_type)

        GaussianProcess_prediction(folder, parameters_test, all_feature_mean, all_feature_std, all_gp_per_feature)
        exp_filename = "../data/incoh_banjo_expdata/merged_incoh_L2_Iq_subtracted_interpolated_Guinier_fit_n65_normalized_Iq.txt"
        GaussianProcess_experiment_data_analysis(exp_filename, all_feature_mean, all_feature_std, all_gp_per_feature)


        '''
        # folder = "../data/20240923"
        #down, mid, up
        lnLmu [2.96947531 2.96937991 2.96836665] [0.56847422 0.5684742  0.56847269]
        lnLsig [0.74834983 0.74834983 0.74834983] [0.07097229 0.07097229 0.07097229]
        L [29.55024371 29.47006761 29.17800976] [16.34289473 16.342378   16.33451951]
        PDI [1.76308105 1.76308105 1.76308105] [0.17532048 0.17532048 0.17532048]
        Rf [0.59907083 0.59929028 0.59675918] [0.11620755 0.11620631 0.11617948]
        Rg2 [22.27012691 22.26818746 22.25214424] [17.20949793 17.20949762 17.20947055]
        wRg2 [73.32925232 73.32925222 73.32924183] [59.10325171 59.10325171 59.10325171]
        '''


        '''
        # folder = "20240924"
        use log Sq for training
        lnLmu [2.49815074 2.49815074 2.49815074] [0.57864607 0.57864607 0.57864607]
        Rf [0.6034279 0.6034279 0.6034279] [0.11568807 0.11568807 0.11568807]
        Rg2 [13.03507684 13.03507684 13.03507684] [10.25880486 10.25880486 10.25880486]
        wRg2 [47.17362099 47.17362099 47.17362099] [38.16872514 38.16872514 38.16872514]
        L [18.650765 18.650765 18.650765] [10.47680502 10.47680502 10.47680502]
        PDI [1.77397991 1.77397991 1.77397991] [0.16923028 0.16923028 0.16923028]
        '''

        #calc_Sq_fitted_Rg2(folder, parameters_test)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
