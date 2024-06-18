#!/opt/homebrew/bin/python3
from plot_analyze import *
from process_analyze import *
import numpy as np
from ML_analysis import *
import sys


def main():
    if len(sys.argv) > 1:
        date = sys.argv[1]
        segment_type = sys.argv[2]
        folder = "/home/dingl1/ladder_polymer/data_Cades/"+date
        print(f"analyzing folder: {folder} segment type {segment_type}")
        Ls = np.arange(50, 209.1, 1)
        logKts = [1.50]
        logKbs = [1.50]
        Rfs = np.arange(0.41, 0.6001, 0.005)
        parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]
        all_feature_names = ["L", "Kt", "Kb", "Rf", "Rg"]  # "Kt", "Kb",
        print("analyzing parameters", parameters)
        plot_svd(folder, parameters, all_feature_names)
        plot_pddf_acf(folder, parameters, all_feature_names)
        GaussianProcess_optimization(folder, parameters, all_feature_names)
        GaussianProcess_prediction(folder, parameters, all_feature_names)

    else:
        print("hello!")
        # folder = "../data/scratch_local/20240610"
        folder = "../data/20240613"
        segment_type = "flat"
        segment_type = "none"
        segment_type = "inplane_twist"
        #segment_type = "outofplane_twist"
        # L = 200
        Ls = np.arange(50, 109.1, 1)
        # Kbs = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        logKts = [1.50]  # np.arange(0.00, 3.001, 0.10)
        logKbs = [1.50]
        # Kt = [0.1, 1.0, 10.0, 100.0]
        # logIbs = np.arange(-2.00, -1.001, 0.10)
        Rfs = np.arange(0.41, 0.6001, 0.010)
        # Rfs = np.arange(0.50, 0.601, 0.01)
        parameters = [[segment_type, L, logKt, logKb, Rf] for L in Ls for logKt in logKts for logKb in logKbs for Rf in Rfs]
        # parameteres = [[segment_type, L, Kt, Kb, Rf] for Kt in Kts for Kb in Kbs for Rf in Rfs]
        print("parameters", parameters)

        all_feature_names = ["L", "Kt", "Kb", "Rf", "Rg"]  # "Kt", "Kb",

        #plot_svd(folder, parameters, all_feature_names)
        #plot_pddf_acf(folder, parameters, all_feature_names)
        #GaussianProcess_optimization(folder, parameters, all_feature_names)
        GaussianProcess_prediction(folder, parameters, all_feature_names)


if __name__ == "__main__":
    main()
