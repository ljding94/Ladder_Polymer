#!/opt/homebrew/bin/python3
from plot_analyze import *
from process_analyze import *
import numpy as np


def main():
    print("hello!")
    plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.10.csv")
    plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.90.csv")
    plot_polymer_config("../data/scratch_local/20240605/config_outofplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.10.csv")
    plot_polymer_config("../data/scratch_local/20240605/config_outofplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.90.csv")
    #plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L100_Kt20.00_Kb20.00_Rf0.50.csv")
    #plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L100_Kt20.00_Kb20.00_Rf0.90.csv")
    #plot_polymer_config("../data/scratch_local/20240604/config_2DCanal_L10_Kt1000.00_Kb1000.00_Rf0.90.csv")
    #plot_polymer_config("../data/scratch_local/20240604/config_2DCanal_L10_Kt1000.00_Kb1000.00_Rf0.10.csv")
    return 0
    '''
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt1.0_Kb1.0.csv")
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt10.0_Kb1.0.csv")
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt100.0_Kb1.0.csv")
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt1000.0_Kb1.0.csv")
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt10000.0_Kb1.0.csv")


    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt1.0_Kb1.0.csv")
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt1.0_Kb3.0.csv")
    plot_polymer_config("../data/scratch_local/20240604/config_none_L500_Kt1.0_Kb10000.0.csv")
    '''

    # plot_polymer_config("../data/scratch_local/config_L100_Kt100_Kb5.csv")
    # plot_polymer_config("../data/scratch_local/30May24/config_L20_Kt10_Kb10.csv")
    # return 0

    # transform_pair_distribution_data_to_structure_factor("../data/scratch_local/30May24/obs_L20_Kt10_Kb10_gr.csv")
    # plot_stats_distribution_function("../data/scratch_local/30May24/obs_L20_Kt10_Kb10_.csv", ["Sqx", "Sqy", "Sqz", "gr"])
    # plot_stats_distribution_function("../data/scratch_local/30May24/pre_random_init/obs_L20_Kt10_Kb10_.csv", ["Sqx", "Sqy", "Sqz"])

    # plot_observable_distribution("../data/scratch_local/obs_L20_Kt1_Kb1_detail.csv")
    # plot_average_pair_distribution_function("../data/scratch_local/obs_L20_Kt1_Kb1_detail.csv")
    # plot_stats_pair_distribution_function("../data/scratch_local/obs_L100_Kt1.0_Kb1.0.csv")

    # plot_stats_obervable_for_multiple_parameters([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])
    # plot_stats_structure_factor_for_multiple_parameters([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])

    folder = "../data/scratch_local/20240604"
    segment_type = "flat"
    segment_type = "none"
    L = 100
    Kts = np.arange(1.0, 20.01, 1.0)
    #Kt = [0.1, 1.0, 10.0, 100.0]
    Kbs = np.arange(1.0, 20.01, 1.0)

    #plot_polymer_config_with_multiple_parameters(folder, segment_type, L, Kt, Kb)
    plot_stats_structure_factor_for_multiple_parameters(folder, segment_type, L, Kts, Kbs)
    plot_stats_structure_factor_for_multiple_parameters(folder, segment_type, L, Kts, Kbs, True)





if __name__ == "__main__":
    main()
