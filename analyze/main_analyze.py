#!/opt/homebrew/bin/python3
from plot_analyze import *
from process_analyze import *
import numpy as np
from experiment_data_analyze import *

def main():
    print("hello!")
    # plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.10.csv")
    # plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.90.csv")
    # plot_polymer_config("../data/scratch_local/20240605/config_outofplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.10.csv")
    # plot_polymer_config("../data/scratch_local/20240605/config_outofplane_twist_L20_Kt1000.00_Kb1000.00_Rf0.90.csv")
    # plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L100_Kt20.00_Kb20.00_Rf0.50.csv")
    # plot_polymer_config("../data/scratch_local/20240605/config_inplane_twist_L100_Kt20.00_Kb20.00_Rf0.90.csv")
    # plot_polymer_config("../data/scratch_local/20240604/config_2DCanal_L10_Kt1000.00_Kb1000.00_Rf0.90.csv")
    # plot_polymer_config("../data/scratch_local/20240604/config_2DCanal_L10_Kt1000.00_Kb1000.00_Rf0.10.csv")
    # return 0

    folder = "../data/scratch_local/20240604"
    segment_type = "flat"
    segment_type = "none"
    L = 100
    Kts = np.arange(1.0, 20.01, 1.0)
    # Kt = [0.1, 1.0, 10.0, 100.0]
    Kbs = np.arange(1.0, 20.01, 1.0)

    # plot_polymer_config_with_multiple_parameters(folder, segment_type, L, Kt, Kb)
    # plot_stats_structure_factor_for_multiple_parameters(folder, segment_type, L, Kts, Kbs)
    # plot_stats_structure_factor_for_multiple_parameters(folder, segment_type, L, Kts, Kbs, True)

    plot_experimental_Sq()


if __name__ == "__main__":
    main()
