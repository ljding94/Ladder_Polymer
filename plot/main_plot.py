#!/opt/homebrew/bin/python3
from config_plot import *
from Sq_plot import *
from SVD_plot import *
from GPR_plot import *


def main():
    #plot_polymer_configs()
    #plot_animate_polymer_configs()

    #plot_polymer_Sq()
    #plot_Rg_L_relation()

    #plot_SVD_data()
    plot_SVD_feature_data()
    #plot_GPR_data()
    #plot_PDDF_ACF_LML_data()

if __name__ == '__main__':
    main()