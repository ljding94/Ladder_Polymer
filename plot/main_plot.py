#!/opt/homebrew/bin/python3
from config_plot import *
from Sq_plot import *
from SVD_plot import *
from GPR_plot import *


def main():

    plot_molecule_structure()

    #plot_illustrative_config()
    #plot_flipping_demo()



    #plot_polymer_Sq()
    #plot_SVD_feature_data()
    #plot_GPR_data()
    #plot_PDDF_ACF_LML_data()

    #plot_Sq_fitted_Rg2()

    # arvhived plots
    #plot_polymer_configs()
    #plot_animate_polymer_configs()
    #plot_SVD_data() # outdated
    #plot_Rg_L_relation() # no longer used


if __name__ == '__main__':
    main()