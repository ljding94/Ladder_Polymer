// Copyright[2024] [Lijie Ding]
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include "biaxial_polymer.h"

int main(int argc, char const *argv[])
{
    std::clock_t c_start = std::clock();
    std::string folder;
    double beta = 1;
    Energy_parameter Epar;

    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm *timeinfo = std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d", timeinfo);
    std::string today(buffer);
    std::cout << today << std::endl;

    std::cout << "running with input argc:" << argc << "\n";
    for (int i = 0; i < argc; i++)
    {
        std::cout << argv[i] << " ";
    }

    // precision run with specified parameters
    if (argc == 9)
    {
        std::string segment_type = argv[1];
        double lnLmu = std::atof(argv[2]);
        double lnLsig = std::atof(argv[3]);
        Epar.Kt = std::atof(argv[4]); // It = Kt/L,logIt = log_10(It)
        // Epar.Kt = std::pow(10, logKt);
        Epar.Kb = std::atof(argv[5]); // Ib = Kb/L ~ normalized persistence length logIb = log_10(Ib)
        // Epar.Kb = std::pow(10, logKb);
        double Rf = std::atof(argv[6]);        // flip rate average if segment type is twist
        double abs_alpha = std::atof(argv[7]); // preferred bending angle
        std::string folder = std::string(argv[8]);

        biaxial_polymer polymer(segment_type, beta, lnLmu, lnLsig, Epar, Rf, abs_alpha);
        std::string finfo = std::string(argv[1]) + "_lnLmu" + std::string(argv[2]) + "_lnLsig" + std::string(argv[3]) + "_Kt" + std::string(argv[4]) + "_Kb" + std::string(argv[5]) + "_Rf" + std::string(argv[6]) + "_alpha" + std::string(argv[7]);

        int number_of_polymer;
        int bin_num;
        int save_n_config = 100;
        number_of_polymer = 2000;
        bin_num = 100;

        polymer.generate_polymer();
        polymer.save_polymer_to_file(folder + "/config_" + finfo + ".csv"); // save sample polymer

        polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder, finfo, save_n_config, true);
        // for precision run, save all measrued observables and Sq
        // polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + "_detail.csv", true);
    }

    // random run
    if (argc == 4)
    {
        std::string segment_type = argv[1];
        int run_num = std::atoi(argv[2]);
        std::string folder = std::string(argv[3]);

        int number_of_polymer = 10;
        int bin_num = 100;

        number_of_polymer = 2000;
        bin_num = 100;

        // single cluster job can run multiple processes
        biaxial_polymer polymer(segment_type, beta, 1, 1, Epar, 1, 1, true);
        std::string finfo = std::string(argv[1]) + "_random_run" + std::to_string(run_num);
        polymer.generate_polymer();
        // polymer.save_polymer_to_file(folder + "/config_" + finfo + ".csv"); // save sample polymer
        polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder, finfo, 0, false);
    }
    std::clock_t c_end = std::clock();
    double time_elapsed = static_cast<double>(c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time_elapsed << " seconds" << std::endl;
    return 0;
}