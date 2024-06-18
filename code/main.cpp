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
    std::string segment_type = argv[1];
    int L = std::atoi(argv[2]);
    Energy_parameter Epar;
    double logKt = std::atof(argv[3]); // It = Kt/L,logIt = log_10(It)
    Epar.Kt = std::pow(10, logKt);
    double logKb = std::atof(argv[4]); // Ib = Kb/L ~ normalized persistence length logIb = log_10(Ib)
    Epar.Kb = std::pow(10, logKb);
    double Rf = std::atof(argv[5]); // flip rate average if segment type is twist

    biaxial_polymer polymer(segment_type, beta, L, Epar, Rf);
    std::string finfo = std::string(argv[1]) + "_L" + std::string(argv[2]) + "_logKt" + std::string(argv[3]) + "_logKb" + std::string(argv[4]) + "_Rf" + std::string(argv[5]);

    int number_of_polymer = 1;
    int bin_num = 100;


    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm* timeinfo = std::localtime(&now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d", timeinfo);
    std::string today(buffer);
    std::cout << today << std::endl;

    if (argc == 7)
    {
        // use "prog name par* local" for local running
        // used for local running!
        std::cout << "running on local machine\n";

        folder = "../data/scratch_local/"+today;
        if (!std::filesystem::exists(folder))
        {
            std::cout<< today << " folder not exist\n";
            std::cout<< "creating folder" << folder << "\n";
            std::filesystem::create_directory(folder);
        }

        //polymer.save_polymer_to_file(folder + "/config_" + finfo + "_init.csv");
        polymer.generate_polymer();
        polymer.save_polymer_to_file(folder + "/config_" + finfo + ".csv"); // save sample polymer

        polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + ".csv");

        //polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + "_detail.csv", true);
    }

    if (argc == 6)
    {
        number_of_polymer = 1000;
        // running on cluster
        std::cout << "running on cluster\n";

        folder = "/home/dingl1/Ladder_Polymer/data_Cades/"+today;
        if (!std::filesystem::exists(folder))
        {
            std::cout<< today << " folder not exist\n";
            std::cout<< "creating folder" << folder << "\n";
            std::filesystem::create_directory(folder);
        }

        // due to job submission limit, run L to L +10 for each cluster job
        for( int l=L ; l < L+10; l++)
        {
            biaxial_polymer polymer(segment_type, beta, l, Epar, Rf);
            std::string finfo = std::string(argv[1]) + "_L" + std::to_string(l) + "_logKt" + std::string(argv[3]) + "_logKb" + std::string(argv[4]) + "_Rf" + std::string(argv[5]);
            polymer.generate_polymer();
            //polymer.save_polymer_to_file(folder + "/config_" + finfo + ".csv"); // save sample polymer

            polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + ".csv");
        }
    }


    std::clock_t c_end = std::clock();
    double time_elapsed = static_cast<double>(c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time_elapsed << " seconds" << std::endl;

    return 0;

}