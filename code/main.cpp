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

    // precision run with specified parameters
    if (argc == 8 || argc == 7)
    {
        std::string segment_type = argv[1];
        int Lmu = std::atoi(argv[2]);
        int Lsig = std::atoi(argv[3]);
        double logKt = std::atof(argv[4]); // It = Kt/L,logIt = log_10(It)
        Epar.Kt = std::pow(10, logKt);
        double logKb = std::atof(argv[5]); // Ib = Kb/L ~ normalized persistence length logIb = log_10(Ib)
        Epar.Kb = std::pow(10, logKb);
        double Rf = std::atof(argv[6]); // flip rate average if segment type is twist

        biaxial_polymer polymer(segment_type, beta, Lmu, Lsig, Epar, Rf);
        std::string finfo = std::string(argv[1]) + "_Lmu" + std::string(argv[2]) + "_Lsig" + std::string(argv[3]) + "_logKt" + std::string(argv[4]) + "_logKb" + std::string(argv[5]) + "_Rf" + std::string(argv[6]);

        int number_of_polymer;
        int bin_num;
        if(argc == 8)
        {
            number_of_polymer = 100;
            bin_num = 100;
            // use "prog name par* local" for local running
            // used for local running!
            std::cout << "running on local machine\n";
            folder = "../data/scratch_local/" + today;
        } else {
            number_of_polymer = 1000;
            bin_num = 100;
            // running on cluster
            std::cout << "running on cluster\n";
            folder = "/home/dingl1/ladder_polymer/data_Cades/data_pool"; // dump data to data pool first
        }
        if (!std::filesystem::exists(folder))
            {
                std::cout << today << " folder not exist\n";
                std::cout << "creating folder" << folder << "\n";
                std::filesystem::create_directory(folder);
            }
        // polymer.save_polymer_to_file(folder + "/config_" + finfo + "_init.csv");
        polymer.generate_polymer();
        polymer.save_polymer_to_file(folder + "/config_" + finfo + ".csv"); // save sample polymer

        polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + ".csv");
        // polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + "_detail.csv", true);
    }

    // random run
    if (argc == 3 || argc == 4)
    {
        std::string segment_type = argv[1];
        int run_num = std::atoi(argv[2]);

        int number_of_polymer = 10;
        int bin_num = 100;

        if(argc==4)
        {
            number_of_polymer = 10;
            bin_num = 100;
            std::cout << "running on local machine\n";
            folder = "../data/scratch_local/" + today;
        } else {
            number_of_polymer = 1000;
            bin_num = 100;
            // running on cluster
            std::cout << "running on cluster\n";
            folder = "/home/dingl1/ladder_polymer/data_Cades/data_pool"; // dump all data to data pool to avoid different finish day issue
        }
        if (!std::filesystem::exists(folder))
        {
            std::cout << today << " folder not exist\n";
            std::cout << "creating folder" << folder << "\n";
            std::filesystem::create_directory(folder);
        }

        // due to job submission limit, run L to L +10 for each cluster job
        // run 10 run per job
        for (int n = run_num; n < run_num + 10; n++)
        {
            biaxial_polymer polymer(segment_type, beta, 1, 1, Epar, 1, true);
            std::string finfo = std::string(argv[1]) + "_random_run" + std::to_string(n);
            polymer.generate_polymer();
            // polymer.save_polymer_to_file(folder + "/config_" + finfo + ".csv"); // save sample polymer
            polymer.generate_and_save_polymer_ensemble(number_of_polymer, bin_num, folder + "/obs_" + finfo + ".csv");
        }
    }

    std::clock_t c_end = std::clock();
    double time_elapsed = static_cast<double>(c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << time_elapsed << " seconds" << std::endl;

    return 0;
    return 0;

}