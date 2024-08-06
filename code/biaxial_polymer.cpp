#include "biaxial_polymer.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
// #define PI 3.14159265358979323846

// initialization
biaxial_polymer::biaxial_polymer(std::string segment_type_, double beta_, double lnLmu_, double lnLsig_, Energy_parameter Epar_, double Rf_, bool rand_param)
{
    // system related
    segment_type = segment_type_;
    beta = beta_;
    lnLmu = lnLmu_;
    lnLsig = lnLsig_;
    // set energy related
    // geometric
    Epar.Kt = Epar_.Kt;
    Epar.Kb = Epar_.Kb;
    Rf = Rf_;

    // set random number generators
    std::random_device rd;
    std::mt19937 gen_set(rd());
    std::uniform_real_distribution<> rand_uni_set(0, 1);
    std::normal_distribution<> rand_norm_set(0, 1); // normal

    gen = gen_set;
    rand_uni = rand_uni_set;
    rand_norm = rand_norm_set;

    if (rand_param)
    {
        // randomize parameters Lmu, Lsig etc
        lnLmu = 2.0 + 2.0 * rand_uni(gen);  // ln(avg_L) = mu + 0.5 sig^2
        lnLsig = 0.5 + 0.1 * rand_uni(gen); // [0.5,0.6] corresponds to roughly [0.8,1.2] PDI
        //Epar.Kt = std::pow(10, 1.0 + 1.0 * rand_uni(gen));
        Epar.Kt = 20 + 20 * rand_uni(gen);
        //Epar.Kb = std::pow(10, 1.0 + 1.0 * rand_uni(gen));
        Epar.Kb = 20 + 20 * rand_uni(gen);
        Rf = 0.4 + 0.4 * rand_uni(gen);
    }
    std::cout << "setting system param:" << "lnLmu:" << lnLmu << ", lnLsig:" << lnLsig << ", Kt:" << Epar.Kt << ", Kb:" << Epar.Kb << ", Rf:" << Rf << "\n";
}

inner_structure biaxial_polymer::calc_rand_ut_vt_alpha(std::vector<double> u, std::vector<double> v, double alpha_pre, double Rf)
{

    inner_structure inner;
    inner.alpha = 0;
    inner.ut = {0, 0, 0};
    inner.vt = {0, 0, 0};
    // calculate ut and vt based on u and v and segment_type and flip rate
    if (segment_type == "none" || segment_type == "flat")
    {
        inner.alpha = 0;
        inner.ut = u;
        inner.vt = v;
    }
    else if (segment_type == "inplane_twist")
    {
        // 2D canal, 51 degree in plane twist of ut and vt
        double alpha = 51.0 / 180.0 * M_PI; // to-be-determined
        double cos_alpha, sin_alpha;
        if (rand_uni(gen) < Rf && alpha_pre > 0)
        {
            alpha = -alpha; // flip with probability Rf
        }
        inner.alpha = alpha;

        cos_alpha = std::cos(alpha);
        sin_alpha = std::sin(alpha);
        for (int k = 0; k < 3; k++)
        {
            inner.ut[k] = u[k] * cos_alpha + v[k] * sin_alpha;
            inner.vt[k] = -u[k] * sin_alpha + v[k] * cos_alpha;
        }
    }
    else if (segment_type == "outofplane_twist")
    {
        // 2D canal, 51 degree out of plane twist of ut and vt
        double alpha = 51.0 / 180.0 * M_PI;
        double cos_alpha, sin_alpha;
        if (rand_uni(gen) < Rf && alpha_pre > 0)
        {
            alpha = -alpha; // flip with probability Rf
        }
        inner.alpha = alpha;

        inner.ut = Rodrigues_rotation(u, v, alpha);
        inner.vt = v;
    }
    else
    {
        throw std::runtime_error("Invalid polymer type");
    }
    // just to test in case there is an error
    if (std::abs(inner_product(inner.ut, inner.vt)) > 1e-5)
    {
        throw std::runtime_error("ut and vt are not orthogonal");
    }
    if (std::abs(inner_product(u, v)) > 1e-5)
    {
        throw std::runtime_error("u and v are not parallel");
    }

    return inner;
}

void biaxial_polymer::reset_polymer()
{
    // reset polymer to a straight line
    for (int i = 0; i < size(polymer); i++)
    {
        polymer[i].phi = 0;
        polymer[i].theta = 0;
        polymer[i].R = {0, 0, 1.0 * i};
        polymer[i].u = {0, 0, 1};
        polymer[i].v = {1, 0, 0};
        polymer[i].ut = {0, 0, 1};
        polymer[i].vt = {1, 0, 0};
        if (segment_type == "none")
        {
            polymer[i].scattering_points = {{0, 0}};
        }
        else if (segment_type == "flat")
        {
            polymer[i].scattering_points = {{0, -0.5}, {0, 0.5}, {0.5, -0.5}, {0.5, 0.5}};
            // polymer[i].scattering_points = {{0, -1}, {0, 1}, {0.5, -1}, {0.5, 1}};
            // polymer[i].scattering_points = {{0, -5}, {0, 5}, {0.5, -5}, {0.5, 5}};
            // polymer[i].scattering_points = {{0, 0}};
        }
        else if (segment_type == "inplane_twist")
        {
            polymer[i].scattering_points = {{0, 0}};
        }
        else if (segment_type == "outofplane_twist")
        {
            polymer[i].scattering_points = {{0, 0}};
        }
        else
        {
            throw std::runtime_error("Invalid polymer type");
        }
    }
}

bool biaxial_polymer::satisfy_self_avoiding_condition(int i)
{
    for (int j = 0; j < i; j++)
    {
        // comparing the distance between i's next segment's center and j
        if (std::pow(polymer[i].R[0] + polymer[i].u[0] - polymer[j].R[0], 2) + std::pow(polymer[i].R[1] + polymer[i].u[1] - polymer[j].R[1], 2) + std::pow(polymer[i].R[2] + polymer[i].u[2] - polymer[j].R[2], 2) < 1)
        {
            return false;
        }
    }
    return true;
}
int biaxial_polymer::generate_polymer()
{
    // determine the length of the polymer
    int L;
    do
    {
        L = int(std::exp(rand_norm(gen) * lnLsig + lnLmu));
    } while (L < 2);
    std::cout<<"generating L:"<<L<<"\n";
    polymer.resize(L);
    // std::cout<<"generating polymer with length L="<<L<<"\n";

    // generate the direction of first segment randomly
    polymer[0].u = rand_uni_vec();
    do
    {
        polymer[0].v = rand_uni_vec();
    } while (std::abs(inner_product(polymer[0].u, polymer[0].v) - 1) < 1e-5); // avoid the case that u and v are parallel
    // make u and v orthogonal
    polymer[0].v = cross_product(polymer[0].u, polymer[0].v);
    // make v unit vector
    double norm = std::sqrt(polymer[0].v[0] * polymer[0].v[0] + polymer[0].v[1] * polymer[0].v[1] + polymer[0].v[2] * polymer[0].v[2]);
    polymer[0].v[0] /= norm;
    polymer[0].v[1] /= norm;
    polymer[0].v[2] /= norm;

    // get ut and vt
    inner_structure inner = calc_rand_ut_vt_alpha(polymer[0].u, polymer[0].v, 1, Rf);
    polymer[0].alpha = inner.alpha;
    polymer[0].ut = inner.ut;
    polymer[0].vt = inner.vt;

    // generate polymer based on gaussian distribution of theta and phi
    double phi, theta;
    double total_trial = 0;
    double trial;
    int start_over_count = 0;
    for (int i = 1; i < L; i++) // keep the 0th segment fixed
    {
        // update R, determined by previous segment
        for (int j = 0; j < 3; j++)
        {
            polymer[i].R[j] = polymer[i - 1].R[j] + polymer[i - 1].u[j];
        }
        trial = 0;
        do
        {
            total_trial++;
            trial++;
            phi = rand_norm(gen) / std::sqrt(beta * Epar.Kt);   // sqrt(beta Kt) * phi ~ N(0,1)
            theta = rand_norm(gen) / std::sqrt(beta * Epar.Kb); // sqrt(beta Kb) * theta ~ N(0,1)

            polymer[i].phi = phi;
            polymer[i].theta = theta;

            // note R is the head of the segment, not center

            // update u and v
            // 1. polymer[i] rorate around polymer[i-1].u by phi
            // 2. polymer[i] rorate around polymer[i].v by theta, (for simplicity)
            // use Rodrigues' rotation formula

            // rotate by phi, from vt and ut
            polymer[i].u = polymer[i - 1].ut; // u unchanged
            polymer[i].v = Rodrigues_rotation(polymer[i - 1].vt, polymer[i - 1].ut, phi);

            // rotate by theta
            polymer[i].u = Rodrigues_rotation(polymer[i].u, polymer[i].v, theta);
            // v unchanged

        } while (!satisfy_self_avoiding_condition(i) && trial < 1e4);
        if (trial >= 1e4)
        {
            start_over_count++;
            // too many failed trial, seems stuck in one place, start over from half of it's place
            i /= 2 * start_over_count; // lower the start over i evry time there is a start over
        }
        if(start_over_count > 1e4)
        {
            std::cout<<"too many start over, break\n";
            return 0;
        }
        // accepted, also update ut and vt
        inner_structure inner = calc_rand_ut_vt_alpha(polymer[i].u, polymer[i].v, polymer[i - 1].alpha, Rf);
        polymer[i].alpha = inner.alpha;
        polymer[i].ut = inner.ut;
        polymer[i].vt = inner.vt;
    }
    return 1; // success
    // std::cout<<"polymer generation accepted rate: " <<(L-1)/total_trial <<std::endl;
}

void biaxial_polymer::save_polymer_to_file(std::string filename)
{
    std::ofstream f(filename);
    if (f.is_open())
    {
        f << "phi,theta,Rx,Ry,Rz,ux,uy,uz,vx,vy,vz,alpha,utx,uty,utz,vtx,vty,vtz\n";

        for (int i = 0; i < size(polymer); i++)
        {
            f << polymer[i].phi << "," << polymer[i].theta << "," << polymer[i].R[0] << "," << polymer[i].R[1] << "," << polymer[i].R[2] << "," << polymer[i].u[0] << "," << polymer[i].u[1] << "," << polymer[i].u[2] << "," << polymer[i].v[0] << "," << polymer[i].v[1] << "," << polymer[i].v[2] << "," << polymer[i].alpha << "," << polymer[i].ut[0] << "," << polymer[i].ut[1] << "," << polymer[i].ut[2] << "," << polymer[i].vt[0] << "," << polymer[i].vt[1] << "," << polymer[i].vt[2] << "\n";
        }
    }
    f.close();

} // end save_polymer_to_file

void biaxial_polymer::save_observable_to_file(std::string filename, std::vector<observable> obs_ensemble, bool save_detail)
{
    std::ofstream f(filename);
    if (f.is_open())
    {
        f << "beta=" << beta << "\n";
        f << "lnLmu=" << lnLmu << "\n";
        f << "lnLsig=" << lnLsig << "\n";
        f << "Kt=" << Epar.Kt << "\n";
        f << "Kb=" << Epar.Kb << "\n";

        int number_of_polymer = obs_ensemble.size();
        if (save_detail)
        {
            f << "index,E,Tot_phi,Tot_theta\n";
            for (int i = 0; i < number_of_polymer; i++)
            {
                f << i << "," << obs_ensemble[i].E << "," << obs_ensemble[i].Tot_phi << "," << obs_ensemble[i].Tot_theta << "\n";
            }
        }
        else
        {
            // find stats
            // calculate average and standard deviation of E
            double avg_E = 0.0;
            double std_dev_E = 0.0;
            double avg_Tot_phi = 0.0;
            double std_dev_Tot_phi = 0.0;
            double avg_Tot_theta = 0.0;
            double std_dev_Tot_theta = 0.0;

            for (int i = 0; i < number_of_polymer; i++)
            {
                avg_E += obs_ensemble[i].E;
                avg_Tot_phi += obs_ensemble[i].Tot_phi;
                avg_Tot_theta += obs_ensemble[i].Tot_theta;
            }
            avg_E /= number_of_polymer;
            avg_Tot_phi /= number_of_polymer;
            avg_Tot_theta /= number_of_polymer;

            for (int i = 0; i < number_of_polymer; i++)
            {
                std_dev_E += (obs_ensemble[i].E - avg_E) * (obs_ensemble[i].E - avg_E);
                std_dev_Tot_phi += (obs_ensemble[i].Tot_phi - avg_Tot_phi) * (obs_ensemble[i].Tot_phi - avg_Tot_phi);
                std_dev_Tot_theta += (obs_ensemble[i].Tot_theta - avg_Tot_theta) * (obs_ensemble[i].Tot_theta - avg_Tot_theta);
            }
            std_dev_E = std::sqrt(std_dev_E / number_of_polymer);
            std_dev_Tot_phi = std::sqrt(std_dev_Tot_phi / number_of_polymer);
            std_dev_Tot_theta = std::sqrt(std_dev_Tot_theta / number_of_polymer);
            // write stats to the file
            f << "stats,E,Tot_phi,Tot_theta,gr_array\n";
            f << "mean," << avg_E << "," << avg_Tot_phi << "," << avg_Tot_theta << "\n";
            f << "std_dev," << std_dev_E << "," << std_dev_Tot_phi << "," << std_dev_Tot_theta << "\n";
        }
    }
    f.close();
}

void biaxial_polymer::save_L_weighted_Sq_to_file(std::string filename, std::vector<observable> obs_ensemble, bool save_detail)
{

    std::ofstream f(filename);
    if (f.is_open())
    {
        if (save_detail)
        {
            f << "beta=" << beta << "\n" << "segment_type=" << segment_type << "\n"
              << "lnLmu=" << lnLmu << "\n" << "lnLsig=" << lnLsig << "\n" << "Kt=" << Epar.Kt << "\n" << "Kb=" << Epar.Kb << "\n"
              << "Rf=" << Rf << "\n";
            f << "Rg2,L,SqB_array\n";
            for (int i = 0; i < obs_ensemble.size(); i++)
            {
                f << obs_ensemble[i].Rg2 << "," << obs_ensemble[i].L;
                for (int j = 0; j < obs_ensemble[i].SqB.size(); j++)
                {
                    f << "," << obs_ensemble[i].SqB[j];
                }
                f << "\n";
            }
        }
        else
        {
            // get Rg statistics
            double avg_Rg2 = 0.0;
            // get L statistivs
            double avg_L = 0.0;
            double avg_L2 = 0.0; // L^2

            for (int i = 0; i < obs_ensemble.size(); i++)
            {
                avg_Rg2 += obs_ensemble[i].Rg2;
                avg_L += obs_ensemble[i].L;
                avg_L2 += obs_ensemble[i].L * obs_ensemble[i].L;
            }
            avg_Rg2 /= obs_ensemble.size();
            avg_L /= obs_ensemble.size();
            avg_L2 /= obs_ensemble.size();

            // find stats
            // calculate average and standard deviation of gr
            std::vector<double> avg_weighted_SqB(obs_ensemble[0].SqB.size(), 0.0);
            for (int j = 0; j < obs_ensemble[0].SqB.size(); j++) // every bin or point
            {
                for (int i = 0; i < obs_ensemble.size(); i++)
                {
                    avg_weighted_SqB[j] += obs_ensemble[i].L * obs_ensemble[i].L * obs_ensemble[i].SqB[j] * obs_ensemble.size()/avg_L2;
                }
            }
            // write stats to the file
            f << "stats,segment_type,lnLmu,lnLsig,Kt,Kb,Rf,Rg2,L,L2,SqB_array\n";
            f << "mean" << "," << segment_type << "," << lnLmu << "," << lnLsig << "," << Epar.Kt << "," << Epar.Kb << "," << Rf << "," << avg_Rg2
            << "," << avg_L << "," << avg_L2;
            for (int j = 0; j < avg_weighted_SqB.size(); j++)
            {
                f << "," << avg_weighted_SqB[j];
            }
            f << "\nstd_dev/sqrt(number of polymer)";
            f << "\nr or qB " << ",NA, NA, NA, NA, NA, NA, NA, NA, NA";
            for (int i = 0; i < obs_ensemble[0].qB.size(); i++)
            {
                f << "," << obs_ensemble[0].qB[i];
            }
        }
    }
    f.close();
}
void biaxial_polymer::generate_and_save_polymer_ensemble(int number_of_polymer, int bin_num, std::string filename, bool save_detail)
{
    std::vector<observable> obs_ensemble;
    // generate polymer ensemble and record observable
    int current_progress = 0;
    int polymer_count = 0;
    while(polymer_count < number_of_polymer)
    {
        reset_polymer();
        if(generate_polymer())
        {
            obs_ensemble.push_back(measure_observable(bin_num));
            polymer_count++;
        }
        if (int((polymer_count + 1) * 100.0 / number_of_polymer) > current_progress + 1)
        {
            current_progress = int((polymer_count + 1) * 100.0 / number_of_polymer);
            std::cout << "Progress: " << current_progress << "\%" << "\n";
        }
    }

    // std::string obs_filename = filename.substr(0, filename.find_last_of(".")) + "_obs.csv";
    // save_observable_to_file(filename, obs_ensemble, save_detail);

    // std::string gr_filename = filename.substr(0, filename.find_last_of(".")) + "_gr.csv";
    // save_distribution_function_to_file(gr_filename, "gr", obs_ensemble, save_detail);

    std::string Sq_filename = filename.substr(0, filename.find_last_of(".")) + "_SqB.csv";
    save_L_weighted_Sq_to_file(Sq_filename, obs_ensemble, save_detail);
    // save_distribution_function_to_file(Sq_filename, "SqB", obs_ensemble, save_detail);
}

observable biaxial_polymer::measure_observable(int bin_num)
{
    // measure observable
    observable obs;
    obs.E = 0;
    obs.Tot_phi = 0;
    obs.Tot_theta = 0;
    for (int i = 0; i < size(polymer); i++)
    {
        obs.E += 0.5 * Epar.Kt * polymer[i].phi * polymer[i].phi + 0.5 * Epar.Kb * polymer[i].theta * polymer[i].theta;
        obs.Tot_phi += polymer[i].phi;
        obs.Tot_theta += polymer[i].theta;
    }
    /*
    obs.gr = calc_pair_dsitribution_function(0, L, bin_num); // no need for this now
    obs.r = std::vector<double>(bin_num,0);
    for (int i = 0; i < bin_num; i++) {
        obs.r[i] = 1.0 * i * L/bin_num;
    }
    */
    obs.Rg2 = calc_radius_of_gyration_square();

    double qB_i = 1e-4; // 0.2*M_PI/L; //0.1/L; ;
    double qB_f = 1e0;  // M_PI;//100.0/L; //M_PI;
    obs.SqB = calc_structure_factor(qB_i, qB_f, bin_num);
    obs.qB = std::vector<double>(bin_num, 0);
    for (int k = 0; k < bin_num; k++)
    {
        obs.qB[k] = qB_i * std::pow(qB_f / qB_i, 1.0 * k / (bin_num - 1)); // uniform in log scale
    }
    obs.L = polymer.size();
    return obs;
}

std::vector<double> biaxial_polymer::calc_pair_dsitribution_function(double r_i, double r_f, int bin_num)
{
    // measure pair distribution function
    int L = size(polymer);

    std::vector<double> gr(bin_num, 0);
    int bin;
    double delta_r = (r_f - r_i) / bin_num;
    double r;
    for (int i = 0; i < L - 1; i++)
    {
        for (int j = i + 1; j < L; j++)
        {
            r = std::sqrt(std::pow(polymer[i].R[0] - polymer[j].R[0], 2) + std::pow(polymer[i].R[1] - polymer[j].R[1], 2) + std::pow(polymer[i].R[2] - polymer[j].R[2], 2));
            bin = int((r - r_i) / delta_r);
            if (bin < bin_num)
            {
                gr[bin] += 2.0 / L;
            }
        }
    }
    return gr;
}

std::vector<double> biaxial_polymer::calc_structure_factor(double qB_i, double qB_f, int bin_num)
{
    // measure structure factor
    int bin;
    double q;

    std::vector<std::vector<double>> R_all{}; // all scattering point's R[axis] value, including all scattering points in each segment
    std::vector<double> R_scatter{0, 0, 0};

    for (int i = 0; i < size(polymer); i++)
    {
        for (int j = 0; j < polymer[i].scattering_points.size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                R_scatter[k] = polymer[i].R[k] + polymer[i].scattering_points[j][0] * polymer[i].u[k] + polymer[i].scattering_points[j][1] * polymer[i].v[k];
            }
            R_all.push_back(R_scatter);
        }
    }
    int N = R_all.size(); // total number of scattering points
    double r;             // for storating distance between two scattering points
    std::vector<double> all_qB{};
    for (int k = 0; k < bin_num; k++)
    {
        all_qB.push_back(qB_i * std::pow(qB_f / qB_i, 1.0 * k / (bin_num - 1))); // uniform in log scale
    }
    // std::cout<< "N:"<<N<<std::endl;
    std::vector<double> SqB(bin_num, 1.0 / N); // initialize with 1 due to self overlaping term (see S.H. Chen 1986 eq 18)
    for (int i = 0; i < N - 1; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            r = std::sqrt(std::pow(R_all[i][0] - R_all[j][0], 2) + std::pow(R_all[i][1] - R_all[j][1], 2) + std::pow(R_all[i][2] - R_all[j][2], 2));
            // calculate S_q
            for (int k = 0; k < bin_num; k++)
            {
                q = all_qB[k];                                     // B = 1
                SqB[k] += 2.0 / N / N * std::sin(q * r) / (q * r); // normalize to 1 at q=0
            }
        }
        // std::cout<<"Progress: " << k << "/" << bin_num <<  "\r";
    }
    return SqB;
}

double biaxial_polymer::calc_radius_of_gyration_square()
{
    int L = size(polymer);
    // calculate radius of gyration
    double Rg2 = 0;
    // find center of mass
    double x_c = 0, y_c = 0, z_c = 0;
    for (int i = 0; i < L; i++)
    {
        x_c = x_c + polymer[i].R[0];
        y_c = y_c + polymer[i].R[1];
        z_c = z_c + polymer[i].R[2];
    }
    x_c = x_c / L;
    y_c = y_c / L;
    z_c = z_c / L;

    // calculate Rg
    for (int i = 0; i < L; i++)
    {
        Rg2 += std::pow(polymer[i].R[0] - x_c, 2) + std::pow(polymer[i].R[1] - y_c, 2) + std::pow(polymer[i].R[2] - z_c, 2);
    }
    Rg2 /= L;
    return Rg2;
}

#pragma region : useful tools
std::vector<double> biaxial_polymer::calc_rod_structure_factor(std::vector<double> qB)
{
    int L = size(polymer);
    // measure structure factor
    std::vector<double> SqB(qB.size(), 1.0 / L); // initialize with 1 due to self overlaping term (see S.H. Chen 1986 eq 18)
    for (int i = 0; i < L - 1; i++)
    {
        for (int j = i + 1; j < L; j++)
        {
            for (int k = 0; k < qB.size(); k++)
            {
                SqB[k] += 2.0 / L / L * std::sin(qB[k] * (j - i)) / (qB[k] * (j - i)); // normalize to 1 at q=0
            }
        }
    }
    return SqB;
}

std::vector<double> biaxial_polymer::cross_product(std::vector<double> a, std::vector<double> b)
{
    std::vector<double> c(3, 0);
    for (int j = 0; j < 3; j++)
    {
        c[j] = a[(j + 1) % 3] * b[(j + 2) % 3] - a[(j + 2) % 3] * b[(j + 1) % 3];
    }
    return c;
}

double biaxial_polymer::inner_product(std::vector<double> a, std::vector<double> b)
{
    double c = 0;
    for (int j = 0; j < 3; j++)
    {
        c += a[j] * b[j];
    }
    return c;
}

std::vector<double> biaxial_polymer::Rodrigues_rotation(std::vector<double> v, std::vector<double> k, double theta)
{
    // v rotate around k by theta
    std::vector<double> v_rot(3, 0);
    // 1. find (k x v), k cross v and kv, k.v
    std::vector<double> kxv = cross_product(k, v);
    double kv = inner_product(k, v);
    // 2. do the calculation
    for (int j = 0; j < 3; j++)
    {
        v_rot[j] = v[j] * std::cos(theta) + kxv[j] * std::sin(theta) + k[j] * kv * (1 - std::cos(theta));
    }
    // 3. renormalize v_rot
    double v_rot_norm = std::sqrt(v_rot[0] * v_rot[0] + v_rot[1] * v_rot[1] + v_rot[2] * v_rot[2]);
    v_rot[0] /= v_rot_norm;
    v_rot[1] /= v_rot_norm;
    v_rot[2] /= v_rot_norm;

    if (std::abs(v_rot[0] * v_rot[0] + v_rot[1] * v_rot[1] + v_rot[2] * v_rot[2] - 1) > 1e-6)
    {
        printf("v_rot^2=%f\n", v_rot[0] * v_rot[0] + v_rot[1] * v_rot[1] + v_rot[2] * v_rot[2]);
        throw std::runtime_error("Error: v_rot^2 is not 1.");
    }

    return v_rot;
}

std::vector<double> biaxial_polymer::rand_uni_vec()
{
    // 1. generat random point in unit ball
    double x, y, z;
    do
    {
        x = 2 * rand_uni(gen) - 1;
        y = 2 * rand_uni(gen) - 1;
        z = 2 * rand_uni(gen) - 1;
    } while (x * x + y * y + z * z > 1);
    // 2. normalize
    double norm = std::sqrt(x * x + y * y + z * z);
    x /= norm;
    y /= norm;
    z /= norm;
    return {x, y, z};
}

#pragma endregion