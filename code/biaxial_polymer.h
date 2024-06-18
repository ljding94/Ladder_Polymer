#ifndef _BIAXIAL_POLYMER_H
#define _BIAXIAL_POLYMER_H
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

// observable
struct observable
{
    // total energy
    double E=0;
    // geometric
    double Tot_phi=0; // total twist
    double Tot_theta=0; // total bend
    std::vector<double> gr; // pair distribution function
    std::vector<double> r; // pair distribution function's r

    double Rg; // radius of gyration
    std::vector<double> SqB; // structure factor, orientational averaged: ref: Pedersen 1996 equ 1,
    std::vector<double> qB; // structure factor's q, rotational anveraged, B is the monomer length = 1 as unit length in the simulation

};
//hamiltonion parameters
struct Energy_parameter
{
    double Kt=0; // twisting modulii
    double Kb=0; // bending modulii
};
struct segment
{
    // generation related
    double phi; // twist: cos(phi) = v.v'
    double theta; // bend: cos(theta) = u.u'
    // configuration related
    std::vector<double> R{0, 0, 0}; // position (x,y,z)
    std::vector<double> u{0, 0, 1.0}; // long axis (u_x,u_y,u_z)
    std::vector<double> v{1.0, 0, 0}; // short axis (v_x,v_y,v_z)

    // internal config related
    double alpha; // record the twist angle alpha for this segment
    std::vector<double> ut{0, 0, 1.0}; // long axis of tail (u_x,u_y,u_z)
    std::vector<double> vt{1.0, 0, 0}; // short axis of tail (v_x,v_y,v_z)
    // need to calculate ut and vt based segment type

    // scattering related
    std::vector<std::vector<double>> scattering_points{{0,0}}; // scattering points for segment, positions in u,v space

};
struct inner_structure
{
    double alpha; // record the twist angle alpha for this segment
    std::vector<double> ut{0, 0, 1.0}; // long axis of tail (u_x,u_y,u_z)
    std::vector<double> vt{1.0, 0, 0}; // short axis of tail (v_x,v_y,v_z)
};


class biaxial_polymer
{
public:
    // eternal parameters
    std::string segment_type; // segment type
    double beta; // system temperature
    int L;       // length, number of segments
    Energy_parameter Epar;
    double Rf; // flip rate
    std::vector<segment> polymer; // the polymer
    //observable obs; // obserrable


    // randomnumber generators
    std::mt19937 gen;
    std::uniform_real_distribution<> rand_uni; // uniform distribution
    std::normal_distribution<> rand_norm; // normal distribution

    std::vector<double> rand_uni_vec(); // generate a random unit vector

    // initialization
    biaxial_polymer(std::string segment_type_, double beta_, int L_, Energy_parameter Epar_, double Rf_);
    // for direct sampling

    // check self-avoid condition
    bool satisfy_self_avoiding_condition(int i); // check if polymer[i] satisfy self-avoiding condition with all k>i segments
    // generation
    //segment create_initialized_segment(); // get a segment based on the type, pointing

    inner_structure calc_rand_ut_vt_alpha(std::vector<double> u, std::vector<double> v, double alpha_pre, double Rf); // calculate ut and vt based on u and v and segment_type and flip rate

    void generate_polymer();
    // reset polymer
    void reset_polymer();
    //void reset_observable();

    // some observable measurement
    observable measure_observable(int bin_num);

    // Gyration tensor measurement
    //std::vector<double> Gij_m(); // no need for now
    // pair distribution function
    std::vector<double> calc_pair_dsitribution_function(double r_i, double r_f, int bin_num);
    std::vector<double> calc_structure_factor(double qB_i, double qB_f, int bin_num); // structure factor of the polymer, orientational averaged: ref: Pedersen 1996 equ 1
    double calc_radius_of_gyration(); // radius of gyration


    // experiment
    void save_polymer_to_file(std::string filename);
    void generate_and_save_polymer_ensemble(int number_of_polymer, int bin_num, std::string filename, bool save_detail = false);
    void save_observable_to_file(std::string filename, std::vector<observable> obs_ensemble, bool save_detail = false);
    void save_distribution_function_to_file(std::string filename, std::string component, std::vector<observable> obs_ensemble,  bool save_detail = false); //component can be "gr" or "Sq"

   // little tool
   std::vector<double> cross_product(std::vector<double> a, std::vector<double> b); // cross product of a and b
   double inner_product(std::vector<double> a, std::vector<double> b); // inner product of a and b
   std::vector<double> Rodrigues_rotation(std::vector<double> v, std::vector<double> k, double theta); // v rotate around k by theta
   std::vector<double> calc_rod_structure_factor(std::vector<double> qB); // calculate the structure factor of a rod

};
#endif