// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++.
//
// PUNC++ is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC++ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC++. If not, see <http://www.gnu.org/licenses/>.

/**
 * @file		interaction.cpp
 * @brief		Main PUNC executable
 */
#include "io.h"
#include <punc.h>
#include <dolfin.h>
#include <csignal>

using namespace punc;
namespace po = boost::program_options;

using std::cout;
using std::cerr;
using std::endl;
using std::size_t;
using std::vector;
using std::accumulate;
using std::string;
using std::ifstream;
using std::ofstream;
using std::make_shared;

const char* fname_hist  = "history.dat";
const char* fname_state = "state.dat";
const char* fname_pop   = "population.dat";
const bool override_status_print = true;
const double tol = 1e-10;

bool exit_immediately = true;
void signal_handler(int signum){
    if(exit_immediately){
        exit(signum);
    } else {
        cout << endl;
        cout << "Completing and storing timestep before exiting. ";
        cout << "Press Ctrl+C again to force quit." << endl;
        exit_immediately = true;
    }
}

template <size_t dim>
int run(const Options &opt)
{
    cout << "Starting " << dim << "D PIC simulation" << endl;

    PhysicalConstants constants;
    double eps0 = constants.eps0;

    /***************************************************************************
     * SETUP MESH AND FIELDS
     **************************************************************************/
    cout << "Setup mesh and fields" << endl;

    string mesh_fname;
    opt.get("mesh", mesh_fname);
    Mesh mesh(mesh_fname);

    auto V = CG1_space(mesh);
    auto W = CG1_vector_space(mesh);
    auto Q = DG0_space(mesh);
    auto P = DG0_vector_space(mesh);
    auto dv_inv = element_volume(V);

    // The electric potential and electric field
    df::Function phi(std::make_shared<const df::FunctionSpace>(V));
    df::Function rho(std::make_shared<const df::FunctionSpace>(V));
    df::Function E(std::make_shared<const df::FunctionSpace>(W));

    // Electron and ion number densities
    df::Function ne(std::make_shared<const df::FunctionSpace>(V));
    df::Function ni(std::make_shared<const df::FunctionSpace>(V));

    // Exponential moving average of number densities
    df::Function ne_ema(std::make_shared<const df::FunctionSpace>(V));
    df::Function ni_ema(std::make_shared<const df::FunctionSpace>(V));

    vector<double> B(dim, 0);
    opt.get_vector("B", B, dim, true);
    double B_norm = accumulate(B.begin(), B.end(), 0.0);

    /***************************************************************************
     * SETUP SPECIES
     **************************************************************************/
    cout << "Setup species" << endl;
    auto species = read_species(opt, mesh);
    create_flux(species, mesh.exterior_facets);

    /***************************************************************************
     * SETUP TIME-STEP
     **************************************************************************/
    cout << "Setup time-step" << endl;
    double dt = read_timestep(opt, mesh, species, B, eps0);

    /***************************************************************************
     * SETUP CIRCUITRY
     **************************************************************************/
    cout << "Setup circuitry" << endl;

    ISourceVector isources;
    VSourceVector vsources;

    vector<vector<double>> vsources_, isources_;
    opt.get_repeated_vector("objects.vsource", vsources_, 3, 0, true);
    opt.get_repeated_vector("objects.isource", isources_, 3, 0, true);
    
    for(auto &s : vsources_){
        VSource source(s[0], s[1], s[2]);
        vsources.push_back(source);
    }

    for(auto &s : isources_){
        ISource source(s[0], s[1], s[2]);
        isources.push_back(source);
    }

    for(auto &s : vsources) cout << "  " << s << endl;
    for(auto &s : isources) cout << "  " << s << endl;

    vector<std::shared_ptr<Object>> objects;
    std::shared_ptr<Circuit> circuit;

    string object_method = "BC";
    opt.get("objects.method", object_method, true);

    vector<double> object_charges(mesh.num_objects, 0);
    opt.get_repeated("objects.charge", object_charges, mesh.num_objects, true);

    if (object_method == "BC") {
        for(size_t i=0; i<mesh.num_objects; i++){
            objects.push_back(std::make_shared<ObjectBC>(V, mesh, i+2, eps0));
            objects[i]->charge = object_charges[i];
        }
        circuit = std::make_shared<CircuitBC>(V, objects, vsources, isources, dt, eps0);

    } else if (object_method == "CM") {
        for(size_t i=0; i<mesh.num_objects; i++){
            objects.push_back(std::make_shared<ObjectCM>(V, mesh, i+2));
            objects[i]->charge = object_charges[i];
        }
        circuit = std::make_shared<CircuitCM>(V, objects, vsources, isources, mesh, dt, eps0);
    }

    /***************************************************************************
     * SETUP DIAGNOSTICS
     **************************************************************************/
    cout << "Setup diagnostics" << endl;

    double relaxation_time = 0;
    opt.get("diagnostics.relaxation_time", relaxation_time, true);

    size_t save_fields_n = 0;
    opt.get("diagnostics.save_fields_n", save_fields_n, true);

    bool save_fields_end = true;
    opt.get("diagnostics.save_fields_end", save_fields_end, true);

    bool save_state_end = true;
    opt.get("diagnostics.save_state_end", save_state_end, true);

    bool compute_potential_energy = false;
    opt.get("diagnostics.compute_potential_energy", compute_potential_energy, true);

    bool binary_population = true;
    opt.get("diagnostics.binary_population", binary_population, true);

    ifstream ifile_state(fname_state);
    ifstream ifile_hist(fname_hist);
    ifstream ifile_pop(fname_pop);

    bool continue_simulation = false;
    if(ifile_state.good() && ifile_hist.good() && ifile_pop.good()){
        continue_simulation = true;
    }

    ifile_state.close();
    ifile_hist.close();
    ifile_pop.close();

    History hist(fname_hist, objects, dim, continue_simulation);
    State state(fname_state);
    FieldWriter fields("Fields/phi.pvd", "Fields/E.pvd", "Fields/rho.pvd", "Fields/ne.pvd", "Fields/ni.pvd");

    /***************************************************************************
     * SETUP PARTICLES
     **************************************************************************/
    cout << "Setup particles" << endl;

    Population<dim> pop(mesh);

    size_t n = 0;
    double t = 0;

    bool prefill = true;
    opt.get("prefill", prefill, true);

    if(continue_simulation){
        cout << "  Continuing previous simulation" << endl;
        state.load(n, t, objects);
        pop.load_file(fname_pop, binary_population);
    } else {
        cout << "  Starting new simulation" << endl;
        if(prefill){
            cout << "  Initializing particles" << endl;
            load_particles(pop, species);
        }
    }


    /***************************************************************************
     * SETUP SOLVERS
     **************************************************************************/
    cout << "Setup solvers" << endl;

    string linalg_method, linalg_preconditioner;
    opt.get("poisson.method", linalg_method, true);
    opt.get("poisson.preconditioner", linalg_preconditioner, true);

    double linalg_abstol = 1e-14;
    double linalg_reltol = 1e-12;
    opt.get("poisson.abstol", linalg_abstol, true);
    opt.get("poisson.reltol", linalg_reltol, true);
    
    auto ext_bc = exterior_bc(V, mesh, species[0].vdf->vd(), B);

    PoissonSolver poisson(V, objects, ext_bc, circuit, eps0, false,
                          linalg_method, linalg_preconditioner);

    poisson.set_abstol(linalg_abstol);
    poisson.set_reltol(linalg_reltol);

    ESolver esolver(W);
    // EFieldMean esolver(P, W);

    /***************************************************************************
     * SETUP TIME LOOP CONTROL
     **************************************************************************/
    cout << "Setup time loop control" << endl;

    size_t steps;
    double stop;
    string stop_suffix;
    opt.get("time.stop", stop, {"s", "steps", ""}, stop_suffix);
    if(stop_suffix=="s"){
        steps = (stop-t)/dt + n;
    } else {
        steps = stop;
    }


    double KE               = 0;
    double PE               = 0;
    double num_e            = pop.num_of_negatives();
    double num_i            = pop.num_of_positives();
    double num_tot          = pop.num_of_particles();

    cout << "  Num positives:  " << num_i;
    cout << ", num negatives: " << num_e;
    cout << " total: " << num_tot << endl;

    vector<string> tasks{"distributor",
                         "poisson",
                         "efield",
                         "update",
                         "PE",
                         "accelerator",
                         "move",
                         "injector",
                         "counting particles",
                         "io",
                         "density"};

    Timer timer(tasks);

    exit_immediately = false;
    auto n_previous = n; // n from previous simulation

    for(; n<=steps; ++n){

        // We are now at timestep n
        // Velocities and currents are at timestep n-0.5 (or 0 if n==0)

        // Print status
        timer.progress(n, steps, n_previous, override_status_print);

        // DISTRIBUTE
        timer.tic("distributor");
        distribute_cg1(pop, rho, dv_inv);
        timer.toc();

        // SOLVE POISSON EQUATION WITH OBJECTS
        timer.tic("poisson");
        poisson.solve_circuit(phi, rho, mesh, objects, circuit);
        timer.toc();

        // ELECTRIC FIELD
        timer.tic("efield");
        esolver.solve(E, phi);
        // esolver.mean(E, phi);
        timer.toc();

        // POTENTIAL ENERGY
        timer.tic("PE");
        if (compute_potential_energy)
        {
            PE = particle_potential_energy_cg1(pop, phi);
        }
        timer.toc();

        // COUNT PARTICLES
        timer.tic("counting particles");
        num_e     = pop.num_of_negatives();
        num_i     = pop.num_of_positives();
        num_tot   = pop.num_of_particles();
        timer.toc();

        // PUSH PARTICLES AND CALCULATE THE KINETIC ENERGY
        // Advancing velocities to n+0.5
        timer.tic("accelerator");
        if (fabs(B_norm)<tol)
        {
            KE = accel_cg1(pop, E, (1.0 - 0.5 * (n == 0)) * dt);
        }else{
            KE = boris_cg1(pop, E, B, (1.0 - 0.5 * (n == 0)) * dt);
        }
        if(n==0) KE = kinetic_energy(pop);
        timer.toc();

        // WRITE HISTORY
        // Everything at n, except currents which are at n-0.5.
        timer.tic("io");
        hist.save(n, t, num_e, num_i, KE, PE, objects);
        timer.toc();

        // MOVE PARTICLES
        // Advancing position to n+1
        timer.tic("move");
        move(pop, dt);
        timer.toc();

        t += dt;

        // UPDATE PARTICLE POSITIONS
        timer.tic("update");
        pop.update(objects, dt);
        timer.toc();

        // INJECT PARTICLES
        timer.tic("injector");
        inject_particles(pop, species, mesh.exterior_facets, dt);
        timer.toc();

        // AVERAGING
        timer.tic("io");
        if (fabs(relaxation_time)>tol)
        {
            density_cg1(V, pop, ne, ni, dv_inv);
            ema(ne, ne_ema, dt, relaxation_time);
            ema(ni, ni_ema, dt, relaxation_time);
            if(save_fields_n !=0 && n%save_fields_n == 0){
                fields.save(phi, E, rho, ne_ema, ni_ema, t);
            }
        }
        else if (save_fields_n !=0 && n%save_fields_n == 0)
        {
            density_cg1(V, pop, ne, ni, dv_inv);
            fields.save(phi, E, rho, ne, ni, t);
        } 
        // SAVE STATE AND BREAK LOOP
        if(exit_immediately || n==steps)
        {
            // SAVE FIELDS
            if (save_fields_end)
            {
                if (fabs(relaxation_time))
                {
                    fields.save(phi, E, rho, ne_ema, ni_ema, t);
                }
                else
                {
                    density_cg1(V, pop, ne, ni, dv_inv);
                    fields.save(phi, E, rho, ne, ni, t);
                }
            }
            // SAVE POPULATION AND STATE
            if (save_state_end)
            {
                pop.save_file(fname_pop, binary_population);
                state.save(n, t, objects);
            }
            break;
        }
        timer.toc();

        if (exit_immediately)
        {
            timer.summary();
        }
    }
    if(override_status_print) cout << endl;

    // Print a summary of tasks
    timer.summary();
    cout << "PUNC++ finished successfully!" << endl;
    return 0;
}

po::typed_value<vector<string>> *value(){
    return po::value<vector<string>>();
}

int main(int argc, char **argv){

    signal(SIGINT, signal_handler);
    df::set_log_level(df::WARNING);
    
    // IMPORTANT:
    // It is important that we maintain a complete and unambiguous description
    // of the arguments here. By doing so, we do not have to fully document it
    // in the user guide, but can refer to this list for up-to-date information.

    po::options_description desc("Options");
    desc.add_options()
        ("help"   , "show help (this)")
        ("input"  , value(), "Input file (.ini)")
        ("mesh"   , value(), "Mesh file (.xml or .hdf5)")
        ("B"      , value(), "Magnetic field [T] (default: zero)")
        ("prefill", value(), "Whether to initialize new simulation by prefilling the domain uniformly with particles. Options: true (default), false")

        ("time.stop"     , value(), "When to stop simulation. Suffixes:\n"
                                    "  s - seconds\n"
                                    "  steps - timesteps (default)\n")
        ("time.dt"       , value(), "Timestep. Suffixes:\n"
                                    "  s - seconds (default)\n"
                                    "  auto - fraction of automatically derived timestep")
        ("time.dt_plasma", value(), "Max timestep in terms of plasma periods")
        ("time.dt_gyro"  , value(), "Max timestep in terms of gyroperiods")
        ("time.dt_cell"  , value(), "Max cells traveled per time-step")
        ("time.sigma"    , value(), "Max fraction of thermal speed to account for when computing timestep based on particle speed")
        ("time.phi_max"  , value(), "Max potential expected in the domain (for computing timestep)")
        ("time.phi_min"  , value(), "Min potential expected in the domain (for computing timestep)")

        ("species.charge"       , value(), "Charge. Suffixes:\n"
                                         "  C - coulomb (default)\n"
                                         "  e - elementary charges")
        ("species.mass"         , value(), "Mass. Suffixes:\n"
                                         "  kg  - kilogram (default)\n"
                                         "  me  - electron masses\n"
                                         "  amu - atomic mass units")
        ("species.density"      , value(), "Number density [1/m^3]")
        ("species.temperature"  , value(), "Temperature. Suffixes:\n"
                                         "  K   - kelvin (default)\n"
                                         "  eV  - electron volts\n"
                                         "  m/s - thermal speed in m/s")
        ("species.vdrift"       , value(), "Drift velocity (default: zero) [m/s]")
        ("species.alpha"        , value(), "Spectral index alpha (default: 0)")
        ("species.kappa"        , value(), "Spectral index kappa (default: 4)")
        ("species.amount"       , value(), "Number of simulation particles (of this species). Suffixes:\n"
                                         "  in total - In total (default)\n"
                                         "  per cell - Particles per cell\n"
                                         "  per volume - Particles per volume\n"
                                         "  phys per sim - Physical particles per simulation particle (statistical weight)")
        ("species.distribution" , value(), "Distribution. Options:\n"
                                         "  maxwellian   - Maxwellian (default)\n"
                                         "  kappa        - Kappa\n"
                                         "  cairns       - Cairns\n"
                                         "  kappa-cairns - Kappa-Cairns")

        ("objects.method" , value(), "Object method. Options:\n"
                                   "  BC - Method described in PUNC++ paper\n"
                                   "  CM - Method described in PTetra paper")
        ("objects.charge" , value(), "Initial object charge (one per object, in ascending order, default: zero) [C]")
        ("objects.vsource", value(), "Voltage source between objects a and b. Syntax: object_a object_b value [V]")
        ("objects.isource", value(), "Current source between objects a and b. Syntax: object_a object_b value [I]")

        ("diagnostics.save_fields_n"           , value(), "Write fields to file every nth time-step. Disable with 0 (default)")
        ("diagnostics.save_fields_end"         , value(), "Write fields to file after simulation. Options: true, false (default)")
        ("diagnostics.save_state_end"          , value(), "Write population and state to file after simulation. Options: true (default), false")
        ("diagnostics.compute_potential_energy", value(), "Calculate potential energy. Options: true, false (default)")
        ("diagnostics.relaxation_time"         , value(), "Exponential moving average relaxation time for number densities [s]. Disable with 0 (default)")
        ("diagnostics.binary_population"       , value(), "Write population files in binary format. Options: true (default), false")

        ("poisson.method"        , value() , "Linear algebra solver. See FEniCS for options. Default depends on object method.")
        ("poisson.preconditioner", value() , "Linear algebra preconditioner. See FEniCS for options. Default depends on object method.")
        ("poisson.abstol"        , value() , "Absolute residual tolerance. Default: 1e-14")
        ("poisson.reltol"        , value() , "Relative residual tolerance. Default: 1e-12")
    ;

    // Setting config file as positional argument
    po::positional_options_description pos_options;
    pos_options.add("input", -1);

    //
    // PARSING INPUT
    //

    // Parse input from command line first, including name of config file.
    // These settings takes precedence over input from config file.
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(pos_options).run(), vm);
    po::notify(vm);

    if(vm.count("help")){
        cout << desc << endl;
        return 1;
    } else if(!vm.count("input")){
        cerr << "Input file missing. See interaction --help" << endl;
        return 1;
    }

    Options opt(vm);

    string fname_ifile;
    opt.get("input", fname_ifile);

    ifstream ifile;
    ifile.open(fname_ifile);
    po::store(po::parse_config_file(ifile, desc), vm);
    po::notify(vm);
    ifile.close();
    opt = Options(vm);

    cout << "PUNC++ started!" << endl;

    string mesh_fname;
    opt.get("mesh", mesh_fname);
    Mesh mesh(mesh_fname);

         if(mesh.dim==1) return run<1>(opt);
    else if(mesh.dim==2) return run<2>(opt);
    else if(mesh.dim==3) return run<3>(opt);
    else {
        cerr << "Only 1D, 2D and 3D supported" << endl;
        return 1;
    }
}
