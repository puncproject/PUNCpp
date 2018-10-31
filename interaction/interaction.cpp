#include <punc.h>
#include <dolfin.h>
#include <boost/program_options.hpp>
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

/**
 * @brief Safely get multiple options
 * @param   key     Key to the option
 * @param   len     Length the resulting vector should have
 * @param   def     Default value in case no vector is present
 */
template <typename T>
vector<T> get_vector(po::variables_map options, string key, size_t len, T def){
    vector<T> res;
    if(options.count(key)){
        res = options[key].as<vector<T>>();
        if(res.size() != len){
            cerr << "Wrong number of " << key << " specified" << endl;
            exit(1);
        }
    } else {
        res = vector<T>(len, def);
    }
    return res;
}

template <size_t dim>
int run(const po::variables_map &options)
{
    PhysicalConstants constants;
    double eps0 = constants.eps0;

    //
    // SETUP MESH AND FIELDS
    //
    cout << "Setup mesh and fields" << endl;

    Mesh mesh(options["mesh"].as<string>());

    auto V = CG1_space(mesh);
    auto W = CG1_vector_space(mesh);
    auto Q = DG0_space(mesh);
    auto dv_inv = element_volume(V);

    // The electric potential and electric field
    df::Function phi(std::make_shared<const df::FunctionSpace>(V));
    df::Function E(std::make_shared<const df::FunctionSpace>(W));

    // Electron and ion number densities
    df::Function ne(std::make_shared<const df::FunctionSpace>(V));
    df::Function ni(std::make_shared<const df::FunctionSpace>(V));

    // Exponential moving average of number densities
    df::Function ne_ema(std::make_shared<const df::FunctionSpace>(V));
    df::Function ni_ema(std::make_shared<const df::FunctionSpace>(V));

    auto u0 = std::make_shared<df::Constant>(0.0);

    // mesh.ext_bnd_id will always be 1, but better not rely on it.
    // Perhaps we can use a function which returns this DirichletBC.
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0,
        std::make_shared<df::MeshFunction<size_t>>(mesh.bnd), mesh.ext_bnd_id);
    vector<df::DirichletBC> ext_bc = {bc};

    vector<double> B(dim, 0); // Magnetic field aligned with x-axis
    B[0] = options["Bx"].as<double>();
    double B_norm = accumulate(B.begin(), B.end(), 0.0);

    //
    // SETUP SPECIES
    //
    cout << "Setup species" << endl;

    vector<double> charge  = options["species.charge"].as<vector<double>>();
    vector<double> mass    = options["species.mass"].as<vector<double>>();
    vector<double> thermal = options["species.thermal"].as<vector<double>>();
    vector<double> density = options["species.density"].as<vector<double>>();

    size_t nSpecies = charge.size();
    if(mass.size()    != nSpecies
    || density.size() != nSpecies
    || thermal.size() != nSpecies){
        
        cerr << "Inconsistent number of species specified. "
             << "Check species.charge, species.mass, species.density and species.thermal" << endl;
        return 1;
    }

    vector<string> distribution = get_vector<string>(options, "species.distribution", nSpecies, "maxwellian");
    vector<int> npc             = get_vector<int>(options, "species.npc", nSpecies, 0);
    cout << "before" << endl;
    vector<int> num             = get_vector<int>(options, "species.num", nSpecies, 0);
    cout << "after" << endl;
    vector<double> kappa        = get_vector<double>(options, "species.kappa", nSpecies, 0);
    vector<double> alpha        = get_vector<double>(options, "species.alpha", nSpecies, 0);
    vector<double> vx           = get_vector<double>(options, "species.vx", nSpecies, 0);

    for(size_t s=0; s<nSpecies; s++){
        charge[s] *= constants.e;
        mass[s] *= constants.m_e;
    }

    // FIXME: Move to input file
    vector<vector<double>> vd(charge.size(), vector<double>(dim));

    vector<std::shared_ptr<Pdf>> pdfs;
    vector<std::shared_ptr<Pdf>> vdfs;

    CreateSpecies create_species(mesh);
    for(size_t s=0; s<charge.size(); s++){

        vd[s][0] = vx[s]; // fill in x-component of velocity vector for each species
        pdfs.push_back(std::make_shared<UniformPosition>(mesh.mesh));

        if(distribution[s]=="maxwellian"){
            vdfs.push_back(std::make_shared<Maxwellian>(thermal[s], vd[s]));
        } else if (distribution[s]=="kappa") {
            vdfs.push_back(std::make_shared<Kappa>(thermal[s], vd[s], kappa[s]));
        }else if (distribution[s] == "cairns"){
            vdfs.push_back(std::make_shared<Cairns>(thermal[s], vd[s], alpha[s]));
        }else if (distribution[s] == "kappa-cairns"){
            vdfs.push_back(std::make_shared<KappaCairns>(thermal[s], vd[s], kappa[s], alpha[s]));
        } else {
            cout << "Unsupported velocity distribution: ";
            cout << distribution[s] << endl;
            return 1;
        }

        create_species.create_raw(charge[s], mass[s], density[s],
                *(pdfs[s]), *(vdfs[s]), npc[s], num[s]);
    }
    auto species = create_species.species;

    create_flux(species, mesh.exterior_facets);

    //
    // TIME CONTROL
    //
    cout << "Setup time" << endl;

    double dt = 0;
    if(options.count("dt")){
        dt = options["dt"].as<double>();
    } else {
        double dtwp = options["dtwp"].as<double>();
        double wp0 = sqrt(pow(charge[0],2)*density[0]/(eps0*mass[0]));
        dt = dtwp/wp0;
    }

    size_t steps = options["steps"].as<size_t>();
    double densities_tau = options["diagnostics.densities_tau"].as<double>();
    size_t n_fields = options["diagnostics.n_fields"].as<size_t>();
    bool fields_end = options["diagnostics.fields_end"].as<bool>();
    bool state_end = options["diagnostics.state_end"].as<bool>();
    bool PE_save = options["diagnostics.PE_save"].as<bool>();

    //
    // IMPOSE CIRCUITRY
    //
    cout << "Setup circuitry" << endl;
    
    double imposed_current = options["imposed_current"].as<double>();
    double imposed_voltage = options["imposed_voltage"].as<double>();
    bool impose_current = options["impose_current"].as<bool>();

    vector<Source> isources;
    vector<Source> vsources;

    if(impose_current)
    {
        isources.push_back(Source{-1,0,-imposed_current});

    } else {
        vsources.push_back(Source{-1, 0, imposed_voltage});
    }


    vector<std::shared_ptr<Object>> objects;
    std::shared_ptr<Circuit> circuit;
    string object_method = options["objects.method"].as<string>();
    if (object_method == "BC")
    {
        objects.push_back(std::make_shared<ObjectBC>(V, mesh, 2, eps0));
        circuit = std::make_shared<CircuitBC>(V, objects, vsources, isources, dt, eps0);
    } else if (object_method == "CM")
    {
        objects.push_back(std::make_shared<ObjectCM>(V, mesh, 2));
        circuit = std::make_shared<CircuitCM>(V, objects, vsources, isources, mesh, dt, eps0);
    }

    //
    // CREATE SOLVERS
    //
    string linalg_method         = options["linalg.method"].as<string>();
    string linalg_preconditioner = options["linalg.preconditioner"].as<string>();
    
    PoissonSolver poisson(V, ext_bc, circuit, eps0, false,
                          linalg_method, linalg_preconditioner);

    poisson.set_abstol(options["linalg.abstol"].as<double>());
    poisson.set_reltol(options["linalg.reltol"].as<double>());

    ESolver esolver(W);

    //
    // LOAD NEW PARTICLES OR CONTINUE SIMULATION FROM FILE
    //
    cout << "Loading particles" << endl;
    Population<dim> pop(mesh);

    size_t n = 0;
    double t = 0;
    bool continue_simulation = false;

    ifstream ifile_state(fname_state);
    ifstream ifile_hist(fname_hist);
    ifstream ifile_pop(fname_pop);

    if(ifile_state.good() && ifile_hist.good() && ifile_pop.good()){
        continue_simulation = true;
    }

    ifile_hist.close();
    ifile_pop.close();

    //
    // HISTORY AND SATE FILES
    //
    History hist(fname_hist, objects, dim, continue_simulation);
    State state(fname_state);
    FieldWriter fields("Fields/phi.pvd", "Fields/E.pvd", "Fields/rho.pvd", "Fields/ne.pvd", "Fields/ni.pvd");

    bool binary = options["diagnostics.binary"].as<bool>();

    if(continue_simulation){
        cout << "Continuing previous simulation" << endl;
        state.load(n, t, objects);
        pop.load_file(fname_pop, binary);
    } else {
        cout << "Starting new simulation" << endl;
        load_particles(pop, species);
    }

    //
    // CREATE HISTORY VARIABLES
    //
    double KE               = 0;
    double PE               = 0;
    double num_e            = pop.num_of_negatives();
    double num_i            = pop.num_of_positives();
    double num_tot          = pop.num_of_particles();

    cout << "imposed_current: " << imposed_current <<'\n';
    cout << "imposed_voltage: " << imposed_voltage << '\n';

    cout << "Num positives:  " << num_i;
    cout << ", num negatives: " << num_e;
    cout << " total: " << num_tot << endl;

    //
    // CREATE TIMER TASKS
    //
    vector<string> tasks{"distributor", "poisson", "efield", "update", "PE", "accelerator", "move", "injector", "counting particles", "io", "density"};
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
        auto rho = distribute_cg1(V, pop, dv_inv);
        timer.toc();

        // SOLVE POISSON EQUATION WITH OBJECTS
        timer.tic("poisson");
        poisson.solve_circuit(phi, rho, mesh, objects, circuit);
        timer.toc();

        // ELECTRIC FIELD
        timer.tic("efield");
        esolver.solve(E, phi);
        timer.toc();

        // compute_object_potentials(objects, E, inv_capacity, mesh.mesh);
        // auto phi1 = poisson.solve(rho, objects);
        // auto E1 = esolver.solve(phi1);

        // POTENTIAL ENERGY
        timer.tic("PE");
        if (PE_save)
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
        if (fabs(densities_tau)<tol)
        {
            density_cg1(V, pop, ne, ni, dv_inv);
            ema(ne, ne_ema, dt, densities_tau);
            ema(ni, ni_ema, dt, densities_tau);
        }
        // SAVE FIELDS
        if(n_fields !=0 && n%n_fields == 0)
        {
            if (fabs(densities_tau)<tol)
            {
                fields.save(phi, E, rho, ne_ema, ni_ema, t);
            }else{
                density_cg1(V, pop, ne, ni, dv_inv);
                fields.save(phi, E, rho, ne, ni, t);
            }
        } 
        // SAVE STATE AND BREAK LOOP
        if(exit_immediately || n==steps)
        {
            // SAVE FIELDS
            if (fields_end)
            {
                if (fabs(densities_tau))
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
            if (state_end)
            {
                pop.save_file(fname_pop, binary);
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

int main(int argc, char **argv){

    signal(SIGINT, signal_handler);
    df::set_log_level(df::WARNING);

    po::options_description desc("Options");
    desc.add_options()
        ("help"  , "show help (this)")
        ("input" , po::value<string>() , "input file (.ini)")
        ("mesh"  , po::value<string>() , "mesh file (.xml or .hdf5)")
        ("steps" , po::value<size_t>() , "number of timesteps")
        ("dt"    , po::value<double>() , "timestep [s] (overrides dtwp)")
        ("dtwp"  , po::value<double>() , "timestep [1/w_p of first specie]")

        ("Bx", po::value<double>()->default_value(0), "magnetic field [T]")

        ("impose_current"  , po::value<bool>()   , "Whether to impose current or voltage (true|false)")
        ("imposed_current" , po::value<double>() , "Current imposed on object [A]")
        ("imposed_voltage" , po::value<double>() , "Voltage imposed on object [V]")

        ("species.charge"       , po::value<vector<double>>() , "charge [elementary chages]")
        ("species.mass"         , po::value<vector<double>>() , "mass [electron masses]")
        ("species.density"      , po::value<vector<double>>() , "number density [1/m^3]")
        ("species.thermal"      , po::value<vector<double>>() , "thermal speed [m/s]")
        ("species.vx"           , po::value<vector<double>>() , "drift velocity [m/s]")
        ("species.alpha"        , po::value<vector<double>>() , "spectral index alpha")
        ("species.kappa"        , po::value<vector<double>>() , "spectral index kappa")
        ("species.npc"          , po::value<vector<int>>()    , "number of particles per cell")
        ("species.num"          , po::value<vector<int>>()    , "number of particles in total (overrides npc)")
        ("species.distribution" , po::value<vector<string>>() , "distribution (maxwellian|kappa|cairns|kappa-cairns)")

        ("objects.method" , po::value<string>()->default_value("BC") , "Object method (BC|CM)")
        ("objects.charge" , po::value<vector<double>>()              , "Initial object charge")

        ("diagnostics.n_fields"      , po::value<size_t>()                    , "write fields to file every nth time-step")
        ("diagnostics.densities_tau" , po::value<double>()                    , "exponential moving average relaxation time (disable with 0)")
        ("diagnostics.fields_end"    , po::value<bool>()                      , "write to file every nth time-step")
        ("diagnostics.state_end"     , po::value<bool>()                      , "write to file every nth time-step")
        ("diagnostics.PE_save"       , po::value<bool>()                      , "calculate and save potential energy")
        ("diagnostics.binary"        , po::value<bool>()->default_value(true) , "write binary population files (true|false)")

        ("linalg.method"         , po::value<string>()->default_value("")    , "Linear algebra solver")
        ("linalg.preconditioner" , po::value<string>()->default_value("")    , "Linear algebra preconditioner")
        ("linalg.abstol"         , po::value<double>()->default_value(1e-14) , "Absolute residual tolerance")
        ("linalg.reltol"         , po::value<double>()->default_value(1e-12) , "Relative residual tolerance")
    ;

    // Setting config file as positional argument
    po::positional_options_description pos_options;
    pos_options.add("input", -1);

    //
    // PARSING INPUT
    //

    // Parse input from command line first, including name of config file.
    // These settings takes precedence over input from config file.
    po::variables_map options;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(pos_options).run(), options);
    po::notify(options);

    if(options.count("help")){
        cout << desc << endl;
        return 1;
    } else if(!options.count("input")){
        cerr << "Input file missing. See interaction --help." << endl;
        return 1;
    }

    string fname_ifile = options["input"].as<string>();
    ifstream ifile;
    ifile.open(fname_ifile);
    po::store(po::parse_config_file(ifile, desc), options);
    po::notify(options);
    ifile.close();

    cout << "PUNC++ started!" << endl;

    Mesh mesh(options["mesh"].as<string>());

         if(mesh.dim==1) return run<1>(options);
    else if(mesh.dim==2) return run<2>(options);
    else if(mesh.dim==3) return run<3>(options);
    else {
        cerr << "Only 1D, 2D and 3D supported" << endl;
        return 1;
    }
}
