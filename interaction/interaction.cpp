#include <punc.h>
#include <dolfin.h>
#include <boost/program_options.hpp>
#include <csignal>

using namespace punc;
namespace po = boost::program_options;

using std::cout;
using std::endl;
using std::size_t;
using std::vector;
using std::accumulate;
using std::string;
using std::ifstream;
using std::ofstream;

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

// Horrible, horrible list of arguments. Hopefully only temporary.
template <size_t dim>
int run(
        std::shared_ptr<const df::Mesh> mesh,
        df::MeshFunction<std::size_t> boundaries,
        size_t steps,
        double dt,
        double Bx,
        bool impose_current,
        double imposed_current,
        double imposed_voltage,
        vector<int> npc,
        vector<int> num,
        vector<double> density,
        vector<double> thermal,
        vector<double> vx,
        vector<double> charge,
        vector<double> mass,
        vector<double> kappa,
        vector<double> alpha,
        vector<string> distribution,
        bool binary
){

    PhysicalConstants constants;
    double eps0 = constants.eps0;

    auto tags = get_mesh_ids(boundaries);
    size_t ext_bnd_id = tags[1];

    auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);

    vector<double> Ld = get_mesh_size(mesh);

    vector<double> B(dim, 0); // Magnetic field aligned with x-axis
    B[0] = Bx;

    double B_norm = accumulate(B.begin(), B.end(), 0.0);

    // Relaxation time:
    // double tau = 100*dt;
    //
    // CREATE SPECIES
    //

    // FIXME: Move to input file
    vector<vector<double>> vd(charge.size(), vector<double>(dim));

    vector<std::shared_ptr<Pdf>> pdfs;
    vector<std::shared_ptr<Pdf>> vdfs;

    CreateSpecies create_species(mesh);
    for(size_t s=0; s<charge.size(); s++){

        vd[s][0] = vx[s]; // fill in x-component of velocity vector for each species
        pdfs.push_back(std::make_shared<UniformPosition>(mesh));

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

    //
    // IMPOSE CIRCUITRY
    //
    vector<vector<int>> isources, vsources;
    vector<double> ivalues, vvalues;

    if(impose_current){

        isources = {{-1,0}};
        ivalues = {-imposed_current};

        vsources = {};
        vvalues = {};

    } else {

        isources = {};
        ivalues = {};

        vsources = {{-1,0}};
        vvalues = {imposed_voltage};
    }

    //
    // CREATE FUNCTION SPACES AND BOUNDARY CONDITIONS
    //
    auto V = function_space(mesh);
    auto Q = DG0_space(mesh);
    auto dv_inv = element_volume(V);

    // Electron and ion number densities
    df::Function ne(std::make_shared<const df::FunctionSpace>(V));
    df::Function ni(std::make_shared<const df::FunctionSpace>(V));

    auto u0 = std::make_shared<df::Constant>(0.0);
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0,
        std::make_shared<df::MeshFunction<size_t>>(boundaries), ext_bnd_id);
    vector<df::DirichletBC> ext_bc = {bc};

    ObjectBC object(V, boundaries, tags[2], eps0);
    vector<ObjectBC> int_bc = {object};

    vector<size_t> bnd_id{tags[2]};
    Circuit circuit(V, int_bc, isources, ivalues, vsources, vvalues, dt, eps0);

    //
    // CREATE SOLVERS
    //
    PoissonSolver poisson(V, ext_bc, circuit, eps0);
    ESolver esolver(V);

    //
    // CREATE FLUX
    //
    cout << "Create flux" << endl;
    create_flux(species, facet_vec);

    //
    // LOAD NEW PARTICLES OR CONTINUE SIMULATION FROM FILE
    //
    cout << "Loading particles" << endl;

    Population<dim> pop(mesh, boundaries);

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

    ofstream file_hist;

    if(continue_simulation){
        cout << "Continuing previous simulation" << endl;

        string line;
        std::getline(ifile_state, line);

        char *s = (char *)line.c_str();
        n                           = strtol(s, &s, 10);
        t                           = strtod(s, &s);
        int_bc[0].charge            = strtod(s, &s);
        int_bc[0].collected_current = strtod(s, &s);

        pop.load_file(fname_pop, binary);
        file_hist.open(fname_hist, ofstream::out | ofstream::app);

    } else {
        cout << "Starting new simulation" << endl;

        load_particles(pop, species);
        file_hist.open(fname_hist, ofstream::out);

        // Preamble for metaplot
        file_hist << "#:xaxis\tt\n";
        file_hist << "#:name\tn\tt\tne\tni\tKE\tPE\tV\tI\tQ\n";
        file_hist << "#:long\ttimestep\ttime\t\"electron density\"\t";
        file_hist << "\"ion density\"\t\"kinetic energy\"\t";
        file_hist << "\"potential energy\"\tvoltage\tcurrent\tcharge\n";
        file_hist << "#:units\t1\ts\tm**(-3)\tm**(-3)\tJ\tJ\tV\tA\tC\n";
    }

    ifile_state.close();

    //
    // CREATE HISTORY VARIABLES
    //
    double KE               = 0;
    double PE               = 0;
    double num_e            = pop.num_of_negatives();
    double num_i            = pop.num_of_positives();
    double num_tot          = pop.num_of_particles();
    double potential        = 0;
    double current_measured = 0;
    double old_charge       = 0;

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

        // SOLVE POISSON
        // reset_objects(int_bc);
        // t_rsetobj[n]= timer.elapsed();
        /* int_bc[0].charge = 0; */
        /* int_bc[0].charge -= current_collected*dt; */
        timer.tic("poisson");
        auto phi = poisson.solve(rho, int_bc, circuit, V);
        timer.toc();


        for(auto& o: int_bc) o.update_charge(phi);

        potential = int_bc[0].update_potential(phi);
        
        // ELECTRIC FIELD
        timer.tic("efield");
        auto E = esolver.solve(phi);
        timer.toc();

        // compute_object_potentials(int_bc, E, inv_capacity, mesh);
        //
        // potential[n] = int_bc[0].potential;
        // auto phi1 = poisson.solve(rho, int_bc);
        //
        // auto E1 = esolver.solve(phi1);

        // POTENTIAL ENERGY
        timer.tic("PE");
        PE = particle_potential_energy_cg1(pop, phi);
        timer.toc();

        // COUNT PARTICLES
        timer.tic("counting particles");
        num_e     = pop.num_of_negatives();
        num_i     = pop.num_of_positives();
        num_tot   = pop.num_of_particles();
        timer.toc();
        /* cout << "ions: "<<num_i; */
        /* cout << "  electrons: " << num_e; */
        /* cout << "  total: " << num_tot << '\n'; */

        // PUSH PARTICLES AND CALCULATE THE KINETIC ENERGY
        // Advancing velocities to n+0.5
        old_charge = int_bc[0].charge;

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
        file_hist << n << "\t";
        file_hist << t << "\t";
        file_hist << num_e << "\t";
        file_hist << num_i << "\t";
        file_hist << KE << "\t";
        file_hist << PE << "\t";
        file_hist << potential << "\t";
        file_hist << current_measured << "\t";
        file_hist << old_charge*constants.e << endl;
        timer.toc();

        // MOVE PARTICLES
        // Advancing position to n+1
        timer.tic("move");
        move(pop, dt);
        timer.toc();

        t += dt;

        // UPDATE PARTICLE POSITIONS
        timer.tic("update");
        pop.update(int_bc);
        timer.toc();

        current_measured = ((int_bc[0].charge - old_charge) / dt);
        int_bc[0].collected_current = current_measured;

        // INJECT PARTICLES
        timer.tic("injector");
        inject_particles(pop, species, facet_vec, dt);
        timer.toc();

        // SAVE STATE AND BREAK LOOP
        if(exit_immediately || n==steps){
            
            df::File ofile("phi.pvd");
            ofile<<phi;

            density_cg1(V, pop, ne, ni, dv_inv);
            df::File ofile1("ne.pvd");
            ofile1 << ne;
            
            df::File ofile2("ni.pvd");
            ofile2 << ni;

            pop.save_file(fname_pop, binary);

            ofstream state_file;
            state_file.open(fname_state, ofstream::out);
            state_file << n+1 << "\t" << t << "\t";
            state_file << int_bc[0].charge << "\t";
            state_file << int_bc[0].collected_current << endl;
            state_file.close();

            break;
        }

        if (exit_immediately)
        {
            timer.summary();
        }
    }
    if(override_status_print) cout << endl;

    file_hist.close();

    // Prints a summary of tasks
    timer.summary();
    cout << "PUNC++ finished successfully!" << endl;
    return 0;
}

int main(int argc, char **argv){

    signal(SIGINT, signal_handler);
    df::set_log_level(df::WARNING);

    //
    // INPUT VARIABLES
    //

    // Global input
    string fname_ifile;
    string fname_mesh;
    size_t steps = 0;
    double dt = 0;
    double dtwp;
    double Bx = 0;
    bool binary = false;

    // Object input
    bool impose_current = true; 
    double imposed_current;
    double imposed_voltage;

    // Species input
    vector<int> npc;
    vector<int> num;
    vector<double> density;
    vector<double> thermal;
    vector<double> vx;
    vector<double> charge;
    vector<double> mass;
    vector<double> kappa;
    vector<double> alpha;
    vector<string> distribution;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "show help (this)")
        ("input", po::value(&fname_ifile), "config file")
        ("mesh", po::value(&fname_mesh), "mesh file")
        ("steps", po::value(&steps), "number of timesteps")
        ("dt", po::value(&dt), "timestep [s] (overrides dtwp)") 
        ("dtwp", po::value(&dtwp), "timestep [1/w_p of first specie]")
        ("binary", po::value(&binary), "Write binary population files (true|false)")
        
        ("Bx", po::value(&Bx), "magnetic field [T]")

        ("impose_current", po::value(&impose_current), "Whether to impose current or voltage (true|false)")
        ("imposed_current", po::value(&imposed_current), "Current imposed on object [A]")
        ("imposed_voltage", po::value(&imposed_voltage), "Voltage imposed on object [V]")

        ("species.charge", po::value(&charge), "charge [elementary chages]")
        ("species.mass", po::value(&mass), "mass [electron masses]")
        ("species.density", po::value(&density), "number density [1/m^3]")
        ("species.thermal", po::value(&thermal), "thermal speed [m/s]")
        ("species.vx", po::value(&vx), "drift velocity [m/s]")
        ("species.alpha", po::value(&alpha), "spectral index alpha")
        ("species.kappa", po::value(&kappa), "spectral index kappa")
        ("species.npc", po::value(&npc), "number of particles per cell")
        ("species.num", po::value(&num), "number of particles in total (overrides npc)")
        ("species.distribution", po::value(&distribution), "distribution (maxwellian)")
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

    // Parse input from config file.
    if(options.count("input")){
        ifstream ifile;
        ifile.open(fname_ifile);
        po::store(po::parse_config_file(ifile, desc), options);
        po::notify(options);
        ifile.close();
    }

    // Print help
    if(options.count("help")){
        cout << desc << endl;
        return 1;
    }

    cout << "PUNC++ started!" << endl;

    //
    // PRE-PROCESS INPUT
    //
    PhysicalConstants constants;
    double eps0 = constants.eps0;

    size_t nSpecies = charge.size();
    for(size_t s=0; s<nSpecies; s++){
        charge[s] *= constants.e;
        mass[s] *= constants.m_e;
    }

    if(dt==0){ // Float-comparison acceptable because it is initialized exactly
        double wp0 = sqrt(pow(charge[0],2)*density[0]/(eps0*mass[0]));
        dt = dtwp/wp0;
    }

    if(num.size() != 0 && npc.size() != 0){
        cout << "Use only npc or num. Not mixed." << endl;
        return 1;
    }
    if(num.size() == 0) num = vector<int>(nSpecies, 0);
    if(npc.size() == 0) npc = vector<int>(nSpecies, 0);
    if(kappa.size() == 0) kappa = vector<double>(nSpecies, 0);
    if(alpha.size() == 0) alpha = vector<double>(nSpecies, 0);
    if(vx.size() == 0) vx = vector<double>(nSpecies, 0);

    // Sanity checks (avoids segfaults)
    if(charge.size()       != nSpecies
    || mass.size()         != nSpecies
    || density.size()      != nSpecies
    || distribution.size() != nSpecies 
    || npc.size()          != nSpecies
    || num.size()          != nSpecies
    || thermal.size()      != nSpecies
    || vx.size()           != nSpecies
    || kappa.size()        != nSpecies
    || alpha.size()        != nSpecies){

        cout << "Wrong arguments for species." << endl;
        return 1;
    }

    //
    // CREATE MESH
    //
    auto mesh = load_mesh(fname_mesh);
    auto boundaries = load_boundaries(mesh, fname_mesh);
    size_t dim = mesh->geometry().dim();

    if(dim==2){
        return run<2>(mesh, boundaries, steps, dt, Bx, impose_current, imposed_current, imposed_voltage, npc, num, density, thermal, vx, charge, mass, kappa, alpha, distribution, binary);
    } else if(dim==3){
        return run<3>(mesh, boundaries, steps, dt, Bx, impose_current, imposed_current, imposed_voltage, npc, num, density, thermal, vx, charge, mass, kappa, alpha, distribution, binary);
    } else {
        cout << "Only 2D and 3D supported" << endl;
        return 1;
    }
}
