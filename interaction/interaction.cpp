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

bool exit_immediately = true;
void signal_handler(int signum){
    if(exit_immediately){
        exit(signum);
    } else {
        cout << "Completing and storing timestep before exiting. ";
        cout << "Press Ctrl+C again to force quit." << endl;
        exit_immediately = true;
    }
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

    // Object input
    bool impose_current = true; 
    double imposed_current;
    double imposed_voltage;

    // Species input
    vector<int> npc;
    vector<int> num;
    vector<double> density;
    vector<double> thermal;
    vector<double> charge;
    vector<double> mass;
    vector<string> distribution;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "show help (this)")
        ("input", po::value(&fname_ifile), "config file")
        ("mesh", po::value(&fname_mesh), "mesh file")
        ("steps", po::value(&steps), "number of timesteps")
        ("dt", po::value(&dt), "timestep [s] (overrides dtwp)") 
        ("dtwp", po::value(&dtwp), "timestep [1/w_p of first specie]")
        
        ("impose_current", po::value(&impose_current), "Whether to impose current or voltage (true|false)")
        ("imposed_current", po::value(&imposed_current), "Current imposed on object [A]")
        ("imposed_voltage", po::value(&imposed_voltage), "Voltage imposed on object [V]")

        ("species.charge", po::value(&charge), "charge [elementary chages]")
        ("species.mass", po::value(&mass), "mass [electron masses]")
        ("species.density", po::value(&density), "number density [1/m^3]")
        ("species.thermal", po::value(&thermal), "thermal speed [m/s]")
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
        double wp0 = sqrt(pow(charge[0],2)*density[0]/(constants.eps0*mass[0]));
        dt = dtwp/wp0;
    }

    if(num.size() != 0 && npc.size() != 0){
        cout << "Use only npc or num. Not mixed." << endl;
        return 1;
    }
    if(num.size() == 0) num = vector<int>(nSpecies, 0);
    if(npc.size() == 0) npc = vector<int>(nSpecies, 0);

    // Sanity checks (avoids segfaults)
    if(charge.size()       != nSpecies
    || mass.size()         != nSpecies
    || density.size()      != nSpecies
    || distribution.size() != nSpecies 
    || npc.size()          != nSpecies
    || num.size()          != nSpecies
    || thermal.size()      != nSpecies){

        cout << "Wrong arguments for species." << endl;
        return 1;
    }

    //
    // CREATE MESH
    //
    auto mesh = load_mesh(fname_mesh);

    // FIXME: This really shouldn't be necessary. This is what polymorphism is
    // for. Well written code don't require recompilation for different input.
    const std::size_t dim = 3;//mesh->geometry().dim();

    auto boundaries = load_boundaries(mesh, fname_mesh);
    auto tags = get_mesh_ids(boundaries);
    size_t ext_bnd_id = tags[1];

    auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);

    vector<double> Ld = get_mesh_size(mesh);

    //
    // CREATE SPECIES
    //

    // FIXME: Move to input file
    vector<double> vd(dim, 0);

    vector<std::shared_ptr<Pdf>> pdfs;
    vector<std::shared_ptr<Pdf>> vdfs;

    CreateSpecies create_species(mesh);
    for(size_t s=0; s<charge.size(); s++){

        pdfs.push_back(std::make_shared<UniformPosition>(mesh));

        if(distribution[s]=="maxwellian"){
            vdfs.push_back(std::make_shared<Maxwellian>(thermal[s], vd));
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

        pop.load_file(fname_pop);
        file_hist.open(fname_hist, ofstream::out | ofstream::app);

    } else {
        cout << "Starting new simulation" << endl;

        load_particles<dim>(pop, species);
        file_hist.open(fname_hist, ofstream::out);

        // Preamble for metaplot
        file_hist << "#:xaxis\tt\n";
        file_hist << "#:name\tn\tt\tne\tni\tKE\tPE\tV\tI\n";
        file_hist << "#:long\ttimestep\ttime\t\"electron density\"\t";
        file_hist << "\"ion density\"\t\"kinetic energy\"\t";
        file_hist << "\"potential energy\"\tvoltage\tcurrent\n";
        file_hist << "#:units\t1\ts\tm**(-3)\tm**(-3)\tJ\tJ\tV\tA\n";
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
    // CREATE TIMER VARIABLES
    //
    Timer timer;
    vector<double> t_dist(steps);
    vector<double> t_poisson(steps);
    vector<double> t_efield(steps);
    vector<double> t_update(steps);
    vector<double> t_potential(steps);
    vector<double> t_accel(steps);
    vector<double> t_move(steps);
    vector<double> t_inject(steps);
    vector<double> t_count(steps);
    vector<double> t_io(steps);
    vector<double> t_rsetobj(steps);
    vector<double> t_objpoten(steps);

    exit_immediately = false;
    for(; n<=steps; ++n){

        // We are now at timestep n
        // Velocities and currents are at timestep n-0.5 (or 0 if n==0)

        if(override_status_print) cout << "\r";
        cout << "Step " << n << " of " << steps;
        if(!override_status_print) cout << "\n";

        // DISTRIBUTE
        timer.reset();
        auto rho = distribute_cg1(V, pop, dv_inv);
        t_dist[n] = timer.elapsed();

        // SOLVE POISSON
        timer.reset();
        // reset_objects(int_bc);
        // t_rsetobj[n]= timer.elapsed();
        /* int_bc[0].charge = 0; */
        /* int_bc[0].charge -= current_collected*dt; */
        auto phi = poisson.solve(rho, int_bc, circuit, V);
        t_poisson[n] = timer.elapsed();


        for(auto& o: int_bc) o.update_charge(phi);

        potential = int_bc[0].update_potential(phi);
        
        // ELECTRIC FIELD
        timer.reset();
        auto E = esolver.solve(phi);
        t_efield[n] = timer.elapsed();
        

        // if (n==100)
        // {
        //     df::File ofile("phi.pvd");
        //     ofile<<phi;
        // }
        // compute_object_potentials(int_bc, E, inv_capacity, mesh);
        // t_objpoten[n] = timer.elapsed();
        // timer.reset();
        //
        // potential[n] = int_bc[0].potential;

        // auto phi1 = poisson.solve(rho, int_bc);
        // t_poisson[n] += timer.elapsed();
        // timer.reset();
        //
        // auto E1 = esolver.solve(phi1);
        // t_efield[n] += timer.elapsed();
        // timer.reset();

        // POTENTIAL ENERGY
        timer.reset();
        PE = particle_potential_energy_cg1<dim>(pop, phi);
        t_potential[n] = timer.elapsed();

        // COUNT PARTICLES
        timer.reset();
        num_e     = pop.num_of_negatives();
        num_i     = pop.num_of_positives();
        num_tot   = pop.num_of_particles();
        t_count[n] = timer.elapsed();
        /* cout << "ions: "<<num_i; */
        /* cout << "  electrons: " << num_e; */
        /* cout << "  total: " << num_tot << '\n'; */

        // PUSH PARTICLES AND CALCULATE THE KINETIC ENERGY
        // Advancing velocities to n+0.5
        old_charge = int_bc[0].charge;

        timer.reset();
        KE = accel_cg1<dim>(pop, E, (1.0-0.5*(n == 0))*dt);
        if(n==0) KE = kinetic_energy(pop);
        t_accel[n] = timer.elapsed();

        // WRITE HISTORY
        // Everything at n, except currents wich are at n-0.5.
        timer.reset();
        file_hist << n << "\t";
        file_hist << t << "\t";
        file_hist << num_e << "\t";
        file_hist << num_i << "\t";
        file_hist << KE << "\t";
        file_hist << PE << "\t";
        file_hist << potential << "\t";
        file_hist << current_measured << endl;
        t_io[n] = timer.elapsed();

        // MOVE PARTICLES
        // Advancing position to n+1
        timer.reset();
        move<dim>(pop, dt);
        t_move[n] = timer.elapsed();
        t += dt;

        // UPDATE PARTICLE POSITIONS
        timer.reset();
        pop.update(int_bc);
        t_update[n]= timer.elapsed();

        current_measured = ((int_bc[0].charge - old_charge) / dt);
        int_bc[0].collected_current = current_measured;

        // INJECT PARTICLES
        timer.reset();
        inject_particles<dim>(pop, species, facet_vec, dt);
        t_inject[n] = timer.elapsed();
        
        // SAVE STATE AND BREAK LOOP
        if(exit_immediately || n==steps){
            pop.save_file(fname_pop);

            ofstream state_file;
            state_file.open(fname_state, ofstream::out);
            state_file << n+1 << "\t" << t << "\t";
            state_file << int_bc[0].charge << "\t";
            state_file << int_bc[0].collected_current << endl;
            state_file.close();

            break;
        }
    }
    if(override_status_print) cout << endl;

    file_hist.close();

    auto time_dist      = accumulate(t_dist.begin(), t_dist.end(), 0.0);
    auto time_poisson   = accumulate(t_poisson.begin(), t_poisson.end(), 0.0);
    auto time_efield    = accumulate(t_efield.begin(), t_efield.end(), 0.0);
    auto time_update    = accumulate(t_update.begin(), t_update.end(), 0.0);
    auto time_potential = accumulate(t_potential.begin(), t_potential.end(), 0.0);
    auto time_accel     = accumulate(t_accel.begin(), t_accel.end(), 0.0);
    auto time_move      = accumulate(t_move.begin(), t_move.end(), 0.0);
    auto time_inject    = accumulate(t_inject.begin(), t_inject.end(), 0.0);
    auto time_count     = accumulate(t_count.begin(), t_count.end(), 0.0);
    auto time_io        = accumulate(t_io.begin(), t_io.end(), 0.0);
    auto time_rsetobj   = accumulate(t_rsetobj.begin(), t_rsetobj.end(), 0.0);
    auto time_objpoten  = accumulate(t_objpoten.begin(), t_objpoten.end(), 0.0);

    double total_time = time_dist + time_poisson + time_efield + time_update;
    total_time += time_potential + time_accel + time_move + time_inject;
    total_time += time_count + time_io + time_rsetobj + time_objpoten;

    cout << "----------------Measured time for each task----------------" << endl;
    cout << "        Task         "
         << " Time  "
         << "  "
         << " Percentage " << endl;
    cout << "inject:              " << time_inject << "    " << 100 * time_inject / total_time << endl;
    cout << "update:              " << time_update << "    " << 100 * time_update / total_time << endl;
    cout << "poisson:             " << time_poisson << "    " << 100 * time_poisson / total_time << endl;
    cout << "efield:              " << time_efield << "    " << 100 * time_efield / total_time << endl;
    cout << "accel:               " << time_accel << "    " << 100 * time_accel / total_time << endl;
    cout << "potential energy:    " << time_potential << "    " << 100 * time_potential / total_time << endl;
    cout << "Distribution:        " << time_dist << "    " << 100 * time_dist / total_time << endl;
    cout << "counting particles:  " << time_count << "    " << 100 * time_count / total_time << endl;
    cout << "move:                " << time_move << "    " << 100 * time_move / total_time << endl;
    cout << "write history:       " << time_io << "    " << 100 * time_io / total_time << endl;
    // cout << "Reset objects:       " << time_rsetobj << "    " << 100 * time_rsetobj / total_time << endl;
    // cout << "object potential:    " << time_objpoten << "    " << 100 * time_objpoten / total_time << endl;
    cout << "Total time:          " << total_time << "    " << 100 * total_time / total_time << endl;

    cout << "PUNC++ finished successfully!" << endl;
    return 0;
}
