#include <dolfin.h>
#include <punc.h>
#include <csignal>

using namespace punc;

using std::cout;
using std::endl;
using std::size_t;
using std::vector;
using std::accumulate;
using std::string;

bool exit_now = false;
void signal_handler(int signum){
    if(exit_now){
        exit(signum);
    } else {
        cout << "Completing and storing timestep before exiting. ";
        cout << "Press Ctrl+C again to force quit." << endl;
        exit_now = true;
    }
}

int main(){

    signal(SIGINT, signal_handler);
    df::set_log_level(df::WARNING);

    //
    // CREATE MESH
    //
    string fname{"../../mesh/3D/laframboise_sphere_in_sphere_res1b"};
    auto mesh = load_mesh(fname);
    const std::size_t dim = 3;//mesh->geometry().dim();

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    size_t ext_bnd_id = tags[1];

    auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);

    vector<double> Ld = get_mesh_size(mesh);

    //
    // CREATE SPECIES
    //
    PhysicalConstants constants;
    double e = constants.e;
    double me = constants.m_e;
    double mi = constants.m_i;
    double kB = constants.k_B;
    double eps0 = constants.eps0;

    int npc = 4;
    double ne = 1.0e10;
    double debye = 1.0;
    double Rp = 1.0*debye;
    double Te = e*e*debye*debye*ne/(eps0*kB);
    double wpe = sqrt(ne*e*e/(eps0*me));
    double vthe = debye*wpe;
    double vthi = vthe/sqrt(1836.);
    vector<double> vd(dim, 0);

    UniformPosition pdfe(mesh); // Electron position distribution
    UniformPosition pdfi(mesh); // Ion position distribution

    Maxwellian vdfe(vthe, vd); // Velocity distribution for electrons
    Maxwellian vdfi(vthi, vd); // Velocity distribution for ions

    CreateSpecies create_species(mesh);
    create_species.create_raw(-e, me, ne, pdfe, vdfe, npc);
    create_species.create_raw( e, mi, ne, pdfi, vdfi, npc);
    auto species = create_species.species;

    //
    // IMPOSE CIRCUITRY
    //
    double V0 = kB*Te/e;
    double I0 = -e*ne*Rp*Rp*sqrt(8.*M_PI*kB*Te/me);

    bool impose_current = true; 
    double imposed_current = 2.945*I0;
    double imposed_potential = 2*V0;

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
        vvalues = {imposed_potential};
    }

    //
    // TIME STEP
    //
    size_t steps = 5000;
    double dt = 0.05/wpe;

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
    create_flux(species, facet_vec);

    //
    // LOAD PARTICLES
    //
    Population<dim> pop(mesh, boundaries);
    load_particles<dim>(pop, species);

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
    cout << "imposed_potential: " << imposed_potential << '\n';

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

    std::ofstream file;

    for(size_t i=0; i<steps;++i)
    {
        cout << "Step: " << i << endl;

        // DISTRIBUTE
        timer.reset();
        auto rho = distribute_cg1(V, pop, dv_inv);
        /* auto rho = distribute_dg0<dim>(Q, pop); */
        t_dist[i] = timer.elapsed();

        // SOLVE POISSON
        timer.reset();
        // reset_objects(int_bc);
        // t_rsetobj[i]= timer.elapsed();
        /* int_bc[0].charge = 0; */
        /* int_bc[0].charge -= current_collected*dt; */
        auto phi = poisson.solve(rho, int_bc, circuit, V);
        t_poisson[i] = timer.elapsed();


        for(auto& o: int_bc) o.update_charge(phi);

        potential = int_bc[0].update_potential(phi);
        
        // ELECTRIC FIELD
        timer.reset();
        auto E = esolver.solve(phi);
        t_efield[i] = timer.elapsed();
        

        // if (i==100)
        // {
        //     df::File ofile("phi.pvd");
        //     ofile<<phi;
        // }
        // compute_object_potentials(int_bc, E, inv_capacity, mesh);
        // t_objpoten[i] = timer.elapsed();
        // timer.reset();
        //
        // potential[i] = int_bc[0].potential;

        // auto phi1 = poisson.solve(rho, int_bc);
        // t_poisson[i] += timer.elapsed();
        // timer.reset();
        //
        // auto E1 = esolver.solve(phi1);
        // t_efield[i] += timer.elapsed();
        // timer.reset();

        // POTENTIAL ENERGY
        timer.reset();
        PE = particle_potential_energy_cg1<dim>(pop, phi);
        t_potential[i] = timer.elapsed();

        // COUNT PARTICLES
        timer.reset();
        num_e     = pop.num_of_negatives();
        num_i     = pop.num_of_positives();
        num_tot   = pop.num_of_particles();
        t_count[i] = timer.elapsed();
        /* cout << "ions: "<<num_i; */
        /* cout << "  electrons: " << num_e; */
        /* cout << "  total: " << num_tot << '\n'; */

        // PUSH PARTICLES AND CALCULATE THE KINETIC ENERGY
        old_charge = int_bc[0].charge;

        timer.reset();
        KE = accel_cg1<dim>(pop, E, (1.0-0.5*(i == 0))*dt);
        if(i==0) KE = kinetic_energy(pop);
        t_accel[i] = timer.elapsed();

        // WRITE HISTORY
        timer.reset();
        file.open("history.dat", std::ofstream::out | std::ofstream::app);
        file << i << "\t";
        file << i*dt << "\t";
        file << num_e << "\t";
        file << num_i << "\t";
        file << KE << "\t";
        file << PE << "\t";
        file << potential << "\t";
        file << current_measured << endl;
        file.close();
        t_io[i] = timer.elapsed();

        // MOVE PARTICLES
        timer.reset();
        move<dim>(pop, dt);
        t_move[i] = timer.elapsed();

        // UPDATE PARTICLE POSITIONS
        timer.reset();
        pop.update(int_bc);
        t_update[i]= timer.elapsed();

        current_measured = ((int_bc[0].charge - old_charge) / dt);

        // INJECT PARTICLES
        timer.reset();
        inject_particles<dim>(pop, species, facet_vec, dt);
        t_inject[i] = timer.elapsed();
        
        // SAVE STATE AND BREAK LOOP
        if(exit_now || i==steps-1){
            // save_state()
            // pop.save_file('population.dat')
            break;
        }
    }

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

    return 0;
}
