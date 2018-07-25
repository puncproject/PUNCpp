#include <dolfin.h>
#include <punc.h>

using namespace punc;

using std::cout;
using std::endl;
using std::size_t;
using std::vector;
using std::accumulate;
using std::string;

int main(){
    df::set_log_level(df::WARNING);

    //
    // CREATE MESH
    //
    string fname{"../../mesh/3D/laframboise_sphere_in_sphere_res1b"};
    auto mesh = load_mesh(fname);
    auto dim = mesh->geometry().dim();

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

    auto pdf = [](vector<double> t)->double{return 1.0;};

    CreateSpecies create_species(mesh, facet_vec);
    create_species.create_raw(-e, me, ne, npc, vthe, vd, pdf, 1.0);
    create_species.create_raw( e, mi, ne, npc, vthi, vd, pdf, 1.0);
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
    size_t steps = 10;
    double dt = 0.2/wpe;

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
    // LOAD PARTICLES
    //
    Population pop(mesh, boundaries);
    load_particles(pop, species);

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

    cout << "Num positives:  " << num_i;
    cout << ", num negatives: " << num_e;
    cout << " total: " << num_tot << endl;

    //
    // CREATE TIMER VARIABLES
    //
    Timer timer;
    vector<double> t_dist(steps);
    vector<double> t_rsetobj(steps);
    vector<double> t_pois(steps);
    vector<double> t_efil(steps);
    vector<double> t_upd(steps);
    vector<double> t_objpoten(steps);
    vector<double> t_pot(steps);
    vector<double> t_ace(steps);
    vector<double> t_mv(steps);
    vector<double> t_inj(steps);
    vector<double> t_pnum(steps);

    std::ofstream file;

    for(size_t i=0; i<steps;++i)
    {
        cout << "Step: " << i << endl;

        // DISTRIBUTE

        // auto rho = distribute(V, pop, dv_inv);
        auto rho = distribute_dg0(Q, pop);
        t_dist[i] = timer.elapsed();
        timer.reset();

        // SOLVE POISSON

        // reset_objects(int_bc);
        // t_rsetobj[i]= timer.elapsed();
        timer.reset();
        /* int_bc[0].charge = 0; */
        /* int_bc[0].charge -= current_collected*dt; */
        auto phi = poisson.solve(rho, int_bc, circuit, V);
        t_pois[i] = timer.elapsed();
        timer.reset();

        for(auto& o: int_bc) o.update_charge(phi);

        potential = int_bc[0].update_potential(phi);
        auto E = esolver.solve(phi);
        t_efil[i] = timer.elapsed();
        timer.reset();

        // compute_object_potentials(int_bc, E, inv_capacity, mesh);
        // t_objpoten[i] = timer.elapsed();
        // timer.reset();
        //
        // potential[i] = int_bc[0].potential;

        // auto phi1 = poisson.solve(rho, int_bc);
        // t_pois[i] += timer.elapsed();
        // timer.reset();
        //
        // auto E1 = esolver.solve(phi1);
        // t_efil[i] += timer.elapsed();
        // timer.reset();

        PE = particle_potential_energy(pop, phi);
        t_pot[i] = timer.elapsed();

        // PUSH PARTICLES

        timer.reset();

        old_charge = int_bc[0].charge;

        KE = accel(pop, E, (1.0-0.5*(i == 0))*dt);
        t_ace[i] = timer.elapsed();
        if(i==0) KE = kinetic_energy(pop);
        timer.reset();
        move(pop, dt);
        t_mv[i] = timer.elapsed();
        timer.reset();

        pop.update(int_bc);
        t_upd[i]= timer.elapsed();

        current_measured = ((int_bc[0].charge - old_charge) / dt);


        // INJECT PARTICLES

        timer.reset();
        inject_particles(pop, species, facet_vec, dt);
        t_inj[i] = timer.elapsed();
        
        // COUNT PARTICLES

        timer.reset();
        num_e     = pop.num_of_negatives();
        num_i     = pop.num_of_positives();
        num_tot   = pop.num_of_particles();
        t_pnum[i] = timer.elapsed();

        timer.reset();

        // WRITE HISTORY

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
    }

    auto time_dist     = accumulate(t_dist.begin()    , t_dist.end()    , 0.0);
    auto time_rsetobj  = accumulate(t_rsetobj.begin() , t_rsetobj.end() , 0.0);
    auto time_pois     = accumulate(t_pois.begin()    , t_pois.end()    , 0.0);
    auto time_efil     = accumulate(t_efil.begin()    , t_efil.end()    , 0.0);
    auto time_upd      = accumulate(t_upd.begin()     , t_upd.end()     , 0.0);
    auto time_objpoten = accumulate(t_objpoten.begin(), t_objpoten.end(), 0.0);
    auto time_pot      = accumulate(t_pot.begin()     , t_pot.end()     , 0.0);
    auto time_ace      = accumulate(t_ace.begin()     , t_ace.end()     , 0.0);
    auto time_mv       = accumulate(t_mv.begin()      , t_mv.end()      , 0.0);
    auto time_inj      = accumulate(t_inj.begin()     , t_inj.end()     , 0.0);
    auto time_pnum     = accumulate(t_pnum.begin()    , t_pnum.end()    , 0.0);

    double total_time = time_dist + time_rsetobj + time_pois + time_efil + time_upd;
    total_time += time_objpoten + time_pot + time_ace + time_mv + time_inj+time_pnum;

    cout << "----------------Measured time for each task----------------" << endl;
    cout << "        Task         " << " Time  "     << "  "   << " Procentage "               << endl;
    cout << "Distribution:        " << time_dist     << "    " << 100*time_dist/total_time     << endl;
    cout << "Reset objects:       " << time_rsetobj  << "    " << 100*time_rsetobj/total_time  << endl;
    cout << "poisson:             " << time_pois     << "    " << 100*time_pois/total_time     << endl;
    cout << "efield:              " << time_efil     << "    " << 100*time_efil/total_time     << endl;
    cout << "update:              " << time_upd      << "    " << 100*time_upd/total_time      << endl;
    cout << "move:                " << time_mv       << "    " << 100*time_mv/total_time       << endl;
    cout << "inject:              " << time_inj      << "    " << 100*time_inj/total_time      << endl;
    cout << "accel:               " << time_ace      << "    " << 100*time_ace/total_time      << endl;
    cout << "potential energy:    " << time_pot      << "    " << 100*time_pot/total_time      << endl;
    cout << "object potential:    " << time_objpoten << "    " << 100*time_objpoten/total_time << endl;
    cout << "counting particles:  " << time_pnum     << "    " << 100*time_pnum/total_time     << endl;
    cout << "Total time:          " << total_time    << "    " << 100*total_time / total_time  << endl;

    return 0;
}
