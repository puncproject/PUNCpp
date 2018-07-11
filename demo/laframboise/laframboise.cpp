#include <dolfin.h>
#include <punc.h>

using namespace punc;

int main()
{
    df::set_log_level(df::WARNING);

    // std::string fname{"../../mesh/3D/laframboise_sphere_in_sphere_res1"};
    std::string fname{"../../mesh/3D/laframboise_sphere_in_sphere_res1b"};
    auto mesh = load_mesh(fname);
    auto dim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);

    std::vector<double> Ld = get_mesh_size(mesh);
    std::vector<double> vd(dim);
    for (std::size_t i = 0; i<dim; ++i)
    {
        vd[i] = 0.0;
    }

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
    double X = Rp;
    double Te = e*e*debye*debye*ne/(eps0*kB);
    double wpe = sqrt(ne*e*e/(eps0*me));
    double vthe = debye*wpe;
    double vthi = vthe/sqrt(1836.);

    double Vlam  = kB*Te/e;
    double Ilam  = -e*ne*Rp*Rp*sqrt(8.*M_PI*kB*Te/me);
    // double Iexp  = 1.987*Ilam;
    // double Iexp = 2.945 * Ilam;
    double Iexp = 2.824 * Ilam;

    double dt = 0.10;
    std::size_t steps = 20;

    CreateSpecies create_species(mesh, facet_vec, X);

    auto pdf = [](std::vector<double> t)->double{return 1.0;};
    create_species.create(-e, me, ne, npc, vthe, vd, pdf, 1.0);
    create_species.create(e, mi, ne, npc, vthi, vd, pdf, 1.0);

    auto species = create_species.species;

    double Inorm  = create_species.Q/create_species.T;
    double Vnorm  = (create_species.M/create_species.Q)*(create_species.X/create_species.T)*(create_species.X/create_species.T);
    Inorm /= fabs(Ilam);
    Vnorm /= Vlam;

    // double cap_factor = 1.;
    double current_collected = Iexp/(create_species.Q/create_species.T);
    double imposed_potential = 2.0/Vnorm;
    eps0 = 1.0;

    std::vector<std::vector<int>> isources{};//{{-1,0}};
    std::vector<double> ivalues{};//{-current_collected};

    std::vector<std::vector<int>> vsources{{-1,0}};
    std::vector<double> vvalues{imposed_potential};

    printf("Q:  %e\n", create_species.Q);
    printf("T:  %e\n", create_species.T);

    printf("Inorm:  %e\n", Inorm);
    printf("Vnorm:  %e\n", Vnorm);

    printf("Laframboise voltage:  %e\n", Vlam);
    printf("Laframboise current:  %e\n", Ilam);
    printf("Expected current:     %e\n", Iexp);
    printf("Imposed potential:    %e\n", imposed_potential);
    printf("Imposed current:    %e\n", -current_collected);

    auto V = function_space(mesh);
    auto Q = DG0_space(mesh);
    auto dv_inv = element_volume(V);

    auto u0 = std::make_shared<df::Constant>(0.0);
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V), u0,
		std::make_shared<df::MeshFunction<std::size_t>>(boundaries), ext_bnd_id);
    std::vector<df::DirichletBC> ext_bc = {bc};

    ObjectBC object(V, boundaries, tags[2]);
	// object.set_potential(0.0);
	std::vector<ObjectBC> int_bc = {object};

    std::vector<std::size_t> bnd_id{tags[2]};
    Circuit circuit(V, int_bc, isources, ivalues, vsources, vvalues, dt);
	// boost_matrix inv_capacity = capacitance_matrix(V, int_bc, boundaries, ext_bnd_id);

    PoissonSolver poisson(V, ext_bc, circuit);
    ESolver esolver(V);
    // reset_objects(int_bc);

    Population pop(mesh, boundaries);

    load_particles(pop, species);

    std::vector<double> KE(steps);
    std::vector<double> PE(steps);
    std::vector<double> TE(steps);
    std::vector<double> num_e(steps);
    std::vector<double> num_i(steps);
    std::vector<double> num_tot(steps);
    // std::vector<double> num_particles_outside(steps - 1);
    // std::vector<double> num_injected_particles(steps - 1);
    std::vector<double> potential(steps, 0.0);
    std::vector<double> current_measured(steps, 0.0);
    std::vector<double> obj_charge(steps, 0.0);

    double old_charge = 0.0;
    Timer timer;
    std::vector<double> dist(steps),rsetobj(steps),pois(steps),efil(steps),upd(steps);
    std::vector<double> objpoten(steps),pot(steps),ace(steps),mv(steps), inj(steps), pnum(steps);
    boost::optional<double> opt = NAN;
    num_i[0] = pop.num_of_positives();
    num_e[0] = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "Num positives:  " << num_i[0];
    std::cout << ", num negatives: " << num_e[0];
    std::cout << " total: " << num3 << '\n';

    std::ofstream file;

    for(int i=0; i<steps;++i)
    {
        std::cout<<"step: "<< i<<'\n';
        // auto rho = distribute(V, pop, dv_inv);
        auto rho = distribute_dg0(Q, pop);
        dist[i] = timer.elapsed();
        timer.reset();
        // reset_objects(int_bc);
        // rsetobj[i]= timer.elapsed();
        timer.reset();
        auto phi = poisson.solve(rho, int_bc, circuit, V);
        // df::File ofile("phi.pvd");
        // ofile << phi;
        pois[i] = timer.elapsed();
        timer.reset();

        for(auto& o: int_bc)
        {
            // printf("Object charge before: %e\n", o.charge);
            auto _charge = o.update_charge(phi);
            // printf("Object charge after: %e\n", o.charge);
        }

        potential[i] = int_bc[0].update_potential(phi)*Vnorm;
        // printf("Object potential: %e\n", potential[i]);
        auto E = esolver.solve(phi);
        efil[i] = timer.elapsed();
        timer.reset();

        // compute_object_potentials(int_bc, E, inv_capacity, mesh);
        // objpoten[i] = timer.elapsed();
        // timer.reset();
        //
        // potential[i] = int_bc[0].potential*Vnorm;

        // auto phi1 = poisson.solve(rho, int_bc);
        // pois[i] += timer.elapsed();
        // timer.reset();
        //
        // auto E1 = esolver.solve(phi1);
        // efil[i] += timer.elapsed();
        // timer.reset();

        PE[i] = particle_potential_energy(pop, phi);
        pot[i] = timer.elapsed();
        timer.reset();

        old_charge = int_bc[0].charge;

        KE[i] = accel(pop, E, (1.0-0.5*(i == 1))*dt);
        ace[i] = timer.elapsed();
        if(i==0)
        {
            KE[i] = kinetic_energy(pop);
        }
        timer.reset();
        move(pop, dt);
        mv[i] = timer.elapsed();
        timer.reset();

        pop.update(int_bc);
        upd[i]= timer.elapsed();

        current_measured[i] = ((int_bc[0].charge - old_charge) / dt) * Inorm;
        // printf("Current: %e\n", current_measured[i]);
        // int_bc[0].charge -= current_collected*dt;
        // obj_charge[i] = int_bc[0].charge;

        timer.reset();

        inject_particles(pop, species, facet_vec, dt);
        inj[i] = timer.elapsed();

        timer.reset();
        num_e[i] = pop.num_of_negatives();
        num_i[i] = pop.num_of_positives();
        num_tot[i] = pop.num_of_particles();
        pnum[i] = timer.elapsed();
        timer.reset();

        file.open("history.dat", std::ofstream::out | std::ofstream::app);
        file << i << "\t" << num_e[i] << "\t" << num_i[i] << "\t" << KE[i] << "\t" << PE[i] << "\t" << potential[i] << "\t" << current_measured[i] << '\n';
        file.close();
    }

    auto time_dist = std::accumulate(dist.begin(), dist.end(), 0.0);
    auto time_rsetobj = std::accumulate(rsetobj.begin(), rsetobj.end(), 0.0);
    auto time_pois = std::accumulate(pois.begin(), pois.end(), 0.0);
    auto time_efil = std::accumulate(efil.begin(), efil.end(), 0.0);
    auto time_upd = std::accumulate(upd.begin(), upd.end(), 0.0);
    auto time_objpoten = std::accumulate(objpoten.begin(), objpoten.end(), 0.0);
    auto time_pot = std::accumulate(pot.begin(), pot.end(), 0.0);
    auto time_ace = std::accumulate(ace.begin(), ace.end(), 0.0);
    auto time_mv = std::accumulate(mv.begin(), mv.end(), 0.0);
    auto time_inj = std::accumulate(inj.begin(), inj.end(), 0.0);
    auto time_pnum = std::accumulate(pnum.begin(), pnum.end(), 0.0);

    double total_time = time_dist + time_rsetobj + time_pois + time_efil + time_upd;
    total_time += time_objpoten + time_pot + time_ace + time_mv + time_inj+time_pnum;

    std::cout << "----------------Measured time for each task----------------" << '\n';
    std::cout<<"        Task         "<<" Time  "      <<"  "<<" Prosentage "<<'\n';
    std::cout<<"Distribution:        "<< time_dist     <<"    "<<100*time_dist/total_time<< '\n';
    std::cout<<"Reset objects:       "<< time_rsetobj  <<"    "<<100*time_rsetobj/total_time<<'\n';
    std::cout<<"poisson:             "<<  time_pois    <<"    "<<100*time_pois/total_time<<'\n';
    std::cout<<"efield:              "<< time_efil     <<"    "<<100*time_efil/total_time <<'\n';
    std::cout<<"update:              "<< time_upd      <<"    "<<100*time_upd/total_time <<'\n';
    std::cout<<"move:                "<< time_mv       <<"    "<<100*time_mv/total_time<<'\n';
    std::cout<<"inject:              "<< time_inj      <<"    "<<100*time_inj/total_time <<'\n';
    std::cout<<"accel:               "<< time_ace      <<"    "<<100*time_ace/total_time <<'\n';
    std::cout<<"potential energy:    "<< time_pot      <<"    "<<100*time_pot/total_time <<'\n';
    std::cout<<"object potential:    "<< time_objpoten <<"    "<<100*time_objpoten/total_time<<'\n';
    std::cout<<"counting particles:  "<< time_pnum     <<"    "<<100*time_pnum/total_time<<'\n';
    std::cout<<"Total time:          " << total_time   <<"    "<<100*total_time / total_time << '\n';
    return 0;
}
