#include <dolfin.h>
#include <punc.h>

using namespace punc;

int main()
{
    df::set_log_level(df::WARNING);
    Timer timer;
    timer.reset();
    double dt = 0.25;
    std::size_t steps = 30;

    std::string fname{"../../mesh/2D/nothing_in_square"};
    auto mesh = load_mesh(fname);
    auto D = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto facet_vec = exterior_boundaries(boundaries, ext_bnd_id);

    bool remove_null_space = true;
    std::vector<double> Ld = get_mesh_size(mesh);
    std::vector<bool> periodic(D);
    std::vector<double> vd(D, 0.0);
    for (std::size_t i = 0; i<D; ++i)
    {
        periodic[i] = true;
        // vd[i] = 0.0;
    }

    auto constr = std::make_shared<PeriodicBoundary>(Ld, periodic);

    auto V = function_space(mesh, constr);

    PhysicalConstants constants;
    double e = constants.e;
    double me = constants.m_e;
    double mi = constants.m_i;
    double eps0 = constants.eps0;

    auto dv_inv = element_volume(V, true);

    double vth = 0.0;
    int npc = 4;
    double ne = 100;

    CreateSpecies create_species(mesh, facet_vec, Ld[0]);

    double A = 0.5, mode = 1.0;
    double pdf_max = 1.0+A;

    auto pdfe = [A, mode, Ld](std::vector<double> t)->double{return 1.0+A*sin(2*mode*M_PI*t[0]/Ld[0]);};
    auto pdfi = [](std::vector<double> t)->double{return 1.0;};

    bool si_units = true;
    if (!si_units)
    {
        create_species.create(-e, me, ne, npc, vth, vd, pdfe, pdf_max);
        create_species.create(e, mi, ne, npc, vth, vd, pdfi, 1.0);
        eps0 = 1.0;
    }
    else
    {
        double wpe = sqrt(ne * e * e / (eps0 * me));
        dt /= wpe;
        create_species.T = 1;
        create_species.Q = 1;
        create_species.M = 1;
        create_species.X = 1;

        create_species.create_raw(-e, me, ne, npc, vth, vd, pdfe, pdf_max);
        create_species.create_raw(e, mi, ne, npc, vth, vd, pdfi, 1.0);
    }

    auto species = create_species.species;
    
    Population pop(mesh, boundaries);

    PoissonSolver poisson(V, boost::none, boost::none, eps0, remove_null_space);
    ESolver esolver(V);

    load_particles(pop, species);

    auto num1 = pop.num_of_positives();
    auto num2 = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "Num positives:  " << num1;
    std::cout << ", num negatives: " << num2;
    std::cout << " total: " << num3 << '\n';

    std::vector<double> KE(steps-1);
    std::vector<double> PE(steps-1);
    std::vector<double> TE(steps-1);
    double KE0 = kinetic_energy(pop);

    auto t0 = timer.elapsed();
    printf("Time - initilazation: %e\n", t0);
    timer.reset();
    for(int i=1; i<steps;++i)
    {
        std::cout<<"step: "<< i<<'\n';
        auto rho = distribute(V, pop, dv_inv);
        auto phi = poisson.solve(rho);
        auto E = esolver.solve(phi);
        PE[i - 1] = particle_potential_energy(pop, phi);
        KE[i-1] = accel(pop, E, (1.0-0.5*(i == 1))*dt);
        move_periodic(pop, dt, Ld);
        pop.update();
    }
    t0 = timer.elapsed();
    printf("Time - loop: %e\n", t0);
    KE[0] = KE0;
    for(int i=0;i<KE.size(); ++i)
    {
        TE[i] = PE[i] + KE[i];
    }
    std::ofstream out;
    out.open("PE.txt");
    for (const auto &e : PE) out << e << "\n";
    out.close();
    std::ofstream out1;
    out1.open("KE.txt");
    for (const auto &e : KE) out1 << e << "\n";
    out1.close();
    std::ofstream out2;
    out2.open("TE.txt");
    for (const auto &e : TE) out2 << e << "\n";
    out2.close();

    return 0;
}
