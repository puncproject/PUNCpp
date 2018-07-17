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

    bool remove_null_space = true;
    std::vector<double> Ld = get_mesh_size(mesh);
    std::vector<bool> periodic(D);
    std::vector<double> vd(D);
    for (std::size_t i = 0; i<D; ++i)
    {
        periodic[i] = true;
        vd[i] = 0.0;
    }

    auto constr = std::make_shared<PeriodicBoundary>(Ld, periodic);

    auto V = function_space(mesh, constr);
    PoissonSolver poisson(V, boost::none, boost::none, remove_null_space);
    ESolver esolver(V);

    auto dv_inv = element_volume(V, true);

    double vth = 0.0;
    int npc = 4;

    CreateSpecies create_species(mesh, Ld[0]);

    double A = 0.5, mode = 1.0;

    LangmuirWave2D pdfe(mesh, A, mode, Ld); // Electron position distribution
    UniformPosition pdfi(mesh);             // Ion position distribution

    Maxwellian vdfe(vth, vd);             // Velocity distribution for electrons
    Maxwellian vdfi(vth, vd);             // Velocity distribution for ions

    // RejectionSampler ePosSampler(pdfe);
    // RejectionSampler iPosSampler(pdfi);
    // RejectionSampler eVelSampler(vdfe);
    // RejectionSampler iVelSampler(vdfi);

    // std::vector<Sampler*> posSampler{&ePosSampler, &iPosSampler};
    // std::vector<Sampler*> velSampler{&eVelSampler, &iVelSampler};

    PhysicalConstants constants;
    double e = constants.e;
    double me = constants.m_e;
    double mi = constants.m_i;

    create_species.create(-e, me, 100, pdfe, vdfe, npc);
    create_species.create(e, mi, 100, pdfi, vdfi, npc);

    auto species = create_species.species;

    Population pop(mesh, boundaries);

    load_particles(pop, species, "RejectionSampler", "RejectionSampler");
    // load_particles(pop, species, rejection_sampler, rejection_sampler);

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
