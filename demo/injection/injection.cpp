#include <dolfin.h>
#include <punc.h>

using namespace punc;

const bool override_status_print = true;

int main()
{
    df::set_log_level(df::WARNING);

    std::vector<std::string> tasks{"move", "update", "injector"};
    Timer timer(tasks);

    const std::size_t dim = 3;

    std::string fname;
    if (dim==2)
    {
        fname = "empty_square";
    }else if (dim==3){
        fname = "empty_cube";
    }
    
    Mesh mesh(fname);
    std::vector<double> Ld = mesh.domain_size();

    PhysicalConstants constants;
    double e = constants.e;
    double me = constants.m_e;
    double eps0 = constants.eps0;

    double amount = 50000;
    double ne = 1e10;
    double debye = 1.0;
    double wpe = sqrt(ne * e * e / (eps0 * me));

    double vthe = debye * wpe;
    std::vector<double> vd(dim, 0.0);

    std::vector<Species> species;

    ParticleAmountType type = ParticleAmountType::in_total;
    type = ParticleAmountType::per_volume;

    std::shared_ptr<Pdf> pdf = std::make_shared<UniformPosition>(mesh); // Position distribution
    // Maxwellian vdf(vthe, vd);  // Maxwellian velocity distribution
    //Kappa vdf(vthe, vd, 3.0);  // Kappa velocity distribution
    // Cairns vdf(vthe, vd, 0.2); // Cairns velocity distribution
    std::shared_ptr<Pdf> vdf = std::make_shared<KappaCairns>(vthe, vd, 4.0, 0.2); // Kappa-Cairns velocity distribution

    std::size_t steps = 100;
    double dt_plasma = 0.05;

    species.emplace_back(-e, me, ne, amount, type, mesh, pdf, vdf, eps0);

    double Tp = min_plasma_period(species, eps0);
    double dt = dt_plasma * Tp;
    
    // CreateSpecies create_species(mesh);

    // bool si_units = true;
    // if (!si_units)
    // {
    //     create_species.X = Ld[0];
    //     create_species.create(-e, me, ne, pdf, vdf, npc);
    //     eps0 = 1.0;
    // }
    // else
    // {
    //     dt /= wpe;
    //     create_species.create_raw(-e, me, ne, pdf, vdf, npc);
    // }

    // auto species = create_species.species;
    std::cout << "Create flux"<<'\n';
    create_flux(species, mesh.exterior_facets);
    std::cout << "flux is created" << '\n';

    Population<dim> pop(mesh);

    std::cout << "load particles" << '\n';
    load_particles(pop, species);
    std::cout << "particles are loaded" << '\n';
    std::string file_name1{"vels_pre.txt"};
    pop.save_file(file_name1, false);

    std::vector<std::shared_ptr<Object>> objects = {};

    std::string hist_file{"history.dat"};
    History hist(hist_file, objects, dim, false);

    auto num1 = pop.num_of_positives();
    auto num2 = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "Num positives:  " << num1;
    std::cout << ", num negatives: " << num2;
    std::cout << " total: " << num3 << '\n';

    std::vector<double> num_e(steps);
    std::vector<double> num_i(steps);
    std::vector<double> num_tot(steps);
    std::vector<double> num_particles_outside(steps - 1);
    std::vector<double> num_injected_particles(steps - 1);

    num_e[0] = pop.num_of_negatives();
    num_i[0] = pop.num_of_positives();
    num_tot[0] = pop.num_of_particles();

    for(std::size_t i = 1; i < steps; ++i)
    {
        auto tot_num0 = pop.num_of_particles();

        timer.tic("move");
        move(pop, dt);
        timer.toc();

        timer.tic("update");
        pop.update(objects, dt);
        timer.toc();

        auto tot_num1 = pop.num_of_particles();
        num_particles_outside[i-1] = tot_num0-tot_num1;
        
        timer.tic("injector");
        inject_particles(pop, species, mesh.exterior_facets, dt);
        timer.toc();

        auto tot_num2 = pop.num_of_particles();
        num_injected_particles[i-1] = tot_num2 - tot_num1;
        num_e[i] = pop.num_of_negatives();
        num_i[i] = pop.num_of_positives();
        num_tot[i] = pop.num_of_particles();

        timer.progress(i, steps, 0, override_status_print);
        std::cout << ", total number of particles: "<<num_tot[i];
    }
    std::cout << '\n';
    timer.summary();

    std::string file_name{"vels_post.txt"};
    pop.save_file(file_name, false);

    std::ofstream out;
    out.open("num_e.txt");
    for (const auto &e : num_e) out << e << "\n";
    out.close();

    out.open("num_i.txt");
    for (const auto &e : num_i)
        out << e << "\n";
    out.close();

    out.open("num_tot.txt");
    for (const auto &e : num_tot)
        out << e << "\n";
    out.close();
    out.open("outside.txt");
    for (const auto &e : num_particles_outside)
        out << e << "\n";
    out.close();
    out.open("injected.txt");
    for (const auto &e : num_injected_particles)
        out << e << "\n";
    out.close();

    return 0;
}
