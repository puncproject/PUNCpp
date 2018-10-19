#include <dolfin.h>
#include <punc.h>

using namespace punc;

int main()
{
    df::set_log_level(df::WARNING);

    std::vector<std::string> tasks{"move", "update", "injector"};
    Timer timer(tasks);

    const int dim = 2;

    std::string fname;
    if (dim==2)
    {
        fname = "../../mesh/2D/nothing_in_square";
    }else if (dim==3){
        fname = "../../mesh/3D/nothing_in_cube";
    }

    auto mesh = load_mesh(fname);

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_bnd = exterior_boundaries(boundaries, ext_bnd_id);
    std::cout<<"num_facets = "<<ext_bnd.size()<<'\n';
    std::vector<double> Ld = get_mesh_size(mesh);

    PhysicalConstants constants;
    double e = constants.e;
    double me = constants.m_e;
    // double mi = constants.m_i;
    // double kB = constants.k_B;
    double eps0 = constants.eps0;

    int npc = 16;
    double ne = 1.0e10;
    double debye = 1.0;
    double wpe = sqrt(ne * e * e / (eps0 * me));

    double vthe = debye * wpe;
    std::vector<double> vd(dim, 0.0);

    UniformPosition pdf(mesh); // Position distribution
    // Maxwellian vdf(vthe, vd);  // Maxwellian velocity distribution
    //Kappa vdf(vthe, vd, 3.0);  // Kappa velocity distribution
    Cairns vdf(vthe, vd, 0.2); // Cairns velocity distribution
    // KappaCairns vdf(vthe, vd, 4.0, 0.2); // Kappa-Cairns velocity distribution

    std::size_t steps = 100;
    double dt = 0.05;

    CreateSpecies create_species(mesh);

    bool si_units = true;
    if (!si_units)
    {
        create_species.X = Ld[0];
        create_species.create(-e, me, ne, pdf, vdf, npc);
        eps0 = 1.0;
    }
    else
    {
        dt /= wpe;
        create_species.create_raw(-e, me, ne, pdf, vdf, npc);
    }

    auto species = create_species.species;

    create_flux(species, ext_bnd);

    Population<dim> pop(mesh, boundaries);

    load_particles(pop, species);

    std::string file_name1{"vels_pre.txt"};
    pop.save_file(file_name1, false);

    std::vector<ObjectBC> objects{};
    History hist("history.dat", objects, false);

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
        pop.update(objects);
        timer.toc();

        auto tot_num1 = pop.num_of_particles();
        num_particles_outside[i-1] = tot_num0-tot_num1;
        
        timer.tic("injector");
        inject_particles(pop, species, ext_bnd, dt);
        timer.toc();

        auto tot_num2 = pop.num_of_particles();
        num_injected_particles[i-1] = tot_num2 - tot_num1;
        num_e[i] = pop.num_of_negatives();
        num_i[i] = pop.num_of_positives();
        num_tot[i] = pop.num_of_particles();

        std::cout<<"step: "<< i<< ", total number of particles: "<<num_tot[i]<<'\n';
    }

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
