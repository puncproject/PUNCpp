#include <dolfin.h>
#include <punc.h>

using namespace punc;

int main()
{
    df::set_log_level(df::WARNING);

    double dt = 0.1;
    std::size_t steps = 100;
    int gdim = 2;

    std::string fname;
    if (gdim==2)
    {
        fname = "../../mesh/2D/nothing_in_square";
    }else if (gdim==3){
        fname = "../../mesh/3D/nothing_in_cube";
    }

    auto mesh = load_mesh(fname);
    auto dim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_bnd = exterior_boundaries(boundaries, ext_bnd_id);

    std::vector<double> Ld = get_mesh_size(mesh);
    std::vector<double> vd(dim, 0.0);
    for (std::size_t i = 0; i<dim; ++i)
    {
        vd[i] = 0.0;
    }

    double vth = 1.0;

    int npc = 100;
    double ne = 100;
    CreateSpecies create_species(mesh, ext_bnd, Ld[0]);

    auto pdf = [](std::vector<double> t)->double{return 1.0;};

    // PhysicalConstants constants;
    // double e = constants.e;
    // double me = constants.m_e;
    // double mi = constants.m_i;
    // create_species.create(-e, me, ne, npc, vth, vd, pdf, 1.0);
    // create_species.create(e, mi, 100, npc, vth, vd, pdf, 1.0);

    create_species.create_raw(-1., 1., ne, npc, vth, vd, pdf, 1.0);

    auto species = create_species.species;
    Population pop(mesh, boundaries);

    load_particles(pop, species);

    std::string file_name1{"vels_pre.txt"};
    pop.save_vel(file_name1);

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

    for(int i=1; i<steps;++i)
    {
        auto tot_num0 = pop.num_of_particles();
        move(pop, dt);
        pop.update();
        auto tot_num1 = pop.num_of_particles();
        num_particles_outside[i-1] = tot_num0-tot_num1;
        inject_particles(pop, species, ext_bnd, dt);
        auto tot_num2 = pop.num_of_particles();
        num_injected_particles[i-1] = tot_num2 - tot_num1;
        num_e[i] = pop.num_of_negatives();
        num_i[i] = pop.num_of_positives();
        num_tot[i] = pop.num_of_particles();
        std::cout<<"step: "<< i<< ", total number of particles: "<<num_tot[i]<<'\n';
    }

    std::string file_name{"vels_post.txt"};
    pop.save_vel(file_name);

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
