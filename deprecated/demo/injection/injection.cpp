#include <dolfin.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <string>
#include <fstream>
#include <cstdio>
#include <limits>
#include <cmath>
#include <stdlib.h>
#include "../../punc/population.h"
#include "../../punc/injector.h"
#include "../../punc/diagnostics.h"
#include "../../punc/distributor.h"
#include "../../punc/pusher.h"
#include "../../punc/poisson.h"

// #include <functional>
// #include <algorithm>
// #include <memory>
// #include <vector>
// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <limits>
// #include <random>
// #include <math.h>
// #include <stdlib.h>

// #include <iostream>
// #include <algorithm>
// #include <math.h>
// #include <memory>
// #include <vector>
// #include <random>
// #include <functional>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <cassert>
// #include <assert.h>
// #include <limits>

// #include <stdlib.h>
// #include <iostream>
// #include <math.h>
// #include <memory>
// #include <vector>
// #include <algorithm>
// #include <random>
// #include <functional>
// #include <sstream>
// #include <string>
// #include <cassert>
// #include <assert.h>
// #include <chrono>
// #include <ctime>
// #include <fstream>
// #include <cstdio>
// #include <limits>
// #include <cmath>
using namespace punc;

int main()
{
    double dt = 0.1;
    std::size_t steps = 1000;
    std::string fname{"/home/diako/Documents/Software/punc/mesh/2D/nothing_in_square"};
    // std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/injection/mesh/box"};

    auto mesh = load_mesh(fname);
    auto dim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    // df::plot(mesh);
    // df::interactive();
    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_bnd = exterior_boundaries(boundaries, ext_bnd_id);
    std::cout << ext_bnd.size()<< '\n';

    std::vector<double> Ld = get_mesh_size(mesh);
    std::vector<double> vd(dim, 0.0);
    for (std::size_t i = 0; i<dim; ++i)
    {
        vd[i] = 0.0;
    }

    double vth = 1.0;

    int npc = 10000;
    double ne = 100;
    SpeciesList listOfSpecies(mesh, ext_bnd, Ld[0]);

    auto pdf = [](std::vector<double> t)->double{return 1.0;};

    // PhysicalConstants constants;
    // double e = constants.e;
    // double me = constants.m_e;
    // double mi = constants.m_i;
    // listOfSpecies.append(-e, me, ne, npc, vth, vd, pdf, 1.0);
    // listOfSpecies.append(e, mi, 100, npc, vth, vd, pdf, 1.0);

    listOfSpecies.append_raw(-1., 1., ne, npc, vth, vd, pdf, 1.0);

    Population pop(mesh, boundaries);

    load_particles(pop, listOfSpecies);

    std::string file_name1{"/home/diako/Documents/cpp/punc_experimental/demo/injection/vels_pre.txt"};
    pop.save_vel(file_name1);

    auto num1 = pop.num_of_positives();
    auto num2 = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "+:  " << num1 << " -: " << num2 << " total: " << num3 << '\n';

    std::vector<double> num_e(steps);
    std::vector<double> num_i(steps);
    std::vector<double> num_tot(steps);
    std::vector<double> num_particles_outside(steps - 1);
    std::vector<double> num_injected_particles(steps - 1);

    num_e[0] = pop.num_of_negatives();
    num_i[0] = pop.num_of_positives();
    num_tot[0] = pop.num_of_particles();
    Timer timer;
    for(int i=1; i<steps;++i)
    {
        // timer.reset();
        auto tot_num0 = pop.num_of_particles();
        // auto t0 = timer.elapsed();
        // std::cout << "count 1: " << t0 << '\n';
        // timer.reset();
        move(pop, dt);
        // t0 = timer.elapsed();
        // std::cout << "Move: " << t0 << '\n';
        // timer.reset();
        pop.update();
        // t0 = timer.elapsed();
        // std::cout << "Update: " << t0 << '\n';
        // timer.reset();
        auto tot_num1 = pop.num_of_particles();
        // t0 = timer.elapsed();
        // std::cout << "count 2: " << t0 << '\n';
        num_particles_outside[i-1] = tot_num0-tot_num1;
        // timer.reset();
        inject_particles(pop, listOfSpecies, ext_bnd, dt);
        // auto t0 = timer.elapsed();
        // std::cout << "All: " << t0 << '\n';
        // timer.reset();
        auto tot_num2 = pop.num_of_particles();
        num_injected_particles[i-1] = tot_num2 - tot_num1;
        num_e[i] = pop.num_of_negatives();
        num_i[i] = pop.num_of_positives();
        num_tot[i] = pop.num_of_particles();
        std::cout<<"step: "<< i<< ", total number of particles: "<<num_tot[i]<<'\n';
    }

    std::string file_name{"/home/diako/Documents/cpp/punc_experimental/demo/injection/vels_post.txt"};
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
