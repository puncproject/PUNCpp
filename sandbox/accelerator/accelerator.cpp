#include <dolfin.h>
#include <punc.h>
#include "EField3D.h"

using namespace punc;

class EField : public df::Expression
{
  public:
    double E0 = 0.007;
    EField() : df::Expression(3) {}
    void eval(df::Array<double> &values, const df::Array<double> &x) const
    {
        values[0] = E0 * (x[0] - 1) / pow(pow((x[0] - 1), 2) + pow((x[1] - 1), 2), 1.5);
        values[1] = E0 * (x[1] - 1) / pow(pow((x[0] - 1), 2) + pow((x[1] - 1), 2), 1.5);
        values[2] = 0.0;
    }
};

int main()
{
    df::set_log_level(df::WARNING);

    Timer timer;

    double dt = 0.01;
    std::size_t steps = 1000;
    int dim = 3;

    std::string fname;
    fname = "../../mesh/3D/nothing_in_cube";

    auto mesh = load_mesh(fname);

    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_bnd = exterior_boundaries(boundaries, ext_bnd_id);

    std::vector<double> Ld = get_mesh_size(mesh);

    PhysicalConstants constants;
    double e = -1.0;//constants.e;
    double me = 5.0;//constants.m_e;
    double eps0 = constants.eps0;

    std::vector<double> x = {1.0, 0.5, 0.5};
    std::vector<double> v = {0.1, 0.0, 0.0};
    auto Np = mesh->num_cells();
    double mul = 1.0;
    for (int i = 0; i < dim; ++i)
    {
        mul *= Ld[i];
        std::cout<<mul<<'\n';

    }
    mul /= Np;
    e *= mul;
    me *= mul;

    Population pop(mesh, boundaries);
    pop.add_particles(x, v, e, me);

    auto W = std::make_shared<EField3D::FunctionSpace>(mesh);
    EField efield;
    df::Function ef(W);
    ef.interpolate(efield);

    auto num1 = pop.num_of_positives();
    auto num2 = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "Num positives:  " << num1;
    std::cout << ", num negatives: " << num2;
    std::cout << " total: " << num3 << '\n';

    std::vector<double> t_accel(steps);

    std::vector<double> pos;
    double KE;
    auto num_cells = pop.num_cells;
    for (int i = 1; i < steps; ++i)
    {
        std::cout << "step: " << i << '\n';
        timer.reset();
        // KE = accel(pop, ef, (1.0 - .5 * (i == 1)) * dt);
        KE = accel_cg(pop, ef, (1.0 - .5 * (i == 1)) * dt);
        t_accel[i] = timer.elapsed();
        move_periodic(pop, dt, Ld);
        pop.update();

        for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
        {
            std::size_t num_particles = pop.cells[cell_id].particles.size();
            for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
            {
                auto particle = pop.cells[cell_id].particles[p_id];
                for (std::size_t j = 0; j < dim; ++j)
                {
                    pos.push_back(pop.cells[cell_id].particles[p_id].x[j]);
                }
            }
        }
    }
    auto tot_time = std::accumulate(t_accel.begin(), t_accel.end(), 0.0);
    
    std::cout << "-----Measured time for the task----------" << '\n';
    std::cout << "Accel:      " << tot_time << '\n';

    std::ofstream out;
    out.open("pos.txt");
    for (const auto &e : pos)
        out << e << "\n";
    out.close();
    return 0;
}
