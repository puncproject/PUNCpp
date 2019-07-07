#include <dolfin.h>
#include <punc.h>
#include "../../punc/ufl/EField3D.h"
using namespace punc;

const bool override_status_print = true;

class Gif
{
  public:
    std::ofstream ofile;

    Gif(const std::string &fname)
    {
        ofile.open(fname, std::ofstream::out);
    }

    ~Gif() { ofile.close(); };

    template <typename PopulationType>
    void save(PopulationType &pop)
    {
        auto dim = pop.g_dim;
        for (auto &cell : pop.cells)
        {
            for (auto &particle : cell.particles)
            {
                for(std::size_t i = 0; i<dim; ++i)
                {
                    ofile << particle.x[i] << "\t";
                }
                ofile << std::endl;
            }
        }
    }
};

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

class BField : public df::Expression
{
  public:
    double B0 = 1.0;
    BField():df::Expression(3){}
    void eval(df::Array<double> &values, const df::Array<double> &x) const
    {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = B0 * pow(pow((x[0] - 1), 2) + pow((x[1] - 1), 2), 0.5);
    }
};

int main()
{
    df::set_log_level(df::WARNING);

    const char *fname_position = "position.txt";

    const std::size_t dim = 3;
    double dt = 0.01;
    std::size_t steps = 4200;

    std::string fname{"empty_cube"};
    Mesh mesh(fname);
    std::vector<double> Ld = mesh.domain_size();
    
    auto W = std::make_shared<EField3D::FunctionSpace>(mesh.mesh);

    Population<dim> pop(mesh);

    double q = -1.0;
    double m = 0.05;
    std::vector<double> x = {1.0,0.5,0.5};
    std::vector<double> v = {0.1,0.0,0.0};
    auto Np = mesh.mesh->num_cells();
    double mul = 1.0;
    for(std::size_t i=0; i<dim; ++i)
    {
        mul *= Ld[i];
    }
    mul /= Np;
    q *= mul;
    m *= mul;
    pop.add_particles(x, v, q, m);

    EField efield;
    BField bfield;
    df::Function ef(W);
    df::Function bf(W);
    ef.interpolate(efield);
    bf.interpolate(bfield);

     std::vector<std::shared_ptr<Object>> objects = {};

    auto num1 = pop.num_of_positives();
    auto num2 = pop.num_of_negatives();
    auto num3 = pop.num_of_particles();
    std::cout << "Num positives:  " << num1;
    std::cout << ", num negatives: " << num2;
    std::cout << " total: " << num3 << '\n';

    Gif gif(fname_position);
    gif.save(pop);

    std::vector<std::string> tasks{"boris", "move", "update", "io"};
    Timer timer(tasks);

    for(std::size_t i=1; i<steps ;++i)
    {
        timer.progress(i, steps, 0, override_status_print);

        timer.tic("boris");
        boris(pop, ef, bf, (1.0-.5*(i==1))*dt);
        timer.toc();

        timer.tic("move");
        move_periodic(pop, dt, Ld);
        timer.toc();

        timer.tic("update");
        pop.update(objects, dt);
        timer.toc();

        timer.tic("io");
        gif.save(pop);
        timer.toc();
    }

    if (override_status_print)
    {
        std::cout << '\n';
    }

    timer.summary();
    return 0;
}
