#include <dolfin.h>
#include <punc.h>

using namespace punc;

using std::cout;
using std::endl;

const bool override_status_print = true;

class LangmuirWave2D : public Pdf
{
  private:
    std::shared_ptr<const df::Mesh> mesh;
    double amplitude, mode;
    const std::vector<double> Ld;
    int dim_;
    std::vector<double> domain_;

  public:
    LangmuirWave2D(std::shared_ptr<const df::Mesh> mesh,
                   double amplitude, double mode,
                   const std::vector<double> &Ld);

    double operator()(const std::vector<double> &x);
    double max() { return 1.0 + amplitude; }
    int dim() { return dim_; }
    std::vector<double> domain() { return domain_; }
};

LangmuirWave2D::LangmuirWave2D(std::shared_ptr<const df::Mesh> mesh,
                               double amplitude, double mode,
                               const std::vector<double> &Ld)
    : mesh(mesh), amplitude(amplitude), mode(mode),
      Ld(Ld)
{
    dim_ = mesh->geometry().dim();
    auto coordinates = mesh->coordinates();
    auto Ld_min = *std::min_element(coordinates.begin(), coordinates.end());
    auto Ld_max = *std::max_element(coordinates.begin(), coordinates.end());
    domain_.resize(2 * dim_);
    for (int i = 0; i < dim_; ++i)
    {
        domain_[i] = Ld_min;
        domain_[i + dim_] = Ld_max;
    }
}

double LangmuirWave2D::operator()(const std::vector<double> &x)
{
    return (locate(mesh, x.data()) >= 0) * (1.0 + amplitude * sin(2 * mode * M_PI * x[0] / Ld[0]));
}

int main()
{
    df::set_log_level(df::WARNING);

    double dt = 0.1;
    std::size_t steps = 30;

    std::string fname{"../../mesh/2D/nothing_in_square"};
    Mesh mesh(fname);

    const std::size_t dim = 2;

    bool remove_null_space = true;
    std::vector<double> Ld = mesh.domain_size();
    std::vector<bool> periodic(dim, true);
    std::vector<double> vd(dim, 0.0);

    auto constr = std::make_shared<PeriodicBoundary>(Ld, periodic);

    auto V = function_space(mesh.mesh, constr);

    PhysicalConstants constants;
    double e = constants.e;
    double me = constants.m_e;
    double mi = constants.m_i;
    double eps0 = constants.eps0;

    auto dv_inv = element_volume(V);

    double vth = 0.0;
    int npc = 64;
    double ne = 100;

    CreateSpecies create_species(mesh, Ld[0]);

    double A = 0.5, mode = 1.0;

    LangmuirWave2D pdfe(mesh.mesh, A, mode, Ld); // Electron position distribution
    UniformPosition pdfi(mesh.mesh);             // Ion position distribution

    Maxwellian vdfe(vth, vd);             // Velocity distribution for electrons
    Maxwellian vdfi(vth, vd);             // Velocity distribution for ions

    bool si_units = true;
    if (!si_units)
    {
        create_species.create(-e, me, ne, pdfe, vdfe, npc);
        create_species.create(e, mi, ne, pdfi, vdfi, npc);
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

        create_species.create_raw(-e, me, ne, pdfe, vdfe, npc);
        create_species.create_raw(e, mi, ne, pdfi, vdfi, npc);
    }

    auto species = create_species.species;

    Population<dim> pop(mesh);

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

    std::vector<std::string> tasks{"distributor", "poisson", "efield", "update", "PE", "accelerator", "move"};
    Timer timer(tasks);

    std::vector<ObjectBC> objects{};
    for (std::size_t i = 1; i < steps; ++i)
    {
        timer.progress(i, steps, 0, override_status_print);

        timer.tic("distributor");
        auto rho = distribute_cg1(V, pop, dv_inv);
        timer.toc();

        timer.tic("poisson");
        auto phi = poisson.solve(rho);
        timer.toc();

        timer.tic("efield");
        auto E = esolver.solve(phi);
        timer.toc();

        timer.tic("PE");
        PE[i - 1] = particle_potential_energy_cg1(pop, phi);
        timer.toc();

        timer.tic("accelerator");
        KE[i - 1] = accel_cg1(pop, E, (1.0 - 0.5 * (i == 1)) * dt);
        timer.toc();

        timer.tic("move");
        move_periodic(pop, dt, Ld);
        timer.toc();

        timer.tic("update");
        pop.update(objects);
        timer.toc();
    }

    if (override_status_print)
    {
        cout << '\n';
    }    

    timer.summary();

    KE[0] = KE0;
    for(std::size_t i=0;i<KE.size(); ++i)
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
