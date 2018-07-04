#ifndef INJECTOR_H
#define INJECTOR_H

#include <dolfin.h>
#include <boost/math/special_functions/erf.hpp>
#include <chrono>
#include <random>

namespace punc
{

namespace df = dolfin;

class Population;
class Species;

class Timer
{
  public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

  private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

struct random_seed_seq
{
    template <typename It>
    void generate(It begin, It end)
    {
        for (; begin != end; ++begin)
        {
            *begin = device();
        }
    }

    static random_seed_seq &get_instance()
    {
        static thread_local random_seed_seq result;
        return result;
    }

  private:
    std::random_device device;
};

std::vector<std::vector<double>> combinations(std::vector<std::vector<double>> vec, double dv);

struct Facet
{
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;
    std::vector<double> basis;
};

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id);

class ORS
{
public:
    std::function<double(std::vector<double> &)> vdf;
    int dim, nbins, num_edges;
    std::vector<double> dv;
    std::vector<std::vector<double>> sp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    ORS(double vth, std::vector<double> &vd,
        std::function<double(std::vector<double> &)> vdf, int num_sp=60);
    std::vector<double> sample(const std::size_t N);
};

enum VDFType {Generic, Maxwellian};

class Flux
{
public:
    std::vector<double> num_particles;
    virtual std::vector<double> sample(const std::size_t N, const std::size_t f);
};

class GenericFlux
{
public:

    int dim, nbins, num_edges;
    std::vector<double> dv;
    std::vector<std::vector<double>> sp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;
    std::vector<std::function<double(std::vector<double> &)>> vdf;
    std::vector<double> num_particles;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    GenericFlux();
    GenericFlux(double vth, std::vector<double> &vd,
        const std::vector<std::vector<double>> &cutoffs,
        int num_sp,
        std::vector<Facet> &facets);
    std::vector<double> sample(const std::size_t N, const std::size_t f);
};

class MaxwellianFlux : public Flux
{
private:
    std::vector<Facet> facets;

    int nsp;
    int dim;
    double v0, dv;

    std::vector<double> pdf_max;
    std::vector<double> cdf;
    std::vector<std::function<double(double)>> vdf;
    std::vector<std::function<double(double, int)>> maxwellian;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    distribution dist;
    random_source rng;

public:
    MaxwellianFlux(double vth, std::vector<double> &vd, std::vector<Facet> &facets);
    std::vector<double> sample(const std::size_t N, const std::size_t f) override;
};

signed long int locate(std::shared_ptr<const df::Mesh> mesh, std::vector<double> x);

std::function<double(std::vector<double> &)> create_mesh_pdf(std::function<double(std::vector<double> &)> pdf,
                                                             std::shared_ptr<const df::Mesh> mesh);

std::vector<double> random_domain_points(
    std::function<double(std::vector<double> &)> pdf,
    double pdf_max, int N,
    std::shared_ptr<const df::Mesh> mesh);

std::vector<double> random_facet_points(const int N, std::vector<double> &facet_vertices);

std::vector<double> maxwellian(double vth, std::vector<double> vd, const int &N);

std::function<double(std::vector<double> &)> maxwellian_vdf(double vth, std::vector<double> &vd);

void inject_particles(Population &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt);

void load_particles(Population &pop, std::vector<Species> &species);

}

#endif
