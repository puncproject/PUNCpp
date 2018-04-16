#include <iostream>
#include <math.h>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <chrono>
#include <dolfin.h>
#include <sstream>
#include <string>
#include <cassert>
#include <assert.h>
#include <ctime>
#include <cstdio>
#include <limits>
#include <cmath>
#include <boost/math/special_functions/erf.hpp>

namespace df = dolfin;

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

class Facet
{
  public:
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;
    std::vector<double> basis;

    Facet(double area,
          std::vector<double> vertices,
          std::vector<double> normal,
          std::vector<double> basis) : area(area),
                                       vertices(vertices),
                                       normal(normal),
                                       basis(basis) {}
};

std::vector<Facet> exterior_boundaries(std::shared_ptr<df::MeshFunction<std::size_t>> boundaries,
                                       std::size_t ext_bnd_id)
{
    auto mesh = boundaries->mesh();
    auto D = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    auto values = boundaries->values();
    auto length = boundaries->size();
    int num_facets = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        if (ext_bnd_id == values[i])
        {
            num_facets += 1;
        }
    }
    df::SubsetIterator facet_iter(*boundaries, ext_bnd_id);
    std::vector<Facet> facet_vec;
    double area;
    std::vector<double> normal(D);
    std::vector<double> vertices(D * D);
    std::vector<double> basis(D * D);
    std::vector<double> vertex(D);
    double norm;
    mesh->init(tdim - 1, tdim);
    for (; !facet_iter.end(); ++facet_iter)
    {
        // area
        df::Cell cell(*mesh, facet_iter->entities(tdim)[0]);
        auto cell_facet = cell.entities(tdim - 1);
        std::size_t num_facets = cell.num_entities(tdim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            if (cell_facet[i] == facet_iter->index())
            {
                area = cell.facet_area(i);
            }
        }
        // vertices
        const unsigned int *facet_vertices = facet_iter->entities(0);
        for (std::size_t i = 0; i < D; ++i)
        {
            const df::Point p = mesh->geometry().point(facet_vertices[i]);
            for (std::size_t j = 0; j < D; ++j)
            {
                vertices[i * D + j] = p[j];
            }
        }
        //normal
        df::Facet facet(*mesh, facet_iter->index());
        for (std::size_t i = 0; i < D; ++i)
        {
            normal[i] = -1 * facet.normal(i);
            basis[i*D] = normal[i];
        }
        // basis
        norm = 0.0;
        for (std::size_t i = 0; i < D; ++i)
        {
            vertex[i] = vertices[i] - vertices[D + i];
            norm += vertex[i] * vertex[i];
        }
        for (std::size_t i = 0; i < D; ++i)
        {
            vertex[i] /= sqrt(norm);
            basis[i * D + 1] = vertex[i];
        }

        if (D == 3)
        {
            basis[2] = normal[1] * vertex[2] - normal[2] * vertex[1];
            basis[5] = normal[0] * vertex[2] - normal[2] * vertex[0];
            basis[8] = normal[0] * vertex[1] - normal[1] * vertex[0];
        }
        facet_vec.push_back(Facet(area, vertices, normal, basis));
    }
    return facet_vec;
}

std::shared_ptr<const df::Mesh> load_mesh(std::string fname)
{
    auto mesh = std::make_shared<const df::Mesh>(fname + ".xml");
    return mesh;
}

std::shared_ptr<df::MeshFunction<std::size_t>> load_boundaries(std::shared_ptr<const df::Mesh> mesh, std::string fname)
{
    auto boundaries = std::make_shared<df::MeshFunction<std::size_t>>(mesh, fname + "_facet_region.xml");
    return boundaries;
}

std::vector<std::size_t> get_mesh_ids(std::shared_ptr<df::MeshFunction<std::size_t>> boundaries)
{
    auto values = boundaries->values();
    auto length = boundaries->size();
    std::vector<std::size_t> tags(length);

    for (std::size_t i = 0; i < length; ++i)
    {
        tags[i] = values[i];
    }
    std::sort(tags.begin(), tags.end());
    tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    return tags;
}

std::function<double(std::vector<double> &)> shifted_maxwellian(double vth, std::vector<double> &vd)
{
    auto dim = vd.size();
    auto pdf = [vth, vd, dim](std::vector<double> &v) {
        double v_sqrt = 0.0;
        for (auto i = 0; i <dim; ++i)
        {
            v_sqrt += (v[i] - vd[i]) * (v[i] - vd[i]);
        }
        return (1.0/(pow(sqrt(2.*M_PI*vth*vth), dim)))*exp(-0.5 * v_sqrt / (vth * vth));
    };

    return pdf;
}

enum VDFType {Generic, Maxwellian};

class Flux
{
public:
    std::vector<double> num_particles;
    // virtual void create_flux_vdf(){};
    virtual std::vector<double> sample(const std::size_t N, const std::size_t f){};
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
    // std::vector<double> num_particles;
    // MaxwellianFlux();

    MaxwellianFlux(double vth, std::vector<double> &vd, std::vector<Facet> &facets);
                   // std::vector<double> &cutoffs, int nsp,
                   // std::vector<Facet> &facets);
    // void create_flux() override;
    std::vector<double> sample(const std::size_t N, const std::size_t f) override;
};

MaxwellianFlux::MaxwellianFlux(double vth, std::vector<double> &vd,
                               std::vector<Facet> &facets)
                               : facets(facets), dist(0.0, 1.0),
                                 rng{random_seed_seq::get_instance()}
{
    if (vth == 0.0)
    {
        vth = std::numeric_limits<double>::epsilon();
    }
    nsp = 60;
    std::vector<double> cutoffs = {*std::max_element(vd.begin(), vd.end()) - 5 * vth,
                                   *std::max_element(vd.begin(), vd.end()) + 5 * vth};
    double vth2 = vth * vth;
    double factor = (1.0 / (sqrt(2. * M_PI * vth2)));
    auto num_facets = facets.size();
    Flux::num_particles.resize(num_facets);
    vdf.resize(num_facets);
    maxwellian.resize(num_facets);

    dim = vd.size();
    v0 = cutoffs[0];
    dv = (cutoffs[1] - cutoffs[0]) / nsp;
    std::vector<double> vdfv(nsp);

    for (auto i = 0; i < num_facets; ++i)
    {
        auto n = facets[i].normal;
        std::vector<double> vdn(dim);
        vdn[0] = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
        for (int j = 1; j < dim; ++j)
        {
            for (int k = 0; k < dim; ++k)
            {
                vdn[j] += facets[i].basis[k * dim + j] * vd[k];
            }
        }

        Flux::num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                           exp(-0.5 * (vdn[0] / vth) * (vdn[0] / vth)) +
                           0.5 * vdn[0] * (1. + erf(vdn[0] / (sqrt(2) * vth))));

        vdf[i] = [vth2, vdn, factor](double v) {
            return (v>0)*v*factor* exp(-0.5 * (v - vdn[0]) * (v - vdn[0]) / vth2);
        };

        for (auto j = 0; j < nsp; ++j)
        {
            vdfv[j] = vdf[i](v0 + j * dv);
        }

        std::transform(vdfv.begin(), vdfv.end() - 1, vdfv.begin() + 1,
                       std::back_inserter(pdf_max),
                       [](double a, double b) { return std::max(a, b); });

        auto normalization_factor = std::accumulate(pdf_max.begin() + i * (nsp - 1),
                                                    pdf_max.begin() + (i + 1) * (nsp - 1),
                                                    0.0);

        std::partial_sum(pdf_max.begin() + i * (nsp - 1),
                         pdf_max.begin() + (i + 1) * (nsp - 1),
                         std::back_inserter(cdf));

        std::transform(cdf.begin() + i * (nsp - 1),
                       cdf.begin() + (i + 1) * (nsp - 1),
                       cdf.begin() + i * (nsp - 1),
                       std::bind1st(std::multiplies<double>(),
                       1. / normalization_factor));

        maxwellian[i] = [vth, vdn](double v, int k){
            return vdn[k] - sqrt(2.0) * vth * boost::math::erfc_inv(2 * v);
        };

    }
}

std::vector<double> MaxwellianFlux::sample(const std::size_t N, const std::size_t f)
{
    random_source rng(random_seed_seq::get_instance());
    std::vector<double> vs(N * dim), vs_new(dim);
    int index, n = 0;
    double p_vs, value;
    while (n < N)
    {
        index = std::distance(cdf.begin() + f * (nsp - 1),
                std::lower_bound(cdf.begin() + f * (nsp - 1),
                                 cdf.begin() + (f + 1) * (nsp - 1),
                                 dist(rng)));

        vs_new[0] = v0 + dv * (index + dist(rng));
        value = vdf[f](vs_new[0]);
        p_vs = pdf_max[index + f * (nsp - 1)] * dist(rng);
        if (p_vs < value)
        {
            for (int k = 1; k < dim; ++k)
            {
                auto r = dist(rng);
                vs_new[k] = maxwellian[f](r, k);
            }
            for (int i = 0; i < dim; ++i)
            {
                for (int j = 0; j < dim; ++j)
                {
                    vs[n * dim + i] += facets[f].basis[i * dim + j] * vs_new[j];
                }
            }
            n += 1;
        }
    }
    return vs;
}

class Species
{
  public:
    double q;
    double m;
    double n;
    int num;
    double vth;
    std::vector<double> vd;
    std::function<double(std::vector<double> &)> pdf;
    double pdf_max;

    // VDFType vdf_type;
    std::unique_ptr<Flux> flux;
    // Flux flux;

    Species(double q, double m, double n, int num, double vth,
            std::vector<double> &vd,
            std::function<double(std::vector<double> &)> pdf,
            double pdf_max, std::vector<Facet> &facets,
            VDFType vdf_type=VDFType::Maxwellian) : q(q), m(m), n(n), num(num),
            vth(vth), vd(vd), pdf(pdf), pdf_max(pdf_max)
    {
        if (vdf_type==VDFType::Maxwellian)
        {
            // MaxwellianFlux flux(vth, vd, facets);
            // this->flux = flux;
            printf("%s\n", "Maxwellian");
            this->flux = std::make_unique<MaxwellianFlux>(vth, vd, facets);
        }
    }
};

int main()
{
    Timer timer;
    std::string fname{"/home/diako/Documents/cpp/punc_experimental/prototypes/mesh/square"};
    // std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/injection/mesh/box"};

    auto mesh = load_mesh(fname);
    auto boundaries = load_boundaries(mesh, fname);
    auto dim = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    printf("gdim: %zu, tdim: %zu \n", dim, tdim);
    // df::plot(mesh);
    // df::interactive();
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_facets = exterior_boundaries(boundaries, ext_bnd_id);
    auto num_facets = ext_facets.size();

    double vth = 100.0;
    std::vector<double> vd(dim, 0.0);
    double q=1.0;
    double m=1.0;
    double n=1000.0;
    int num = 10;

    auto pdf = [](std::vector<double> t)->double{return 1.0;};
    double pdf_max = 1.0;

    std::vector<Species> list;
    Species sps1(q,m,n,num,vth,vd,pdf,pdf_max,ext_facets);
    list.push_back(std::move(sps1));
    Species sps2(q,m,n,num,vth,vd,pdf,pdf_max,ext_facets);
    list.push_back(std::move(sps2));
    auto numps = list[0].flux->num_particles[0];
    printf("%e\n", numps);
    return 0;
}
