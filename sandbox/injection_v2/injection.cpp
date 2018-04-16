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

std::vector<std::vector<double>> comb(std::vector<std::vector<double>> vec, double dv)
{
    auto dim = vec.size()/2 + 1;
    auto len = pow(2,dim);
    std::vector<std::vector<double>> arr;
    arr.resize(len, std::vector<double>(dim, 0.0));

    auto plen = int(len/2);

    for (auto i = 0; i < len; ++i)
    {
        for (auto j = 0; j <dim-1; ++j)
        {
            arr[i][j] = vec[i%plen][j];
        }
    }
    for (auto i = 0; i < pow(2, dim-1); ++i)
    {
        arr[i][dim-1] = 0.0;
        arr[pow(2, dim-1)+i][dim-1] = dv;
    }

    return arr;
}

class MaxwellianFlux
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

    MaxwellianFlux(double vth, std::vector<double> &vd,
                   const std::vector<std::vector<double>> &cutoffs,
                   int num_sp,
                   std::vector<Facet> &facets) :
                   dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
    {
        Timer timer;
        double vth2 = vth * vth;
        double factor = (1.0 / (sqrt(2. * M_PI * vth2)));

        auto num_facets = facets.size();
        num_particles.resize(num_facets);
        vdf.resize(num_facets);

        dim = vd.size();
        dv.resize(dim);
        std::vector<double> nsp(3,1.0), diff(dim);

        nsp[0] = num_sp;
        for (auto i = 0; i < dim; ++i)
        {
            diff[i] = cutoffs[i][1] - cutoffs[i][0];
        }
        for (auto i = 1; i < dim; ++i)
        {
            nsp[i] = nsp[i - 1] * diff[i] / diff[i - 1];
        }
        for (auto i = 0; i < dim; ++i)
        {
            dv[i] = diff[i] / nsp[i];
        }

        nbins = std::accumulate(nsp.begin(), nsp.end(), 1, std::multiplies<int>());

        std::vector<std::vector<double>> points, edges{{0.0}, {dv[0]}};
        for (auto i = 1; i < dim; ++i)
        {
            points = comb(edges, dv[i]);
            edges = points;
        }
        num_edges = edges.size();

        int rows, cols, cells;
        rows = (int)nsp[0];
        cols = (int)nsp[1];
        cells = (int)nsp[2];
        int num_bins = rows*cols*cells;
        std::vector<int> indices(dim);

        sp.resize(num_bins*num_edges);
        for (auto i = 0; i < num_bins; ++i)
        {
            indices[0] = i/(cols*cells);
            indices[1] = (i/cells)%cols;
            indices[2] = i - indices[0] * cols*cells - indices[1] * cells;
            for (auto j = 0; j < num_edges; ++j)
            {
                for (auto k = 0; k < dim; ++k)
                {
                    sp[i*num_edges+j].push_back(cutoffs[k][0] + dv[k]*indices[k]+edges[j][k]);
                    // std::cout<<"i: "<<i<<"   "<<sp[i * num_edges +j][k]<<'\n';
                }
            }
        }

        auto vdf_maxwellian = shifted_maxwellian(vth, vd);

        pdf_max.resize(num_facets*nbins);
        vdf.resize(num_facets);
        double vdn, max, value;
        auto t0 = timer.elapsed();
        std::cout << "Initial: " << t0 << '\n';
        timer.reset();
        for(auto i=0; i<num_facets; ++i)
        {
            auto n = facets[i].normal;
            vdn = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
            num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                            exp(-0.5 * (vdn / vth) * (vdn / vth)) +
                            0.5 * vdn * (1. + erf(vdn / (sqrt(2) * vth))));

            vdf[i] = [vdf_maxwellian, n](std::vector<double> &v) {
                    auto vn = std::inner_product(std::begin(n), std::end(n), std::begin(v), 0.0);
                    return (vn > 0.0) * vn * vdf_maxwellian(v);
                };

            for (auto j = 0; j < nbins; ++j)
            {
                max = 0.0;
                for (auto k = 0; k < num_edges; ++k)
                {
                    value = vdf[i](sp[j * num_edges + k]);
                    max = std::max(max, value);
                }
                pdf_max[i*nbins + j] = max;
            }

            auto normalization_factor = std::accumulate(pdf_max.begin()+i*nbins, pdf_max.begin()+(i+1)*nbins, 0.0);

            std::partial_sum(pdf_max.begin()+i*nbins, pdf_max.begin()+(i+1)*nbins, std::back_inserter(cdf));

            std::transform(cdf.begin()+i*nbins, cdf.begin()+(i+1)*nbins, cdf.begin()+i*nbins,
                           std::bind1st(std::multiplies<double>(), 1. / normalization_factor));

        }
        t0 = timer.elapsed();
        std::cout << "Create pdfs: " << t0 << '\n';
    }

    std::vector<double> sample(const int N, int f)
    {
        std::vector<double> vs(N*dim), vs_new(dim);
        int index, n = 0;
        double p_vs, value;
        while (n < N)
        {
            index = std::distance(cdf.begin()+f*nbins,
                        std::lower_bound(cdf.begin()+f*nbins, cdf.begin()+(f+1)*nbins, dist(rng)));

            for (int i = n * dim; i < (n + 1) * dim; ++i)
            {
                vs_new[i%dim] = sp[index*num_edges][i%dim] + dv[i % dim] * dist(rng);
                vs[i] = vs_new[i%dim];
            }
            value = vdf[f](vs_new);
            p_vs = pdf_max[index+f*nbins] * dist(rng);
            n = n + (p_vs<value);
        }
        return vs;
    }
};

int main()
{
    Timer timer;

    std::vector<std::vector<double>> cutoffs{{-6., 6.},{-6., 6.},{-0., 0.}};
    double vth = 1.0;
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

    std::vector<double> vd(dim, 0.0);
    vd[0] = 1;
    // for (std::size_t i = 0; i<dim; ++i)
    // {
    //     vd[i] = 0.0;
    // }
    //--------------------------------------------------------------------------
    // Precalculated stuff
    //--------------------------------------------------------------------------
    int nsp = 50;

    //--------------------------------------------------------------------------
    // Create proposal pdf
    //--------------------------------------------------------------------------
    timer.reset();
    MaxwellianFlux gen(vth,vd,cutoffs,nsp,ext_facets);
    auto t0 = timer.elapsed();
    std::cout << "Creating cdf: " << t0 << '\n';

    //--------------------------------------------------------------------------
    // Sample velocities
    //--------------------------------------------------------------------------
    std::vector<double> v1, v2, v3, v4, v5, v6;
    timer.reset();
    for(std::size_t k = 0; k < num_facets; ++k)
    {
        // std::cout<<"k: "<<k<<'\n';
        int N = 100000;//*num_particles[k];
        auto vs = gen.sample(N, k);
        auto n = ext_facets[k].normal;
        for (std::size_t i = 0; i < N; ++i)
        {
            for (std::size_t j = 0; j < dim; ++j)
            {
                v5.push_back(vs[i * dim + j]);
            }
        }
        if (n[0] >0)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t j = 0; j < dim; ++j)
                {
                    v1.push_back(vs[i * dim + j]);
                }
            }
        }
        if (n[1] > 0)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t j = 0; j < dim; ++j)
                {
                    v2.push_back(vs[i * dim + j]);
                }
            }
        }
        if (n[0] < 0 )
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t j = 0; j < dim; ++j)
                {
                    v3.push_back(vs[i * dim + j]);
                }
            }
        }
        if (n[1] < 0)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t j = 0; j < dim; ++j)
                {
                    v4.push_back(vs[i * dim + j]);
                }
            }
        }
        // if (n[2] > 0 && n[0] == 0.0)
        // {
        //     for (std::size_t i = 0; i < N; ++i)
        //     {
        //         for (std::size_t j = 0; j < dim; ++j)
        //         {
        //             v5.push_back(vs[i * dim + j]);
        //         }
        //     }
        // }
        // if (n[2] < 0 && n[0] == 0.0)
        // {
        //     for (std::size_t i = 0; i < N; ++i)
        //     {
        //         for (std::size_t j = 0; j < dim; ++j)
        //         {
        //             v6.push_back(vs[i * dim + j]);
        //         }
        //     }
        // }
    }
    t0 = timer.elapsed();
    std::cout << "Total time: " << t0 << '\n';

    std::ofstream file;
    file.open("vs1.txt");
    for (const auto &e : v1)
        file << e << "\n";
    file.close();

    file.open("vs2.txt");
    for (const auto &e : v2)
        file << e << "\n";
    file.close();
    file.open("vs3.txt");
    for (const auto &e : v3)
        file << e << "\n";
    file.close();
    file.open("vs4.txt");
    for (const auto &e : v4)
        file << e << "\n";
    file.close();
    file.open("vs5.txt");
    for (const auto &e : v5)
        file << e << "\n";
    file.close();
    // file.open("vs6.txt");
    // for (const auto &e : v6)
    //     file << e << "\n";
    // file.close();
    return 0;
}
