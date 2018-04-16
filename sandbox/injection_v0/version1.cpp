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

class MaxwellianFlux
{
public:
    double v0, dv; 
    int nsp;

    std::vector<double> pdf_max;
    std::vector<double> cdf;

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;

    MaxwellianFlux(double vth, const std::vector<double> & vd, 
                   const std::vector<double> &cutoffs, int nsp,
                   const std::vector<Facet> &facets) : nsp(nsp), 
                   dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
    {

        double vth2 = vth * vth;
        double factor = (1.0 / (sqrt(2. * M_PI * vth2)));

        auto num_facets = facets.size();
        std::vector<double> num_particles(num_facets);
        std::vector<std::function<double(double)>> vdf(num_facets);
        std::vector<std::function<double(double, int)>> maxwellian(num_facets);

        auto dim = vd.size();
        double v0 = cutoffs[0];
        double dv = (cutoffs[1] - cutoffs[0]) / nsp;
        std::vector<double> vdfv(nsp);
        std::vector<double> vdn(dim);
        for(auto i=0; i<num_facets; ++i)
        {
            auto n = facets[i].normal;
            vdn[0] = std::inner_product(n.begin(), n.end(), vd.begin(), 0.0);
            for(int j=1; j<dim; ++j)
            {
                for (int k = 0; k < dim; ++k)
                {
                    vdn[j] += facets[i].basis[k * dim + j]*vd[k];
                }
            }
            
            num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                            exp(-0.5 * (vdn[0] / vth) * (vdn[0] / vth)) +
                            0.5 * vdn[0] * (1. + erf(vdn[0] / (sqrt(2) * vth))));

            vdf[i] = [vth2, vdn, factor](double v) {
                return v*exp(-0.5 * (v - vdn[0]) * (v - vdn[0]) / (vth2));
            };

            for (auto j = 0; j < nsp; ++j)
            {
                vdfv[j] = vdf[i](v0 + j * dv);
            }
            std::transform(vdfv.begin(), vdfv.end() - 1, vdfv.begin() + 1,
                           std::back_inserter(pdf_max), [](double a, double b) { return std::max(a, b); });

            auto normalization_factor = std::accumulate(pdf_max.begin()+i*(nsp-1), pdf_max.begin()+(i+1)*(nsp-1), 0.0);
            std::partial_sum(pdf_max.begin() + i * (nsp-1), pdf_max.begin() + (i + 1) * (nsp-1), cdf.begin() + i * (nsp-1));

            std::transform(cdf.begin() + i * (nsp-1), cdf.begin() + (i + 1) * (nsp-1), cdf.begin() + i*(nsp-1),
                           std::bind1st(std::multiplies<double>(), 1. / normalization_factor));

            maxwellian[i] = [vth, vdn](double v, int k) { return vdn[k] - sqrt(2.0) * vth * erfc_inv(2 * v); };
        }

    }

    std::vector<double> sample(const int N, int f)
    {
        std::vector<double> vs(N*dim); 
        int i, index, n = 0;
        double p_vs, value, vs_new;
        while (n < N)
        {
            index = std::distance(cdf.begin()+f*(nsp-1), 
                        std::lower_bound(cdf.begin()+f*(nsp-1), cdf.begin()+(f+1)*(nsp-1), dist(rng)));

            vs_new[0] = v0 + dv*(index + dist(rng));
            value = vdf[f](vs_new[0]);
            p_vs = pdf_max[index+f*(nsp-1)] * dist(rng);
            if(p_vs<value)
            {
                for (int k = 1; k < dim; ++k)
                {
                    vs_new[k] = maxwellian[f](dist(rng), k);
                }
                for (int i = 1; i < dim; ++i)
                {
                    for (int j = 1; j < dim; ++j)
                    {
                        vs[n*dim + i] += facets[f].basis[i*dim+j]*vs_new[j]
                    }
                }
                n += 1;
            }
        }
        return vs;
    }
};

int main()
{
    Timer timer;

    std::vector<double> cutoffs{-6., 6.};
    double vth = 1.0;
    std::vector<double> vd{0.0};

    std::string fname{"/home/diako/Documents/cpp/punc_experimental/prototypes/mesh/square"};
    // std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/injection/mesh/box"};

    auto mesh = load_mesh(fname);
    auto boundaries = load_boundaries(mesh, fname);
    auto gdim = mesh.geometry().dim();
    auto tdim = mesh.topology().dim();
    printf("gdim: %zu, tdim: %zu \n", gdim, tdim);
    // df::plot(mesh);
    // df::interactive();
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto ext_facets = exterior_boundaries(boundaries, ext_bnd_id);
    
    //--------------------------------------------------------------------------
    // Precalculated stuff
    //--------------------------------------------------------------------------
    int N = 10000000;
    int dim = 1;
    int nsp = 50;
    double v0 = cutoffs[0];
    double dv = (cutoffs[1]-cutoffs[0])/nsp;    

    //--------------------------------------------------------------------------
    // Create proposal pdf
    //--------------------------------------------------------------------------
    timer.reset();
    ORS ors(vdf_flux, v0, dv, nsp);
    auto t0 = timer.elapsed();
    std::cout << "Creating cdf: " << t0 << '\n';
    timer.reset();

    //--------------------------------------------------------------------------
    // Sample velocities
    //--------------------------------------------------------------------------
    auto vs  = ors.sample(N);
    t0 = timer.elapsed();
    std::cout << "Sampling time: " << t0 << '\n'; 
    std::cout << "Number of samples: " << vs.size() << '\n';

    std::ofstream file;
    file.open("vs.txt");
    for (const auto &e : vs)
        file << e << "\n";
    file.close();
    return 0;
}