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


template <typename T>
std::vector<double> linspace(const T &start_in, const T &end_in, const int &num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);

    return linspaced;
}

std::vector<std::vector<double>> cartesian(std::vector<std::vector<double>> &v)
{
    auto product = [](long long a, std::vector<double> &b) { return a * b.size(); };
    const long long N = std::accumulate(v.begin(), v.end(), 1LL, product);
    std::vector<std::vector<double>> u(N, std::vector<double>(v.size()));
    for (long long n = 0; n < N; ++n)
    {
        std::lldiv_t q{n, 0};
        for (long long i = v.size() - 1; 0 <= i; --i)
        {
            q = std::div(q.quot, v[i].size());
            u[n][i] = v[i][q.rem];
        }
    }
    return u;
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

class ORS
{
public:
    std::function<double(std::vector<double> &)> pdf;
    std::vector<double> dv;
    std::vector<std::vector<double>> mesh_grid;
    std::vector<double> pdf_max;
    std::vector<double> weights;
    int dim;
    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng;
    distribution dist;
    ORS(std::function<double(std::vector<double> &)> pdf,
        std::vector<std::vector<double>> cutoffs, int num_sp):pdf(pdf),
        dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
        {

            Timer timer;
            timer.reset();
            dim = cutoffs.size();
            std::vector<double> nsp(dim);
            nsp[0] = num_sp;

            std::vector<double> diff(dim);
            dv.resize(dim);

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

            std::vector<std::vector<double>> s_points;
            for (auto i = 0; i < dim; ++i)
            {
                auto vec = linspace(cutoffs[i][0], cutoffs[i][1] - dv[i], nsp[i]);
                s_points.push_back(vec);
            }
            auto t0 = timer.elapsed();
            std::cout << "sp: " << t0 << '\n';
            timer.reset();
            mesh_grid = cartesian(s_points);
            t0 = timer.elapsed();
            std::cout << "meshgrid: " << t0 << '\n';
            timer.reset();
            auto volume = std::accumulate(std::begin(dv), std::end(dv), 1.0, std::multiplies<double>());
            std::vector<std::vector<double>> points, sp{{0.0}, {dv[0]}};
            for (auto i = 1; i < dim; ++i)
            {
                points = comb(sp, dv[i]);
                sp = points;
            }
            t0 = timer.elapsed();
            std::cout << "comb: " << t0 << '\n';
            timer.reset();
            auto size_sp = sp.size();
            double max;
            std::vector<double> vertex(dim);

            auto num_bins = mesh_grid.size();
            pdf_max.resize(num_bins);
            std::vector<double> pdf_sum(num_bins);
            for (auto i = 0; i < num_bins; ++i)
            {
                max = 0.0;
                for (auto j = 0; j < size_sp; ++j)
                {
                    for (auto k = 0; k < dim; ++k)
                    {
                        vertex[k] = mesh_grid[i][k] + sp[j][k];
                    }
                    if (pdf(vertex) > max)
                    {
                        max = pdf(vertex);
                    }
                }
                pdf_max[i] = max;
                pdf_sum[i] = max * volume;
            }
            t0 = timer.elapsed();
            std::cout << "proposal pdf: " << t0 << '\n';
            timer.reset();
            auto integral = std::accumulate(pdf_sum.begin(), pdf_sum.end(), 0.0);
            t0 = timer.elapsed();
            std::cout << "integral: " << t0 << '\n';
            timer.reset();
            std::vector<double> w_integral(num_bins);
            weights.resize(num_bins);
            for (auto i = 0; i < num_bins; ++i)
            {
                w_integral[i] = pdf_sum[i] / integral;
            }
            t0 = timer.elapsed();
            std::cout << "pdf: " << t0 << '\n';
            timer.reset();
            std::partial_sum(w_integral.begin(), w_integral.end(), weights.begin());
            t0 = timer.elapsed();
            std::cout << "weights: " << t0 << '\n';
            timer.reset();
        }

        std::vector<double> sample(int N)
        {
            // Timer timer;
            std::vector<double> vs(N * dim), vs_new(dim);
            // std::vector<double> t1,t2,t3;
            int n = 0;
            double p_vs;
            int rej=0;
            while (n < N)
            {
                // timer.reset();
                auto ind = std::distance(weights.begin(), std::lower_bound(weights.begin(), weights.end(), dist(rng)));
                // t1.push_back(timer.elapsed());
                // timer.reset();
                for (int j = 0; j < dim; j++)
                {
                    vs_new[j] = mesh_grid[ind][j] + dv[j] * dist(rng);
                }
                // t2.push_back(timer.elapsed());
                // timer.reset();
                p_vs = pdf_max[ind] * dist(rng);
                if (p_vs < pdf(vs_new))
                {
                    for (int i = n * dim; i < (n + 1) * dim; ++i)
                    {
                        vs[i] = vs_new[i % dim];
                    }
                    n += 1;
                }else{
                    rej +=1;
                }
                // t3.push_back(timer.elapsed());
                // timer.reset();
            }
            std::cout<<"Number of rejections: "<<rej<<'\n';
            // auto _t1 = std::accumulate(t1.begin(), t1.end(), 0.0);
            // auto _t2 = std::accumulate(t2.begin(), t2.end(), 0.0);
            // auto _t3 = std::accumulate(t3.begin(), t3.end(), 0.0);
            // std::cout << "lower bound: " << _t1 << '\n';
            // std::cout << "index: " << _t2 << '\n';
            // std::cout << "vs new: " << _t3 << '\n';
            return vs;
        }
};

std::vector<double> ORS1(std::function<double(std::vector<double> &)> pdf,
                        std::vector<std::vector<double>> cutoffs, int num_sp, int N)
{

    auto dim = cutoffs.size();
    std::vector<double> nsp(dim);
    nsp[0] = num_sp;

    std::vector<double> diff(dim), dv(dim);
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

    std::vector<std::vector<double>> s_points;
    for (auto i = 0; i < dim; ++i)
    {
        auto vec = linspace(cutoffs[i][0], cutoffs[i][1] - dv[i], nsp[i]);
        s_points.push_back(vec);
    }

    auto mesh_grid = cartesian(s_points);

    auto volume = std::accumulate(std::begin(dv), std::end(dv), 1.0, std::multiplies<double>());
    std::vector<std::vector<double>> points, sp{{0.0}, {dv[0]}};
    for (auto i = 1; i < dim; ++i)
    {
        points = comb(sp, dv[i]);
        sp = points;
    }

    auto size_sp = sp.size();
    double max;
    std::vector<double> vertex(dim);

    auto num_bins = mesh_grid.size();
    std::vector<double> pdf_max(num_bins);
    std::vector<double> pdf_sum(num_bins);
    for (auto i = 0; i < num_bins; ++i)
    {
        max = 0.0;
        for (auto j = 0; j < size_sp; ++j)
        {
            for (auto k = 0; k < dim; ++k)
            {
                vertex[k] = mesh_grid[i][k] + sp[j][k];
            }
            if (pdf(vertex) > max)
            {
                max = pdf(vertex);
            }
        }
        pdf_max[i] = max;
        pdf_sum[i] = max * volume;
    }

    auto integral = std::accumulate(pdf_sum.begin(), pdf_sum.end(), 0.0);

    std::vector<double> w_integral(num_bins), weights(num_bins);
    for (auto i = 0; i < num_bins; ++i)
    {
        w_integral[i] = pdf_sum[i] / integral;
    }

    std::partial_sum(w_integral.begin(), w_integral.end(), weights.begin());

    typedef std::mt19937_64 random_source;
    typedef std::uniform_real_distribution<double> distribution;
    random_source rng{random_seed_seq::get_instance()};
    distribution dist(0.0, 1.0);

    std::vector<double> vs(N * dim), vs_new(dim);
    int n = 0;
    double p_vs;
    while (n < N)
    {
        auto index = std::lower_bound(weights.begin(), weights.end(), dist(rng));
        auto ind = std::distance(weights.begin(), index);

        for (int j = 0; j < dim; j++)
        {
            vs_new[j] = mesh_grid[ind][j] + dv[j] * dist(rng);
        }
        p_vs = pdf_max[ind] * dist(rng);
        if (p_vs < pdf(vs_new))
        {
            for (int i = n * dim; i < (n + 1) * dim; ++i)
            {
                vs[i] = vs_new[i % dim];
            }
            n += 1;
        }
    }
    return vs;
}

class Facet
{
  public:
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;

    Facet(double area,
          std::vector<double> vertices,
          std::vector<double> normal) : area(area),
                                        vertices(vertices), normal(normal) {}
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
        }

        facet_vec.push_back(Facet(area, vertices, normal));
    }
    return facet_vec;
}

void flux(double vth, std::vector<double> vd,
          std::function<double(std::vector<double> &)> vdf,
          std::vector<std::vector<double>> &cutoffs,
          int nsp,
          std::vector<Facet> &facets,
          std::vector<double> &num_particles,
          std::vector<std::function<double(std::vector<double> &)>> &vdf_flux)
{
    auto num_facets = facets.size();
    num_particles.resize(num_facets);
    vdf_flux.resize(num_facets);
    auto dim = cutoffs.size();
    for(auto i=0; i<num_facets; ++i)
    {
        auto n = facets[i].normal;
        auto vdn = std::inner_product(std::begin(n), std::end(n), std::begin(vd), 0.0);
        num_particles[i] = facets[i].area * (vth / (sqrt(2 * M_PI)) *
                           exp(-0.5 * (vdn / vth) * (vdn / vth)) +
                           0.5 * vdn * (1. + erf(vdn / (sqrt(2) * vth))));

        vdf_flux[i] = [vdf, n](std::vector<double> &v) {
            auto vdn = std::inner_product(std::begin(n), std::end(n), std::begin(v), 0.0);
            if (vdn >= 0.0)
            {
                return vdn * vdf(v);
            }else{
                return 0.0;
            }
        };
    }
}

// void inject_particles(Population &pop, SpeciesList &listOfSpecies,
//                       std::vector<Facet> &facets, double dt)
// {
//     typedef std::mt19937_64 random_source;
//     typedef std::uniform_real_distribution<double> distribution;
//     random_source rng{random_seed_seq::get_instance()};
//     distribution dist(0.0, 1.0);

//     auto dim = pop.gdim;
//     std::size_t num_species = listOfSpecies.species.size();
//     auto num_facets = facets.size();
//     std::vector<double> xs_tmp(dim);
//     for (std::size_t i = 0; i < num_species; ++i)
//     {
//         auto s = listOfSpecies.species[i];
//         auto num_particles = s.num_particles;
//         auto vdf_flux = s.vdf_flux;
//         auto n_p = s.n;
//         std::vector<double> xs, vs;
//         for(std::size_t j = 0; j < num_facets; ++j)
//         {
//             int N = int(n_p*dt*num_particles[j]);
//             if (dist(rng) < (n_p * dt * num_particles[j]-N))
//             {
//                 N += 1;
//             }
//             auto count = 0;
//             while (count <N)
//             {
//                 auto n = N - count;
//                 auto xs_new = random_facet_points(n, facets.vertices);
//                 auto vs_new = ORS(vdf_flux[j], s.cutoffs, s.num_sp, n);

//                 for(auto k=0; k<n; ++k)
//                 {
//                     auto r = dist(rng);
//                     for (auto l = 0; l <dim; ++l)
//                     {
//                         xs_tmp[l] = xs_new[k*dim + l] + dt*r*vs_new[k*dim + l];
//                     }
//                     if (pop.locate(xs_tmp) >= 0)
//                     {
//                         for (auto l = 0; l < dim; ++l)
//                         {
//                             xs.push_back(xs_tmp[l]);
//                             vs.push_back(vs_new[k * dim + l]);
//                         }
//                     }
//                     count += 1;
//                 }
//             }
//         }
//         pop.add_particles(xs, vs, s.q, s.m);
//     }
// }

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

int main()
{
    Timer timer;
    // std::string fname{"/home/diako/Documents/cpp/punc_experimental/prototypes/mesh/square"};
    std::string fname{"/home/diako/Documents/cpp/punc_experimental/demo/injection/mesh/box"};
    auto mesh = load_mesh(fname);
    auto D = mesh->geometry().dim();
    auto tdim = mesh->topology().dim();
    // df::plot(mesh);
    // df::interactive();
    // exit(EXIT_FAILURE);
    auto boundaries = load_boundaries(mesh, fname);
    auto tags = get_mesh_ids(boundaries);
    std::size_t ext_bnd_id = tags[1];

    auto facets = exterior_boundaries(boundaries, ext_bnd_id);
    auto num_facets = facets.size();

    int num_sp = 100;
    std::vector<std::vector<double>> cutoffs{{-7., 7.}, {-7., 7.}, {-7., 7.}};

    double vth = 1.0;
    std::vector<double> vd{0.0,0.0,0.0};
    std::vector<double> n{1.0, 0.0,0.0};

    auto pdf = shifted_maxwellian(vth, vd);

    auto vdf_flux = [pdf, n](std::vector<double> &v) {
        auto vdn = std::inner_product(std::begin(n), std::end(n), std::begin(v), 0.0);
        if (vdn >= 0.0)
        {
            return vdn * pdf(v);
        }
        else
        {
            return 0.0;
        }
    };
    int N = 500000;
    timer.reset();
    ORS ors(vdf_flux, cutoffs, num_sp);
    auto t0 = timer.elapsed();
    std::cout << "Initialization time: " << t0 << '\n';
    timer.reset();
    auto vs = ors.sample(N);
    t0 = timer.elapsed();
    std::cout << "Sampling time: " << t0 << '\n';
    std::cout << "Number of samples: " << vs.size() << '\n';
    // auto vs = ORS(vdf_flux, cutoffs, num_sp, N);

    // std::vector<double> num_particles;
    // std::vector<std::function<double(std::vector<double> &)>> vdf_flux;
    // flux(vth, vd, pdf, cutoffs, num_sp, facets, num_particles, vdf_flux);

    // std::vector<ORS> ors_vec;
    // for(std::size_t k = 0; k < num_facets; ++k)
    // {
    //     ORS ors(vdf_flux[k], cutoffs, num_sp);
    //     ors_vec.push_back(ors);
    // }

    // std::vector<double> v1, v2, v3, v4;
    // timer.reset();
    // for(std::size_t k = 0; k < num_facets; ++k)
    // {
    //     std::cout<<"k: "<<k<<'\n';
    //     int N = 100;//*num_particles[k];
    //     // auto vs = ORS1(vdf_flux[k], cutoffs, num_sp, N);
    //     auto vs = ors_vec[k].sample(N);
    //     // auto n = facets[k].normal;
    //     // if (n[0] >0 && n[1]==0.0)
    //     // {
    //     //     for (std::size_t i = 0; i < N; ++i)
    //     //     {
    //     //         for (std::size_t j = 0; j < D; ++j)
    //     //         {
    //     //             v1.push_back(vs[i * D + j]);
    //     //         }
    //     //     }
    //     // }
    //     // if (n[1] > 0 && n[0] == 0.0)
    //     // {
    //     //     for (std::size_t i = 0; i < N; ++i)
    //     //     {
    //     //         for (std::size_t j = 0; j < D; ++j)
    //     //         {
    //     //             v2.push_back(vs[i * D + j]);
    //     //         }
    //     //     }
    //     // }
    //     // if (n[0] < 0 && n[1] == 0.0)
    //     // {
    //     //     for (std::size_t i = 0; i < N; ++i)
    //     //     {
    //     //         for (std::size_t j = 0; j < D; ++j)
    //     //         {
    //     //             v3.push_back(vs[i * D + j]);
    //     //         }
    //     //     }
    //     // }
    //     // if (n[1] < 0 && n[0] == 0.0)
    //     // {
    //     //     for (std::size_t i = 0; i < N; ++i)
    //     //     {
    //     //         for (std::size_t j = 0; j < D; ++j)
    //     //         {
    //     //             v4.push_back(vs[i * D + j]);
    //     //         }
    //     //     }
    //     // }
    // }
    // auto t0 = timer.elapsed();
    // std::cout << "Total time: " << t0 << '\n';
    // // std::cout << "\n";
    // // exit(EXIT_FAILURE);

    // // auto vs = ORS(pdf, cutoffs, num_sp, N);
    // // auto t0 = timer.elapsed();
    // // std::cout << "Total time: " << t0 << '\n';
    // std::ofstream file;
    // file.open("vs1.txt");
    // for (const auto &e : v1)
    //     file << e << "\n";
    // file.close();

    // file.open("vs2.txt");
    // for (const auto &e : v2)
    //     file << e << "\n";
    // file.close();
    // file.open("vs3.txt");
    // for (const auto &e : v3)
    //     file << e << "\n";
    // file.close();
    // file.open("vs4.txt");
    // for (const auto &e : v4)
    //     file << e << "\n";
    // file.close();
    return 0;
}
