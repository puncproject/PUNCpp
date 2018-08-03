// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++.
//
// PUNC++ is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC++ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC++. If not, see <http://www.gnu.org/licenses/>.

#ifndef INJECTOR_H
#define INJECTOR_H

#include <dolfin.h>
#include "population.h"
#include <boost/math/special_functions/erf.hpp>
#include <chrono>
#include <random>

namespace punc
{

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

struct Facet
{
    double area;
    std::vector<double> vertices;
    std::vector<double> normal;
    std::vector<double> basis;
};

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id);

class UniformPosition : public Pdf
{
private:
    std::shared_ptr<const df::Mesh> mesh;
    int dim_;
    std::vector<double> domain_;
public:
    UniformPosition(std::shared_ptr<const df::Mesh> mesh) : mesh(mesh)
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
    double operator()(const std::vector<double> &x) 
    { 
        return (locate(mesh, x.data()) >= 0) * 1.0; 
    };
    double max() { return 1.0;};
    int dim() { return dim_; };
    std::vector<double> domain() { return domain_; };
};

class Maxwellian : public Pdf
{
private:
    double vth_;
    std::vector<double> vd_;
    int dim_;
    std::vector<double> domain_;
    double vth2, factor;
    std::vector<double> n_;
    bool has_flux = false;
    bool has_cdf = true;
    std::vector<double> max_vec;
public:
    Maxwellian(double vth, std::vector<double> &vd, double vdf_range = 5.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &x, const std::vector<double> &n);
    double max() { return factor; };
    int dim() { return dim_; }
    std::vector<double> domain() { return domain_; };
    double vth() { return vth_; };
    std::vector<double> vd() { return vd_; };
    void set_vth(double v) { vth_ = v; };
    void set_vd(std::vector<double> &v) { vd_ = v; };
    double flux(const std::vector<double> &n) { return 0; };
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        n_ = n;
    }
    double flux_max(std::vector<double> &n);
    double flux_num(const std::vector<double> &n, double S);
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    std::vector<double> cdf(const std::size_t N);
};

class Kappa : public Pdf
{
private:
    double vth_;
    std::vector<double> vd_;
    double k;
    int dim_;
    std::vector<double> domain_;
    double vth2, factor;
public:
    Kappa(double vth, std::vector<double> &vd, double k, double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double max() { return factor; }
    int dim() { return dim_; }
    std::vector<double> domain() { return domain_; }
    double vth() { return vth_; }
    std::vector<double> vd() { return vd_; };
    void set_vth(double v) { vth_ = v; }
    void set_vd(std::vector<double> &v) { vd_ = v; }
    double flux(const std::vector<double> &n) { return 0; }
    double flux_num(const std::vector<double> &n, double S) { return 0; }
};

std::vector<double> rejection_sampler(const std::size_t N,
                                      std::function<double(std::vector<double> &)> pdf,
                                      double pdf_max, int dim,
                                      const std::vector<double> &domain);

std::vector<double> random_facet_points(const std::size_t N, 
                                        const std::vector<double> &vertices);

template <std::size_t _dim>
void inject_particles(Population<_dim> &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt)
{
    std::mt19937_64 rng{random_seed_seq::get_instance()};
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    auto g_dim = pop.g_dim;
    auto num_species = species.size();
    auto num_facets = facets.size();
    // std::vector<double> xs_tmp(g_dim);
    double xs_tmp[g_dim];

    for (std::size_t i = 0; i < num_species; ++i)
    {
        std::vector<double> xs, vs;
        for (std::size_t j = 0; j < num_facets; ++j)
        {
            auto normal_i = facets[j].normal;
            auto N_float = species[i].n * dt * species[i].vdf.flux_num(normal_i, facets[j].area);
            int N = int(N_float);
            if (rand(rng) < (N_float - N))
            {
                N += 1;
            }
            auto vdf = [i, &normal_i, &species](std::vector<double> &v) -> double {
                return species[i].vdf(v, normal_i);
            };
    
            
            auto pdf_max = species[i].vdf.flux_max(normal_i);
            auto count = 0;
            while (count < N)
            {
                auto n = N - count;
                auto xs_new = random_facet_points(n, facets[j].vertices);
                auto vs_new = rejection_sampler(n, vdf, species[i].vdf.flux_max(normal_i),
                                                species[i].vdf.dim(),
                                                species[i].vdf.domain());

                for (auto k = 0; k < n; ++k)
                {
                    auto r = rand(rng);
                    for (std::size_t l = 0; l < g_dim; ++l)
                    {
                        xs_tmp[l] = xs_new[k * g_dim + l] + dt * r * vs_new[k * g_dim + l];
                    }
                    if (pop.locate(xs_tmp) >= 0)
                    {
                        for (std::size_t l = 0; l < g_dim; ++l)
                        {
                            xs.push_back(xs_tmp[l]);
                            vs.push_back(vs_new[k * g_dim + l]);
                        }
                    }
                    count += 1;
                }
            }
        }
        pop.add_particles(xs, vs, species[i].q, species[i].m);
    }
}

template <std::size_t _dim>
void load_particles(Population<_dim> &pop, std::vector<Species> &species)
{
    auto num_species = species.size();
    std::vector<double> xs, vs;
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto s = species[i];
        auto pdf = [&s](std::vector<double> &x) -> double { return s.pdf(x); };
        auto vdf = [&s](std::vector<double> &v) -> double { return s.vdf(v); };

        xs = rejection_sampler(s.num, pdf, s.pdf.max(), s.pdf.dim(), s.pdf.domain());
        if (s.vdf.has_cdf)
        {
            vs = s.vdf.cdf(s.num);
        }
        else
        {
            vs = rejection_sampler(s.num, vdf, s.vdf.max(), s.vdf.dim(), s.vdf.domain());
        }
        pop.add_particles(xs, vs, s.q, s.m);
    }
}

}

#endif
