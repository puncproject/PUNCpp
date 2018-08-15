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

/**
 * @file		injector.h
 * @brief		Particle injector
 *
 * Velocity distribution functions and functions for injecting particles.
 */

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
    int _dim;
    std::vector<double> _domain;
public:
    UniformPosition(std::shared_ptr<const df::Mesh> mesh) : mesh(mesh)
    {
        _dim = mesh->geometry().dim();
        auto coordinates = mesh->coordinates();
        auto Ld_min = *std::min_element(coordinates.begin(), coordinates.end());
        auto Ld_max = *std::max_element(coordinates.begin(), coordinates.end());
        _domain.resize(2 * _dim);
        for (int i = 0; i < _dim; ++i)
        {
            _domain[i] = Ld_min;
            _domain[i + _dim] = Ld_max;
        }
    }
    double operator()(const std::vector<double> &x) 
    { 
        return (locate(mesh, x.data()) >= 0) * 1.0; 
    };
    double max() { return 1.0;};
    int dim() { return _dim; };
    std::vector<double> domain() { return _domain; };
};

class Maxwellian : public Pdf
{
private:
    double _vth;
    std::vector<double> _vd;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

public:
   
    std::vector<double> pdf_max, num_particles;
    
    Maxwellian(double vth, std::vector<double> &vd, bool has_cdf = true, 
               bool has_flux_num=true, bool has_flux_max=true, 
               double vdf_range = 5.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max() { return factor; };
    std::vector<double> domain() { return _domain; };
    double vth() { return _vth; };
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; };
    void set_vd(std::vector<double> &v) { _vd = v; };
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    std::vector<double> cdf(const std::size_t N);
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
    double flux_num_particles(const std::vector<double> &n, double S);
    double flux_max(std::vector<double> &n);
};

class Kappa : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double k;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;
    Kappa(double vth, std::vector<double> &vd, double k, bool has_cdf = false,
          bool has_flux_num = true, bool has_flux_max = true, 
          double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max() { return factor; }
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
    double flux_num_particles(const std::vector<double> &n, double S);
    double flux_max(std::vector<double> &n);
};

class Cairns : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double alpha;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;
    Cairns(double vth, std::vector<double> &vd, double alpha, 
           bool has_cdf = false, bool has_flux_num = true, 
           bool has_flux_max = false, double vdf_range = 7.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max();
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
    double flux_num_particles(const std::vector<double> &n, double S);
};

class KappaCairns : public Pdf
{
  private:
    double _vth;
    std::vector<double> _vd;
    double k;
    double alpha;
    int _dim;
    bool _has_cdf;
    bool _has_flux_number;
    bool _has_flux_max;
    std::vector<double> _domain;
    double vth2, factor;
    std::vector<double> _n;
    bool has_flux = false;

  public:
    std::vector<double> pdf_max, num_particles;
    KappaCairns(double vth, std::vector<double> &vd, double k, double alpha,
                bool has_cdf = false, bool has_flux_num = false,
                bool has_flux_max = false, double vdf_range = 25.0);
    double operator()(const std::vector<double> &v);
    double operator()(const std::vector<double> &v, const std::vector<double> &n);
    int dim() { return _dim; }
    double max();
    std::vector<double> domain() { return _domain; }
    double vth() { return _vth; }
    std::vector<double> vd() { return _vd; };
    void set_vth(double v) { _vth = v; }
    void set_vd(std::vector<double> &v) { _vd = v; }
    bool has_cdf() { return _has_cdf; };
    bool has_flux_max() { return _has_flux_max; };
    bool has_flux_number() { return _has_flux_number; };
    void eval(df::Array<double> &values, const df::Array<double> &x) const;
    void set_flux_normal(std::vector<double> &n)
    {
        has_flux = true;
        _n = n;
    }
};

void create_flux_FEM(std::vector<Species> &species, std::vector<Facet> &facets);

void create_flux(std::vector<Species> &species, std::vector<Facet> &facets);

std::vector<double> rejection_sampler(const std::size_t N,
                                      std::function<double(std::vector<double> &)> pdf,
                                      double pdf_max, int dim,
                                      const std::vector<double> &domain);

std::vector<double> random_facet_points(const std::size_t N, 
                                        const std::vector<double> &vertices);

template <std::size_t len>
void inject_particles(Population<len> &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, const double dt)
{
    std::mt19937_64 rng{random_seed_seq::get_instance()};
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    auto g_dim = pop.g_dim;
    auto num_species = species.size();
    auto num_facets = facets.size();
    double xs_tmp[g_dim];

    for (std::size_t i = 0; i < num_species; ++i)
    {
        std::vector<double> xs, vs;
        for (std::size_t j = 0; j < num_facets; ++j)
        {
            auto normal_i = facets[j].normal;
            auto N_float = species[i].n * dt * species[i].vdf.num_particles[j];
            int N = int(N_float);
            if (rand(rng) < (N_float - N))
            {
                N += 1;
            }
            auto vdf = [i, &normal_i, &species](std::vector<double> &v) -> double {
                return species[i].vdf(v, normal_i);
            };
    
            auto pdf_max = species[i].vdf.pdf_max[j];
            auto count = 0;
            while (count < N)
            {
                auto n = N - count;
                auto xs_new = random_facet_points(n, facets[j].vertices);
                auto vs_new = rejection_sampler(n, vdf, pdf_max,
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

template <std::size_t len>
void load_particles(Population<len> &pop, std::vector<Species> &species)
{
    auto num_species = species.size();
    std::vector<double> xs, vs;
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto s = species[i];
        auto pdf = [&s](std::vector<double> &x) -> double { return s.pdf(x); };
        auto vdf = [&s](std::vector<double> &v) -> double { return s.vdf(v); };

        xs = rejection_sampler(s.num, pdf, s.pdf.max(), s.pdf.dim(), s.pdf.domain());
        if (s.vdf.has_cdf())
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

} // namespace punc

#endif // INJECTOR_H
