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
#include <random>

namespace punc
{

namespace df = dolfin;

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

void create_flux_FEM(std::vector<Species> &species, std::vector<Facet> &facets);

void create_flux(std::vector<Species> &species, std::vector<Facet> &facets);

std::vector<double> rejection_sampler(std::size_t N,
                                      std::function<double(std::vector<double> &)> pdf,
                                      double pdf_max, int dim,
                                      const std::vector<double> &domain,
                                      std::uniform_real_distribution<double> &rand,
                                      std::mt19937_64 &rng);

std::vector<double> random_facet_points(std::size_t N, 
                                        const std::vector<double> &vertices,
                                        std::uniform_real_distribution<double> &rand,
                                        std::mt19937_64 &rng);

template <typename PopulationType>
void inject_particles(PopulationType &pop, std::vector<Species> &species,
                      std::vector<Facet> &facets, double dt)
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
                auto xs_new = random_facet_points(n, facets[j].vertices,
                                                  rand, rng);

                auto vs_new = rejection_sampler(n, vdf, pdf_max,
                                                species[i].vdf.dim(),
                                                species[i].vdf.domain(),
                                                rand, rng);

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

template <typename PopulationType>
void load_particles(PopulationType &pop, std::vector<Species> &species)
{
    std::mt19937_64 rng{random_seed_seq::get_instance()};
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    auto num_species = species.size();
    std::vector<double> xs, vs;
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto s = species[i];
        auto dim = s.vdf.dim();
        auto pdf = [&s](std::vector<double> &x) -> double { return s.pdf(x); };
        auto vdf = [&s](std::vector<double> &v) -> double { return s.vdf(v); };

        xs = rejection_sampler(s.num, pdf, s.pdf.max(), dim, s.pdf.domain(),
                               rand, rng);
        if (s.vdf.has_cdf())
        {
            std::vector<double> rand_tmp(s.num * dim, 0.0);
            for (auto j = 0; j < dim; ++j)
            {
                for (auto k = 0; k < s.num; ++k)
                {
                    rand_tmp[k * dim + j] = rand(rng);
                }
            }
            vs = s.vdf.cdf(rand_tmp);
        }
        else
        {
            vs = rejection_sampler(s.num, vdf, s.vdf.max(), dim, s.vdf.domain(),
                                   rand, rng);
        }
        pop.add_particles(xs, vs, s.q, s.m);
    }
}

} // namespace punc

#endif // INJECTOR_H
