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

#include "population.h"
#include <random>

namespace punc
{

namespace df = dolfin;

/**
 * @brief Generates random numbers to seed std::mt19937_64, i.e., the Mersenne 
 * Twister pseudo-random generator of 64-bit numbers
 */
class RandomSeed
{
public:
    /**
     * @brief generates a sequence of random numbers using std::random_device
     */
    template <typename T>
    void generate(T begin, T end)
    {
        for (; begin != end; ++begin)
        {
            *begin = device();
        }
    }
    /**
     * @brief Used to initialize std::mt19937_64
     */
    static RandomSeed &get_instance()
    {
        static thread_local RandomSeed seed;
        return seed;
    }

private:
    std::random_device device; ///< The random device to seed std::mt19937_64
};

/**
 * @brief Creates flux needed for injecting particles through exterior boundary facets
 * @param  species[in] - A vector containing all the plasma species
 * @param  facets[in] - A vector containing all the exterior facets
 * 
 * For each species and for each exterior boundary facet, calculates the number 
 * of particles to be injected through the facet. In addition, for each facet
 * finds the maximum value of the flux probability distribution function by 
 * using Monte Carlo integration. This value is needed in the rejection sampler.
 */
void create_flux(std::vector<Species> &species, std::vector<ExteriorFacet> &facets);

/**
 * @brief Standard rejection sampler
 * @param N[in] - number of random numbers to be generated
 * @param pdf[in] - a probability distribution function to sample from
 * @param pdf_max[in] - maximum value of pdf
 * @param dim[in] - the dimension of the space pdf is defined on
 * @param domain[in] - the domian in which the random numbers are constrained to
 * @param rand[in] - the uniform distribution function
 * @param rng[in] - The Mersenne Twister pseudo-random generator of 64-bit numbers
 * @return A vector of random numbers from pdf 
 * 
 * Standard rejection sampler uses a simpler proposal distribution, which in 
 * this case is a uniform distribution function, U, to generate random samples. 
 * A sample is then accepted with probability pdf(v)/ U(v), or discarded 
 * otherwise. This process is repeated until a sample is accepted.
 */
std::vector<double> rejection_sampler(std::size_t N,
                                      std::function<double(std::vector<double> &)> pdf,
                                      double pdf_max, int dim,
                                      const std::vector<double> &domain,
                                      std::uniform_real_distribution<double> &rand,
                                      std::mt19937_64 &rng);

/**
 * @brief Generates uniformly distributed random particle positions on a given exterior facet
 * @param N[in] - number of random particle positions to be generated
 * @param vertices[in] - a vector containing all the vertices of the facet
 * @param rand[in] - the uniform distribution function
 * @param rng[in] - The Mersenne Twister pseudo-random generator of 64-bit numbers
 * @return A vector of random numbers from pdf 
 * 
 * In 1D, a facet is a single point. Hence, there is no point in generating 
 * uniformly distributed random particle positions on a point. Therefore, this 
 * function is only valid for 2D and 3D. In 2D, the facet is a straight line, 
 * and in 3D the facet is triangle. Random positions are generated within the 
 * facet depending on the geometrical dimension. 
 */
std::vector<double> random_facet_points(std::size_t N, 
                                        const std::vector<double> &vertices,
                                        std::uniform_real_distribution<double> &rand,
                                        std::mt19937_64 &rng);

/**
 * @brief Injects particles through the exterior boundary facets
 * @param pop[in] - the plasma particle population
 * @param species[in] - a vector containing all the plasma species
 * @param facets[in] - a vector containing all the exterior facets
 * @param dt[in] - duration of a time-step 
 * 
 * Generates random particle velocities and positions for each species from each
 * exterior boundary facet, and adds the newly created particles to plasma 
 * population. 
 */
template <typename PopulationType>
void inject_particles(PopulationType &pop, std::vector<Species> &species,
                      std::vector<ExteriorFacet> &facets, double dt)
{
    std::mt19937_64 rng{RandomSeed::get_instance()};
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
                                                species[i].vdf.dim,
                                                species[i].vdf.domain,
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

/**
 * @brief Generates particle velocities and positions for each species, and populates the simultion domain
 * @param pop[in] - the plasma particle population
 * @param species[in] - a vector containing all the plasma species
 * 
 * Generates random particle velocities based on the prespecified velocity
 * distribution function for each species, and populates the simulation domain
 * by generating uniformly distributed random positions in the entire domain.
 * The newly created particles are then added to the plasma population.
 */
template <typename PopulationType>
void load_particles(PopulationType &pop, std::vector<Species> &species)
{
    std::mt19937_64 rng{RandomSeed::get_instance()};
    std::uniform_real_distribution<double> rand(0.0, 1.0);

    auto num_species = species.size();
    std::vector<double> xs, vs;
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto s = species[i];
        auto dim = s.vdf.dim;
        auto pdf = [&s](std::vector<double> &x) -> double { return s.pdf(x); };
        auto vdf = [&s](std::vector<double> &v) -> double { return s.vdf(v); };

        xs = rejection_sampler(s.num, pdf, s.pdf.max(), dim, s.pdf.domain, rand, rng);
        if (s.vdf.has_icdf)
        {
            std::vector<double> rand_tmp(s.num * dim, 0.0);
            for (auto j = 0; j < dim; ++j)
            {
                for (auto k = 0; k < s.num; ++k)
                {
                    rand_tmp[k * dim + j] = rand(rng);
                }
            }
            vs = s.vdf.icdf(rand_tmp);
        }
        else
        {
            vs = rejection_sampler(s.num, vdf, s.vdf.max(), dim, s.vdf.domain, rand, rng);
        }

        pop.add_particles(xs, vs, s.q, s.m);
    }
}

} // namespace punc

#endif // INJECTOR_H
