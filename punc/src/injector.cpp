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

#include "../include/punc/injector.h"

namespace punc
{

typedef std::uniform_real_distribution<double> rand_uniform;

std::vector<double> rejection_sampler(std::size_t N,
                                      std::function<double(std::vector<double> &)> pdf,
                                      double pdf_max, int dim,
                                      const std::vector<double> &domain,
                                      rand_uniform &rand,
                                      std::mt19937_64 &rng)
{
    std::vector<double> xs(N * dim), tmp(dim);
    std::size_t n = 0;
    while (n < N)
    {
        for (int i = 0; i < dim; ++i)
        {
            double lower = domain[i];
            double upper = domain[i+dim];
            tmp[i] = lower + (upper-lower)*rand(rng);
        }
        if (rand(rng)*pdf_max < pdf(tmp))
        {
            for(int i = 0; i < dim; ++i)
            {
                xs[n*dim+i] = tmp[i];
            }
            n += 1;
        }
    }
    return xs;
}

std::vector<double> random_facet_points(std::size_t N, 
                                        const std::vector<double> &vertices,
                                        rand_uniform &rand,
                                        std::mt19937_64 &rng)
{
    auto size = vertices.size();
    auto g_dim = sqrt(size);
    std::vector<double> xs(N * g_dim);

    for (std::size_t i = 0; i < N; ++i)
    {
        double r = rand(rng);
        for (int j = 0; j < g_dim; ++j)
        {
            xs[i*g_dim + j] = (1.0 - r) * vertices[j] + r * vertices[j + g_dim];
        }
        for (int j = 1; j < g_dim-1; ++j)
        {
            r = sqrt(rand(rng));
            for (int k = 0; k < g_dim; ++k)
            {
                xs[i*g_dim+k] = r * xs[i*g_dim+k] + (1.0 - r) * vertices[(j + 1) * g_dim + k];
            }
        }
    }
    return xs;
}

void create_flux(std::vector<Species> &species, std::vector<ExteriorFacet> &facets)
{
    rand_uniform rand(0.0, 1.0);
    std::mt19937_64 rng(RandomSeed::get_instance());

    auto num_species = species.size();
    auto num_facets = facets.size();

    double pdf_x, volume, sum, max;
    int n_iter = 500000;
    
    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto dim = species[i].vdf.dim();
        auto domain = species[i].vdf.domain();
        volume = 1.0;
        for (int k = 0; k < dim; k++)
        {
            volume *= domain[k + dim] - domain[k];
        }

        for (std::size_t j = 0; j < num_facets; ++j)
        {
            if (species[i].vdf.has_flux_number())
            {
                auto num = species[i].vdf.flux_num_particles(facets[j].normal, facets[j].area);
                species[i].vdf.num_particles.push_back(num);
            }
            else
            {
                sum = 0; 
                std::vector<double> x(dim);
                for (int n = 0; n < n_iter; ++n)
                {
                    for (int k = 0; k < dim; ++k)
                    {
                        x[k] = domain[k] + rand(rng) * (domain[k + dim] - domain[k]);
                    }

                    sum += species[i].vdf(x, facets[j].normal);
                }
                auto num = facets[j].area*sum*volume/n_iter;
                species[i].vdf.num_particles.push_back(num);
            }

            if (species[i].vdf.has_flux_max())
            {
                species[i].vdf.pdf_max.push_back(species[i].vdf.flux_max(facets[j].normal));
            }
            else
            {
                max = 0;
                std::vector<double> x(dim);
                for (int n = 0; n < n_iter; ++n)
                {
                    for (int k = 0; k < dim; ++k)
                    {
                        x[k] = domain[k] + rand(rng) * (domain[k + dim] - domain[k]);
                    }

                    pdf_x = species[i].vdf(x, facets[j].normal);
                    max = max > pdf_x ? max : pdf_x;
                }
                species[i].vdf.pdf_max.push_back(max*1.01);
            }
        }
    }
}

} // namespace punc
