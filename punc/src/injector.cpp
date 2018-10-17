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
#include "../ufl/Number.h"

namespace punc
{

typedef std::uniform_real_distribution<double> rand_uniform;

std::vector<Facet> exterior_boundaries(df::MeshFunction<std::size_t> &boundaries,
                                       std::size_t ext_bnd_id)
{
    auto mesh = boundaries.mesh();
    auto g_dim = mesh->geometry().dim();
    auto t_dim = mesh->topology().dim();
    auto values = boundaries.values();
    auto length = boundaries.size();
    int num_facets = 0;
    for (std::size_t i = 0; i < length; ++i)
    {
        if (ext_bnd_id == values[i])
        {
            num_facets += 1;
        }
    }
    std::vector<Facet> ext_facets;
 
    double area = 0.0;
    std::vector<double> normal(g_dim);
    std::vector<double> vertices(g_dim * g_dim);
    std::vector<double> basis(g_dim * g_dim);
    std::vector<double> vertex(g_dim);
    double norm;
    int j;
    mesh->init(t_dim - 1, t_dim);
    df::SubsetIterator facet_iter(boundaries, ext_bnd_id);
    for (; !facet_iter.end(); ++facet_iter)
    {
        df::Cell cell(*mesh, facet_iter->entities(t_dim)[0]);
        auto cell_facet = cell.entities(t_dim - 1);
        std::size_t num_facets = cell.num_entities(t_dim - 1);
        for (std::size_t i = 0; i < num_facets; ++i)
        {
            if (cell_facet[i] == facet_iter->index())
            {
                area = cell.facet_area(i);
                for (std::size_t j = 0; j < g_dim; ++j)
                {
                    normal[j] = -1 * cell.normal(i, j);
                    basis[j * g_dim] = normal[j];
                }
            }
        }
        assert(area != 0.0 && "The facet area cannot be zero!");

        j = 0;
        for (df::VertexIterator v(*facet_iter); !v.end(); ++v)
        {
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                vertices[j * g_dim + i] = v->x(i);
            }
            j += 1;
        }
        norm = 0.0;
        for (std::size_t i = 0; i < g_dim; ++i)
        {
            vertex[i] = vertices[i] - vertices[g_dim + i];
            norm += vertex[i] * vertex[i];
        }
        for (std::size_t i = 0; i < g_dim; ++i)
        {
            vertex[i] /= sqrt(norm);
            basis[i * g_dim + 1] = vertex[i];
        }

        if (g_dim == 3)
        {
            basis[2] = normal[1] * vertex[2] - normal[2] * vertex[1];
            basis[5] = normal[2] * vertex[0] - normal[0] * vertex[2];
            basis[8] = normal[0] * vertex[1] - normal[1] * vertex[0];
        }
        ext_facets.push_back(Facet{area, vertices, normal, basis});
    }
    return ext_facets;
}

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

void create_flux_FEM(std::vector<Species> &species, std::vector<Facet> &facets)
{
    auto num_species = species.size();
    auto num_facets = facets.size();
    std::vector<int> nsp = {60, 60, 60};

    for (std::size_t i = 0; i < num_species; ++i)
    {
        auto dim = species[i].vdf.dim();
        auto domain = species[i].vdf.domain();
        df::Point p0, p1;
        for (int k = 0; k < dim; ++k)
        {
            p0[k] = domain[k];
            p1[k] = domain[k + dim];
        }

        std::shared_ptr<const df::Mesh> mesh;
        if (dim == 1)
        {
            df::IntervalMesh interval(nsp[0], domain[0], domain[1]);
            mesh = std::make_shared<const df::Mesh>(interval);
        }
        else if (dim == 2)
        {
            df::RectangleMesh rectangle(p0, p1, nsp[0], nsp[1]);
            mesh = std::make_shared<const df::Mesh>(rectangle);
        }
        else if (dim == 3)
        {
            df::BoxMesh box(p0, p1, nsp[0], nsp[1], nsp[2]);
            mesh = std::make_shared<const df::Mesh>(box);
        }

        auto V = Number::CoefficientSpace_w0(mesh);
        for (std::size_t j = 0; j < num_facets; ++j)
        {
            species[i].vdf.set_flux_normal(facets[j].normal);
            
            df::Function vdf_func(std::make_shared<df::FunctionSpace>(V));
            vdf_func.interpolate(species[i].vdf);
            auto vdf_vector = vdf_func.vector();
            
            if (species[i].vdf.has_flux_number())
            {
                auto num = species[i].vdf.flux_num_particles(facets[j].normal, facets[j].area);
                species[i].vdf.num_particles.push_back(num);
            }
            else
            {
                std::shared_ptr<df::Form> form;
                auto vdf_func_ptr = std::make_shared<df::Function>(vdf_func);
                if (dim == 1)
                {
                    form = std::make_shared<Number::Form_0>(mesh, vdf_func_ptr);
                }
                else if (dim == 2)
                {
                    form = std::make_shared<Number::Form_1>(mesh, vdf_func_ptr);
                }
                else if (dim == 3)
                {
                    form = std::make_shared<Number::Form_2>(mesh, vdf_func_ptr);
                }
                species[i].vdf.num_particles.push_back(df::assemble(*form));
            }

            if (species[i].vdf.has_flux_max())
            {
                species[i].vdf.pdf_max.push_back(species[i].vdf.flux_max(facets[j].normal));
            }
            else
            {
                species[i].vdf.pdf_max.push_back(vdf_vector->max());
            }
        }
    }
}

void create_flux(std::vector<Species> &species, std::vector<Facet> &facets)
{
    rand_uniform rand(0.0, 1.0);
    std::mt19937_64 rng(random_seed_seq::get_instance());

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
