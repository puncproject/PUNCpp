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

#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "population.h"

namespace punc
{

namespace df = dolfin;

// static inline void matrix_vector_product(double *y, const double *A,
//                                          const double *x, std::size_t m,
//                                          std::size_t n);

template <std::size_t _dim>
double kinetic_energy(Population<_dim> &pop)
{
    double KE = 0.0;
    for (df::MeshEntityIterator e(*pop.mesh, pop.t_dim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            auto m = particle.m;
            auto v = particle.v;
            for (std::size_t i = 0; i < pop.g_dim; ++i)
            {
                KE += 0.5 * m * v[i] * v[i];
            }
        }
    }
    return KE;
}

template <std::size_t _dim>
double kinetic_energy_new(Population<_dim> &pop)
{
    double KE = 0.0;
    for (auto &cell : pop.cells)
    {
        for (auto &particle : cell.particles)
        {
            auto m = particle.m;
            auto v = particle.v;
            for (std::size_t i = 0; i < pop.g_dim; ++i)
            {
                KE += 0.5 * m * v[i] * v[i];
            }
        }
    }
    return KE;
}

double mesh_potential_energy(df::Function &phi, df::Function &rho);

template <std::size_t _dim>
double particle_potential_energy(Population<_dim> &pop, const df::Function &phi)
{
    auto V = phi.function_space();
    auto mesh = V->mesh();
    auto t_dim = mesh->topology().dim();
    auto element = V->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double PE = 0.0;

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        phi.restrict(&coefficients[0], *element, _cell,
                     vertex_coordinates.data(), ufc_cell);

        std::vector<double> basis(v_dim);
        basis_matrix.resize(v_dim);
        for (std::size_t i = 0; i < v_dim; ++i)
        {
            basis_matrix[i].resize(s_dim);
        }

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            std::vector<double> phii(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x,
                                        vertex_coordinates.data(),
                                        cell_orientation);

                for (std::size_t j = 0; j < v_dim; ++j)
                {
                    basis_matrix[j][i] = basis[j];
                }
            }
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                for (std::size_t j = 0; j < v_dim; j++)
                {
                    phii[j] += coefficients[i] * basis_matrix[j][i];
                }
            }
            auto q = particle.q;
            for (std::size_t j = 0; j < v_dim; j++)
            {
                PE += 0.5 * q * phii[j];
            }
        }
    }
    return PE;
}

template <std::size_t _dim>
double particle_potential_energy_cg1(Population<_dim> &pop, const df::Function &phi)
{
    auto V = phi.function_space();
    auto mesh = V->mesh();
    auto element = V->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / v_dim; // Number of vertices

    double phi_x, PE = 0.0;

    double coeffs[n_dim];
    double values[s_dim];
    for (auto &cell : pop.cells)
    {
        phi.restrict(values, *element, cell, cell.vertex_coordinates.data(), cell.ufc_cell);

        for (auto &particle : cell.particles)
        {
            matrix_vector_product(&coeffs[0], cell.basis_matrix.data(),
                                  particle.x, n_dim, n_dim);
            phi_x = 0.0;
            for (std::size_t i = 0; i < n_dim; ++i)
            {
                phi_x += coeffs[i] * values[i];
            }

            PE += 0.5 * particle.q * phi_x;
        }
    }
    return PE;
}

// static inline void matrix_vector_product(double *y, const double *A,
//                                          const double *x, std::size_t n,
//                                          std::size_t m)
// {
//     for (std::size_t i = 0; i < n; ++i)
//     {
//         y[i] = A[i * m];
//     }
//     for (std::size_t i = 0; i < n; ++i)
//     {
//         for (std::size_t j = 0; j < m - 1; ++j)
//         {
//             y[i] += A[i * m + j + 1] * x[j];
//         }
//     }
// }
}

#endif
