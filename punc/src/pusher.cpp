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

#include "../include/pusher.h"

namespace punc
{

double accel(Population &pop, const df::Function &E, const double dt)
{
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto tdim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(&coefficients[0], *element, _cell,
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
            std::vector<double> Ei(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
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
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                }
            }

            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
            }
            for (std::size_t j = 0; j < v_dim; j++)
            {
                pop.cells[cell_id].particles[p_id].v[j] += Ei[j];
            }
        }
    }
    return KE;
}

double boris(Population &pop, const df::Function &E,
             const std::vector<double> &B, const double dt)
{
    auto dim = B.size();
    assert(dim == 3 && "The algorithm is only valid for 3D.");
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto tdim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;
    double t_mag2;

    std::vector<double> v_minus(dim), v_prime(dim), v_plus(dim);
    std::vector<double> t(dim), s(dim);

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(&coefficients[0], *element, _cell,
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
            std::vector<double> Ei(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
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
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                }
            }

            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;

            t_mag2 = 0.0;
            for (auto i = 0; i < dim; ++i)
            {
                t[i] = tan((dt * q / (2.0 * m)) * B[i]);
                t_mag2 += t[i] * t[i];
            }

            for (auto i = 0; i < dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }

            for (auto i = 0; i < dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }

            for (auto i = 0; i < dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (auto i = 0; i < dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }

            auto v_prime_cross_s = cross(v_prime, s);
            for (auto i = 0; i < dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }

            for (auto i = 0; i < dim; ++i)
            {
                pop.cells[cell_id].particles[p_id].v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

double boris_nonuniform(Population &pop, const df::Function &E,
                        const df::Function &B, const double dt)
{
    auto dim = pop.gdim;
    assert(dim == 3 && "The algorithm is only valid for 3D.");
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto tdim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;
    double t_mag2;

    std::vector<double> v_minus(dim), v_prime(dim), v_plus(dim);
    std::vector<double> t(dim), s(dim);

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> coefficients_B(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(&coefficients[0], *element, _cell,
                   vertex_coordinates.data(), ufc_cell);

        B.restrict(&coefficients_B[0], *element, _cell,
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
            std::vector<double> Ei(v_dim, 0.0);
            std::vector<double> Bi(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
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
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                    Bi[j] += coefficients_B[i] * basis_matrix[j][i];
                }
            }
            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;
            t_mag2 = 0.0;
            for (auto i = 0; i < dim; ++i)
            {
                t[i] = tan( (dt * q / (2.0 * m)) * Bi[i]);
                t_mag2 += t[i]*t[i];
            }
            for (auto i = 0; i < dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }
            for (auto i = 0; i < dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }
            for (auto i = 0; i < dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (auto i = 0; i < dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }
            auto v_prime_cross_s = cross(v_prime, s);
            for (auto i = 0; i < dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }
            for (auto i = 0; i < dim; ++i)
            {
                pop.cells[cell_id].particles[p_id].v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

void move_periodic(Population &pop, const double dt, const std::vector<double> &Ld)
{
    auto dim = Ld.size();
    auto num_cells = pop.num_cells;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t j = 0; j < dim; ++j)
            {
                pop.cells[cell_id].particles[p_id].x[j] += dt * pop.cells[cell_id].particles[p_id].v[j];
                pop.cells[cell_id].particles[p_id].x[j] -= Ld[j] * floor(pop.cells[cell_id].particles[p_id].x[j] / Ld[j]);
            }
        }
    }
}

void move(Population &pop, const double dt)
{
    auto dim = pop.gdim;
    auto num_cells = pop.num_cells;
    for (std::size_t cell_id = 0; cell_id < num_cells; ++cell_id)
    {
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t j = 0; j < dim; ++j)
            {
                pop.cells[cell_id].particles[p_id].x[j] += dt * pop.cells[cell_id].particles[p_id].v[j];
            }
        }
    }
}

std::vector<double> cross(std::vector<double> &v1, std::vector<double> &v2)
{
    std::vector<double> r(v1.size());
    r[0] = v1[1] * v2[2] - v1[2] * v2[1];
    r[1] = -v1[0] * v2[2] + v1[2] * v2[0];
    r[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return r;
}

}
