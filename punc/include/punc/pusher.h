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
 * @file		pusher.h
 * @brief		Particle pusher
 *
 * Functions for pushing (accelerating and moving) particles.
 */

#ifndef PUSHER_H
#define PUSHER_H

#include "population.h"

namespace punc
{

namespace df = dolfin;

static inline std::vector<double> cross(const std::vector<double> &v1,
                                        const std::vector<double> &v2);

/**
 * @brief Accelerates particles in absence of a magnetic field
 * @param[in,out]   pop     Population
 * @param           E       Electric field
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see boris()
 *
 * Advances particle velocities according to
 * \f[
 *      \frac{\mathbf{v}^\mathrm{new}-\mathbf{v}^\mathrm{old}}{\Delta t}
 *      \approx \dot{\mathbf{v}} = \frac{q}{m}\mathbf{E}
 * \f]
 *
 * If the velocities are time-staggered about the electric field (i.e. particle
 * positions), this is a centered difference. Otherwise, it is a forward
 * difference.
 * 
 * To initialize from time-step \f$ n=0 \f$ a time-staggered grid where
 * velocities are at half-integer time-steps, the particles can be accelerated
 * only half a time-step the first time.
 */
template <typename PopulationType>
double accel(PopulationType &pop, const df::Function &E, double dt)
{
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto t_dim = mesh->topology().dim(); // FIXME: Shouldn't this be geometric dimension?
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;

    std::vector<double> basis_matrix(v_dim * s_dim);
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates(t_dim);
    std::vector<double> Ei(v_dim);

    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(coefficients.data(), *element, _cell,
                   vertex_coordinates.data(), ufc_cell);

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            element->evaluate_basis_all(basis_matrix.data(),
                                        particle.x,
                                        vertex_coordinates.data(),
                                        cell_orientation);

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] = 0;
                for (std::size_t i = 0; i < s_dim; ++i)
                {
                    Ei[j] += coefficients[i] * basis_matrix[i * v_dim + j];
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

template <typename PopulationType>
double accel_cg1(PopulationType &pop, const df::Function &E, double dt)
{

    auto W = E.function_space();
    auto mesh = W->mesh();
    auto element = W->element();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / v_dim; // Number of vertices

    double KE = 0.0;

    std::vector<double> vertex_coordinates(t_dim);
    double Ei[v_dim];
    double coeffs[n_dim];
    double values[s_dim];

    for (auto &cell : pop.cells)
    {
        E.restrict(values, *element, cell, cell.vertex_coordinates.data(), cell.ufc_cell);

        for (auto &particle : cell.particles)
        {
            double m = particle.m;
            double q = particle.q;
            auto &vel = particle.v;

            matrix_vector_product(&coeffs[0], cell.basis_matrix.data(),
                                  particle.x, n_dim, n_dim);

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] = 0.0;
                for (std::size_t i = 0; i < n_dim; ++i)
                {
                    Ei[j] += coeffs[i] * values[j * n_dim + i];
                }
            }

            for (std::size_t j = 0; j < g_dim; j++)
            {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
                particle.v[j] += Ei[j];
            }
        }
    }
    return KE;
}

/**
 * @brief Accelerates particles in a homogeneous magnetic field (Only valid for E in CG1)
 * @param[in,out]   pop     Population
 * @param           E       Electric field (CG1)
 * @param           B       Magnetic flux density (std::vector)
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see accel()
 *
 * Advances particle velocities according to the Boris scheme:
 * \f[
 *      \frac{\mathbf{v}^\mathrm{new}-\mathbf{v}^\mathrm{old}}{\Delta t}
 *      \approx \dot{\mathbf{v}} =
 *      \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})
 *      \approx \frac{q}{m}\left(\mathbf{E}+
 *      \frac{\mathbf{v}^\mathrm{new}+\mathbf{v}^\mathrm{old}}{2}
 *      \times\mathbf{B}\right)
 * \f]
 * 
 * To initialize from time-step \f$ n=0 \f$ a time-staggered grid where
 * velocities are at half-integer time-steps, the particles must be accelerated
 * only half a time-step the first time.
 */
template <typename PopulationType>
double boris_cg1(PopulationType &pop, const df::Function &E,
             const std::vector<double> &B, double dt)
{
    auto W = E.function_space();
    auto element = W->element();
    auto mesh = W->mesh();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / v_dim; // Number of vertices

    auto B_dim = B.size();
    assert(B_dim == 3 && "The algorithm is only valid for 3D.");

    double KE = 0.0;
    double t_mag2;

    std::vector<double> v_minus(g_dim), v_prime(g_dim), v_plus(g_dim);
    std::vector<double> t(g_dim), s(g_dim);

    std::vector<double> vertex_coordinates(t_dim);
    double Ei[v_dim];
    double coeffs[n_dim];
    double values[s_dim];

    for (auto &cell : pop.cells)
    {
        E.restrict(values, *element, cell, cell.vertex_coordinates.data(), cell.ufc_cell);

        for (auto &particle : cell.particles)
        {
            double m = particle.m;
            double q = particle.q;
            auto &vel = particle.v;

            matrix_vector_product(&coeffs[0], cell.basis_matrix.data(),
                                  particle.x, n_dim, n_dim);

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] = 0.0;
                for (std::size_t i = 0; i < n_dim; ++i)
                {
                    Ei[j] += coeffs[i] * values[j * n_dim + i];
                }
            }

            t_mag2 = 0.0;
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                t[i] = tan((dt * q / (2.0 * m)) * B[i]);
                t_mag2 += t[i] * t[i];
            }

            for (std::size_t i = 0; i < g_dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }

            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }

            for (std::size_t i = 0; i < g_dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }

            auto v_prime_cross_s = cross(v_prime, s);
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }

            for (std::size_t i = 0; i < g_dim; ++i)
            {
                particle.v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

/**
 * @brief Accelerates particles in a homogeneous magnetic field
 * @param[in,out]   pop     Population
 * @param           E       Electric field
 * @param           B       Magnetic flux density
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see accel()
 *
 * Advances particle velocities according to the Boris scheme:
 * \f[
 *      \frac{\mathbf{v}^\mathrm{new}-\mathbf{v}^\mathrm{old}}{\Delta t}
 *      \approx \dot{\mathbf{v}} =
 *      \frac{q}{m}(\mathbf{E}+\mathbf{v}\times\mathbf{B})
 *      \approx \frac{q}{m}\left(\mathbf{E}+
 *      \frac{\mathbf{v}^\mathrm{new}+\mathbf{v}^\mathrm{old}}{2}
 *      \times\mathbf{B}\right)
 * \f]
 * 
 * To initialize from time-step \f$ n=0 \f$ a time-staggered grid where
 * velocities are at half-integer time-steps, the particles must be accelerated
 * only half a time-step the first time.
 */
template <typename PopulationType>
double boris(PopulationType &pop, const df::Function &E,
             const std::vector<double> &B, double dt)
{
    auto g_dim = B.size();
    assert(g_dim == 3 && "The algorithm is only valid for 3D.");
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto t_dim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;
    double t_mag2;

    std::vector<double> v_minus(g_dim), v_prime(g_dim), v_plus(g_dim);
    std::vector<double> t(g_dim), s(g_dim);

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
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                }
            }

            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;

            t_mag2 = 0.0;
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                t[i] = tan((dt * q / (2.0 * m)) * B[i]);
                t_mag2 += t[i] * t[i];
            }

            for (std::size_t i = 0; i < g_dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }

            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }

            for (std::size_t i = 0; i < g_dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }

            auto v_prime_cross_s = cross(v_prime, s);
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }

            for (std::size_t i = 0; i < g_dim; ++i)
            {
                pop.cells[cell_id].particles[p_id].v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

/**
 * @brief Accelerates particles in a inhomogeneous magnetic field
 * @param[in,out]   pop     Population
 * @param           E       Electric field
 * @param           B       Magnetic flux density
 * @param           dt      Time-step
 * @return                  Kinetic energy at mid-step
 * @see boris()
 *
 * Same as punc::boris but for inhomogeneous magnetic field.
 */
template <typename PopulationType>
double boris(PopulationType &pop, const df::Function &E,
             const df::Function &B, double dt)
{
    auto g_dim = pop.g_dim;
    assert(g_dim == 3 && "The algorithm is only valid for 3D.");
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto t_dim = mesh->topology().dim();
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;
    double t_mag2;

    std::vector<double> v_minus(g_dim), v_prime(g_dim), v_plus(g_dim);
    std::vector<double> t(g_dim), s(g_dim);

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> coefficients_B(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e)
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
                    Ei[j] += coefficients[i] * basis_matrix[j][i];
                    Bi[j] += coefficients_B[i] * basis_matrix[j][i];
                }
            }
            auto m = particle.m;
            auto q = particle.q;
            auto vel = particle.v;
            t_mag2 = 0.0;
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                t[i] = tan((dt * q / (2.0 * m)) * Bi[i]);
                t_mag2 += t[i] * t[i];
            }
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }
            for (std::size_t i = 0; i < g_dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }
            auto v_prime_cross_s = cross(v_prime, s);
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }
            for (std::size_t i = 0; i < g_dim; ++i)
            {
                pop.cells[cell_id].particles[p_id].v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

/**
 * @brief Move particles
 * @param[in,out]   pop     Population
 * @param           dt      Time-step
 * 
 * Move particles according to:
 * \f[
 *      \frac{\mathbf{x}^\mathrm{new}-\mathbf{x}^\mathrm{old}}{\Delta t} 
 *      \approx \dot\mathbf{x} = \mathbf{v}
 * \f]
 */
template <typename PopulationType>
void move(PopulationType &pop, double dt)
{
    auto g_dim = pop.g_dim;
    for (auto &cell : pop.cells)
    {
        for (auto &particle : cell.particles)
        {
            for (std::size_t j = 0; j < g_dim; ++j)
            {
                particle.x[j] += dt * particle.v[j];
            }
        }
    }
}

// FIXME: Make a separate function for imposing periodic BCs *after* move
// FIXME: This function works only for meshes that have one of the corners at the origin
template <typename PopulationType>
void move_periodic(PopulationType &pop, const double dt, const std::vector<double> &Ld)
{
    auto g_dim = Ld.size();
    for (auto &cell : pop.cells)
    {
        for (auto &particle : cell.particles)
        {
            for (std::size_t j = 0; j < g_dim; ++j)
            {
                particle.x[j] += dt * particle.v[j];
                particle.x[j] -= Ld[j] * floor(particle.x[j] / Ld[j]);
            }
        }
    }
}

static inline std::vector<double> cross(const std::vector<double> &v1,
                                        const std::vector<double> &v2)
{
    std::vector<double> r(v1.size());
    r[0] = v1[1] * v2[2] - v1[2] * v2[1];
    r[1] = -v1[0] * v2[2] + v1[2] * v2[0];
    r[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return r;
}

} // namespace punc

#endif // PUSHER_H
