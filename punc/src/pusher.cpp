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
 * @file		pusher.cpp
 * @brief		Particle pusher
 *
 * Functions for pushing (accelerating and moving) particles.
 */

#include "../include/punc/pusher.h"

namespace punc
{

/*******************************************************************************
 * LOCAL DECLARATIONS
 ******************************************************************************/

/**
 * @brief Cross product af two 3-vectors
 * @param   v1  vector
 * @param   v2  vector
 * @return  v1 cross v2
 */
static inline std::vector<double> cross(const std::vector<double> &v1,
                                        const std::vector<double> &v2);
/**
 * @brief Get the linear expansion coefficients for a function at a cell
 * @param[out]  coefficients    Expansion coefficients
 * @param       coords          The (x,y,z) coordinates of the vertices
 * @param       values          The function values at the vertices
 *
 * A linear function on a tetrahedral cell can be written as a function of
 * barycentric coordinates (lambda_1, labmda_2, lambda_3, lambda_4):
 *
 *      f(lambda_1, lambda_2, lambda_3, lambda_4) = f_1*lambda_1
 *                                                + f_2*lambda_2
 *                                                + f_3*lambda_3
 *                                                + f_4*lambda_4
 *
 * Where f_i is the value of f at vertex i. If A is the matrix converting from
 * cartesian coordinates x_j to barycentric coordinates lambda_i;
 *
 *      lambda_i = sum_j A_ij x_j
 *
 * where the vector x = (1, x, y, z) (mind the 1 as the first element) one can
 * write:
 *
 *      f(x) = sum_j c_j x_j
 *
 * where
 * 
 *      c_j = sum_i A_ij f_j    // FIXME: Do proper LaTeX-ninja
 *
 * This function returns the coefficients c_j, which can be reused on all
 * particles in the same cell.
 *
 * To use this with vector functions, one must use it once per component of
 * the vector as values.
 */
static inline void get_expcoeffs_3d(double *coeffs, const double *coords, 
                                    const double *values);

static inline void get_expcoeffs_2d(double *coeffs, const double *coords, 
                                    const double *values);

static inline void get_coord_transform_1d(double *transform,
                                          const double *coords);

static inline void get_coord_transform_2d(double *transform,
                                          const double *coords);

static inline void get_coord_transform_3d(double *transform,
                                          const double *coords);

static inline void matrix_vector_product(double *y, const double *A,
                                         const double *x, std::size_t m,
                                         std::size_t n);

/*******************************************************************************
 * GLOBAL DEFINITIONS
 ******************************************************************************/

double accel(Population &pop, const df::Function &E, double dt)
{
    auto W = E.function_space();
    auto mesh = W->mesh();
    auto tdim = mesh->topology().dim(); // FIXME: Shouldn't this be geometric dimension?
    auto element = W->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double KE = 0.0;

    std::vector<double> basis_matrix(v_dim*s_dim);
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates(tdim);
    std::vector<double> Ei(v_dim);

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
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
                                        particle.x.data(),
                                        vertex_coordinates.data(),
                                        cell_orientation);
            
            for (std::size_t j = 0; j < v_dim; j++) {
                Ei[j] = 0;
                for (std::size_t i = 0; i < s_dim; ++i) {
                    Ei[j] += coefficients[i] * basis_matrix[i*v_dim+j];
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

double accel_cg1(Population &pop, const df::Function &E, double dt) {

    auto W = E.function_space();
    auto mesh = W->mesh();
    auto element = W->element();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim/g_dim; // Number of vertices

    double KE = 0.0;

    std::vector<double> vertex_coordinates(t_dim);
    double Ei[v_dim];
    double coeffs[s_dim];
    double values[s_dim];

    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e){

        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(values, *element, _cell, vertex_coordinates.data(), ufc_cell);

        for(std::size_t j=0; j<v_dim; j++){
            get_expcoeffs_3d(&coeffs[n_dim*j],
                    vertex_coordinates.data(), &values[n_dim*j]);
        }

        for(auto &particle : pop.cells[cell_id].particles) {

            double m = particle.m;
            double q = particle.q;
            auto &vel = particle.v;
            
            for (std::size_t j = 0; j < v_dim; j++){
                double *coeffs_d = &coeffs[n_dim*j];
                Ei[j]  = coeffs_d[0];
                Ei[j] += coeffs_d[1]*particle.x[0];
                Ei[j] += coeffs_d[2]*particle.x[1];
                Ei[j] += coeffs_d[3]*particle.x[2];
            }

            for (std::size_t j = 0; j < v_dim; j++) {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
                particle.v[j] += Ei[j];
            }
        }
    }
    return KE;
}

double accel_cg_new(Population &pop, const df::Function &E, double dt)
{

    auto W = E.function_space();
    auto mesh = W->mesh();
    auto element = W->element();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / g_dim; // Number of vertices

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
                                  particle.x.data(), n_dim, n_dim);

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] = 0.0;
                for (std::size_t i = 0; i < n_dim; ++i)
                {
                    Ei[j] += coeffs[i] * values[j * n_dim + i];
                }
            }

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
                particle.v[j] += Ei[j];
            }
        }
    }
    return KE;
}

double accel_cg_2d(Population &pop, const df::Function &E, double dt)
{

    auto W = E.function_space();
    auto mesh = W->mesh();
    auto element = W->element();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / g_dim; // Number of vertices

    double KE = 0.0;

    std::vector<double> vertex_coordinates(t_dim);
    double Ei[v_dim];
    double coeffs[n_dim];
    double values[s_dim];
    double coordinates[s_dim];
    double transform[n_dim * n_dim];
    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e)
    {

        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);

        const std::size_t num_vertices = _cell.num_vertices();
        const unsigned int *vertices = _cell.entities(0);

        for (std::size_t i = 0; i < num_vertices; i++)
        {
            for (std::size_t j = 0; j < g_dim; j++)
            {
                coordinates[i * g_dim + j] = mesh->geometry().x(vertices[i])[j];
            }
        }

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(values, *element, _cell, &coordinates[0], ufc_cell);

        get_coord_transform_2d(&transform[0], &coordinates[0]);

        for (auto &particle : pop.cells[cell_id].particles)
        {

            double m = particle.m;
            double q = particle.q;
            auto &vel = particle.v;

            matrix_vector_product(&coeffs[0], &transform[0], particle.x.data(),
                                  n_dim, n_dim);

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] = 0.0;
                for (std::size_t i = 0; i < n_dim; ++i)
                {
                    Ei[j] += coeffs[i] * values[j * n_dim + i];
                }
            }

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
                particle.v[j] += Ei[j];
            }
        }
    }
    return KE;
}

double accel_cg(Population &pop, const df::Function &E, double dt)
{

    auto W = E.function_space();
    auto mesh = W->mesh();
    auto element = W->element();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / g_dim; // Number of vertices

    double KE = 0.0;

    std::vector<double> vertex_coordinates(t_dim);
    double Ei[v_dim];
    double coeffs[n_dim];
    double values[s_dim];
    double coordinates[s_dim];
    for (df::MeshEntityIterator e(*mesh, t_dim); !e.end(); ++e)
    {

        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);

        const std::size_t num_vertices = _cell.num_vertices();
        const unsigned int *vertices = _cell.entities(0);

        for (std::size_t i = 0; i < num_vertices; i++)
        {
            for (std::size_t j = 0; j < g_dim; j++)
            {
                coordinates[i * g_dim + j] = mesh->geometry().x(vertices[i])[j];
            }
        }

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        E.restrict(values, *element, _cell, &coordinates[0], ufc_cell);

        for (auto &particle : pop.cells[cell_id].particles)
        {

            double m = particle.m;
            double q = particle.q;
            auto &vel = particle.v;

            get_expcoeffs_3d(&coeffs[0], &coordinates[0], particle.x.data());

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] = 0.0;
                for (std::size_t i = 0; i < n_dim; ++i)
                {
                    Ei[j] += coeffs[i] * values[j * n_dim + i];
                }
            }

            for (std::size_t j = 0; j < v_dim; j++)
            {
                Ei[j] *= dt * (q / m);
                KE += 0.5 * m * vel[j] * (vel[j] + Ei[j]);
                particle.v[j] += Ei[j];
            }
        }
    }
    return KE;
}

double boris(Population &pop, const df::Function &E, 
             const std::vector<double> &B, double dt)
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
            for (std::size_t i = 0; i < dim; ++i)
            {
                t[i] = tan((dt * q / (2.0 * m)) * B[i]);
                t_mag2 += t[i] * t[i];
            }

            for (std::size_t i = 0; i < dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }

            for (std::size_t i = 0; i < dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }

            for (std::size_t i = 0; i < dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (std::size_t i = 0; i < dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }

            auto v_prime_cross_s = cross(v_prime, s);
            for (std::size_t i = 0; i < dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }

            for (std::size_t i = 0; i < dim; ++i)
            {
                pop.cells[cell_id].particles[p_id].v[i] = v_plus[i] + 0.5 * dt * (q / m) * Ei[i];
            }
        }
    }
    return KE;
}

double boris(Population &pop, const df::Function &E,
             const df::Function &B, double dt)
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
            for (std::size_t i = 0; i < dim; ++i)
            {
                t[i] = tan( (dt * q / (2.0 * m)) * Bi[i]);
                t_mag2 += t[i]*t[i];
            }
            for (std::size_t i = 0; i < dim; ++i)
            {
                s[i] = 2 * t[i] / (1 + t_mag2);
            }
            for (std::size_t i = 0; i < dim; ++i)
            {
                v_minus[i] = vel[i] + 0.5 * dt * (q / m) * Ei[i];
            }
            for (std::size_t i = 0; i < dim; i++)
            {
                KE += 0.5 * m * v_minus[i] * v_minus[i];
            }

            auto v_minus_cross_t = cross(v_minus, t);
            for (std::size_t i = 0; i < dim; ++i)
            {
                v_prime[i] = v_minus[i] + v_minus_cross_t[i];
            }
            auto v_prime_cross_s = cross(v_prime, s);
            for (std::size_t i = 0; i < dim; ++i)
            {
                v_plus[i] = v_minus[i] + v_prime_cross_s[i];
            }
            for (std::size_t i = 0; i < dim; ++i)
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

void move(Population &pop, double dt)
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

/*******************************************************************************
 * LOCAL DEFINITIONS
 ******************************************************************************/

static inline std::vector<double> cross(const std::vector<double> &v1,
                                        const std::vector<double> &v2)
{
    std::vector<double> r(v1.size());
    r[0] = v1[1] * v2[2] - v1[2] * v2[1];
    r[1] = -v1[0] * v2[2] + v1[2] * v2[0];
    r[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return r;
}

static inline void get_coord_transform_1d(double *transform,
                                          const double *coords)
{
    double x1 = coords[0];
    double x2 = coords[1];

    double det = x2 - x1;

    transform[0] = x2 / det;
    transform[1] = -1.0 / det;
    transform[2] = -x1 / det;
    transform[3] = 1.0 / det;
}

static inline void get_coord_transform_2d(double *transform,
                                          const double *coords)
{
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];

    double det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

    transform[0] = (x2 * y3 - x3 * y2) / det;
    transform[1] = (y2 - y3) / det;
    transform[2] = (x3 - x2) / det;
    transform[3] = (x3 * y1 - x1 * y3) / det;
    transform[4] = (y3 - y1) / det;
    transform[5] = (x1 - x3) / det;
    transform[6] = (x1 * y2 - x2 * y1) / det;
    transform[7] = (y1 - y2) / det;
    transform[8] = (x2 - x1) / det;
}

static inline void get_coord_transform_3d(double *transform,
                                          const double *coords)
{
    double x1 = coords[0];
    double y1 = coords[1];
    double z1 = coords[2];
    double x2 = coords[3];
    double y2 = coords[4];
    double z2 = coords[5];
    double x3 = coords[6];
    double y3 = coords[7];
    double z3 = coords[8];
    double x4 = coords[9];
    double y4 = coords[10];
    double z4 = coords[11];

    double x12 = x1 - x2;
    double y12 = y1 - y2;
    double z12 = z1 - z2;
    double x13 = x1 - x3;
    double y13 = y1 - y3;
    double z13 = z1 - z3;
    double x14 = x1 - x4;
    double y14 = y1 - y4;
    double z14 = z1 - z4;
    double x21 = x2 - x1;
    double y21 = y2 - y1;
    double z21 = z2 - z1;

    double x24 = x2 - x4;
    double y24 = y2 - y4;
    double z24 = z2 - z4;
    double x31 = x3 - x1;
    double y31 = y3 - y1;
    double z31 = z3 - z1;
    double x32 = x3 - x2;
    double y32 = y3 - y2;
    double z32 = z3 - z2;
    double x34 = x3 - x4;
    double y34 = y3 - y4;
    double z34 = z3 - z4;

    double x42 = x4 - x2;
    double y42 = y4 - y2;
    double z42 = z4 - z2;
    double x43 = x4 - x3;
    double y43 = y4 - y3;
    double z43 = z4 - z3;

    double V01 = (x2 * (y3 * z4 - y4 * z3) + x3 * (y4 * z2 - y2 * z4) + x4 * (y2 * z3 - y3 * z2)) / 6.0;
    double V02 = (x1 * (y4 * z3 - y3 * z4) + x3 * (y1 * z4 - y4 * z1) + x4 * (y3 * z1 - y1 * z3)) / 6.0;
    double V03 = (x1 * (y2 * z4 - y4 * z2) + x2 * (y4 * z1 - y1 * z4) + x4 * (y1 * z2 - y2 * z1)) / 6.0;
    double V04 = (x1 * (y3 * z2 - y2 * z3) + x2 * (y1 * z3 - y3 * z1) + x3 * (y2 * z1 - y1 * z2)) / 6.0;
    double V = V01 + V02 + V03 + V04;
    double V6 = 6 * V;

    transform[0] = V01 / V;
    transform[4] = V02 / V;
    transform[8] = V03 / V;
    transform[12] = V04 / V;

    transform[1] = (y42 * z32 - y32 * z42) / V6;  // a1 / 6V
    transform[5] = (y31 * z43 - y34 * z13) / V6;  // a2 / 6V
    transform[9] = (y24 * z14 - y14 * z24) / V6;  // a3 / 6V
    transform[13] = (y13 * z21 - y12 * z31) / V6; // a4 / 6V

    transform[2] = (x32 * z42 - x42 * z32) / V6;  // b1 / 6V
    transform[6] = (x43 * z31 - x13 * z34) / V6;  // b2 / 6V
    transform[10] = (x14 * z24 - x24 * z14) / V6; // b3 / 6V
    transform[14] = (x21 * z13 - x31 * z12) / V6; // b4 / 6V

    transform[3] = (x42 * y32 - x32 * y42) / V6;  // c1 / 6V
    transform[7] = (x31 * y43 - x34 * y13) / V6;  // c2 / 6V
    transform[11] = (x24 * y14 - x14 * y24) / V6; // c3 / 6V
    transform[15] = (x13 * y21 - x12 * y31) / V6; // c4 / 6V
}

static inline void matrix_vector_product(double *y, const double *A,
                                         const double *x, std::size_t n,
                                         std::size_t m)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        y[i] = A[i * m];
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < m - 1; ++j)
        {
            y[i] += A[i * m + j + 1] * x[j];
        }
    }
}

static inline void get_expcoeffs_2d(double *coeffs,
                                    const double *coords, const double *values)
{
    double x1 = coords[0];
    double y1 = coords[1];
    double x2 = coords[2];
    double y2 = coords[3];
    double x3 = coords[4];
    double y3 = coords[5];

    double det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

    double A11 = x2 * y3 - x3 * y2;
    double A12 = y2 - y3;
    double A13 = x3 - x2;
    double A21 = x3 * y1 - x1 * y3;
    double A22 = y3 - y1;
    double A23 = x1 - x3;
    double A31 = x1 * y2 - x2 * y1;
    double A32 = y1 - y2;
    double A33 = x2 - x1;

    coeffs[0] = (A11 + values[0] * A12 + values[1] * A13) / det;
    coeffs[1] = (A21 + values[0] * A22 + values[1] * A23) / det;
    coeffs[2] = (A31 + values[0] * A32 + values[1] * A33) / det;

    // coeffs[0] = ((y2-y3)*(values[0]-x3)+(x3-x2)*(values[1]-y3))/det;
    // coeffs[1] = ((y3-y1)*(values[0]-x3)+(x1-x3)*(values[1]-y3))/det;
    // coeffs[2] = 1.0 - coeffs[0] - coeffs[1];
}

static inline void get_expcoeffs_3d(double *coeffs,
                                    const double *coords, const double *values)
{

    double x1 = coords[0];
    double y1 = coords[1];
    double z1 = coords[2];
    double x2 = coords[3];
    double y2 = coords[4];
    double z2 = coords[5];
    double x3 = coords[6];
    double y3 = coords[7];
    double z3 = coords[8];
    double x4 = coords[9];
    double y4 = coords[10];
    double z4 = coords[11];

    // I'm using lots of intermediate variables to make it possible for me to
    // write almost mathematical notation, and easier to compare with
    // literature. The compiler should optimize these away.
    // See 9.1.6 in
    // https://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf

    double x12 = x1 - x2;
    double y12 = y1 - y2;
    double z12 = z1 - z2;
    double x13 = x1 - x3;
    double y13 = y1 - y3;
    double z13 = z1 - z3;
    double x14 = x1 - x4;
    double y14 = y1 - y4;
    double z14 = z1 - z4;
    double x21 = x2 - x1;
    double y21 = y2 - y1;
    double z21 = z2 - z1;
    //double x23 = x2-x3;   double y23 = y2-y3;   double z23 = z2-z3;
    double x24 = x2 - x4;
    double y24 = y2 - y4;
    double z24 = z2 - z4;
    double x31 = x3 - x1;
    double y31 = y3 - y1;
    double z31 = z3 - z1;
    double x32 = x3 - x2;
    double y32 = y3 - y2;
    double z32 = z3 - z2;
    double x34 = x3 - x4;
    double y34 = y3 - y4;
    double z34 = z3 - z4;
    //double x41 = x4-x1;   double y41 = y4-y1;   double z41 = z4-z1;
    double x42 = x4 - x2;
    double y42 = y4 - y2;
    double z42 = z4 - z2;
    double x43 = x4 - x3;
    double y43 = y4 - y3;
    double z43 = z4 - z3;

    double V01 = (x2 * (y3 * z4 - y4 * z3) + x3 * (y4 * z2 - y2 * z4) + x4 * (y2 * z3 - y3 * z2)) / 6.0;
    double V02 = (x1 * (y4 * z3 - y3 * z4) + x3 * (y1 * z4 - y4 * z1) + x4 * (y3 * z1 - y1 * z3)) / 6.0;
    double V03 = (x1 * (y2 * z4 - y4 * z2) + x2 * (y4 * z1 - y1 * z4) + x4 * (y1 * z2 - y2 * z1)) / 6.0;
    double V04 = (x1 * (y3 * z2 - y2 * z3) + x2 * (y1 * z3 - y3 * z1) + x3 * (y2 * z1 - y1 * z2)) / 6.0;
    double V = V01 + V02 + V03 + V04;
    double V6 = 6 * V;

    double A11 = V01 / V;
    double A12 = V02 / V;
    double A13 = V03 / V;
    double A14 = V04 / V;

    double A21 = (y42 * z32 - y32 * z42) / V6; // a1 / 6V
    double A22 = (y31 * z43 - y34 * z13) / V6; // a2 / 6V
    double A23 = (y24 * z14 - y14 * z24) / V6; // a3 / 6V
    double A24 = (y13 * z21 - y12 * z31) / V6; // a4 / 6V

    double A31 = (x32 * z42 - x42 * z32) / V6; // b1 / 6V
    double A32 = (x43 * z31 - x13 * z34) / V6; // b2 / 6V
    double A33 = (x14 * z24 - x24 * z14) / V6; // b3 / 6V
    double A34 = (x21 * z13 - x31 * z12) / V6; // b4 / 6V

    double A41 = (x42 * y32 - x32 * y42) / V6; // c1 / 6V
    double A42 = (x31 * y43 - x34 * y13) / V6; // c2 / 6V
    double A43 = (x24 * y14 - x14 * y24) / V6; // c3 / 6V
    double A44 = (x13 * y21 - x12 * y31) / V6; // c4 / 6V

    const double *f = values;
    coeffs[0] = A11 + A21 * f[0] + A31 * f[1] + A41 * f[2];
    coeffs[1] = A12 + A22 * f[0] + A32 * f[1] + A42 * f[2];
    coeffs[2] = A13 + A23 * f[0] + A33 * f[1] + A43 * f[2];
    coeffs[3] = A14 + A24 * f[0] + A34 * f[1] + A44 * f[2];
    // coeffs[0] = A11*f[0] + A21*f[1] + A31*f[2] + A41*f[3];
    // coeffs[1] = A12*f[0] + A22*f[1] + A32*f[2] + A42*f[3];
    // coeffs[2] = A13*f[0] + A23*f[1] + A33*f[2] + A43*f[3];
    // coeffs[3] = A14*f[0] + A24*f[1] + A34*f[2] + A44*f[3];
}

} // namespace punc
