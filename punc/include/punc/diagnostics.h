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
 * @file		diagnostics.h
 * @brief		Kinetic and potential energy calculations
 *
 * Functions for calculating the kinetic and potential energies.
 */

#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "population.h"

namespace punc
{

namespace df = dolfin;

/**
 * @brief Calculates the total kinetic energy
 * @param[in]   pop     Population
 * @return              Total kinetic energy 
 * 
 * The total kinetic energy is given by 
 * \f[
 *      E_k = \sum_{i=0}^{N}\frac{1}{2}m_i \mathbf{v}_i\cdot\mathbf{v}_i,
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain.
 */
template <typename PopulationType>
double kinetic_energy(PopulationType &pop)
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

/**
 * @brief Calculates the total potential energy using FEM approach 
 * @param   phi     Electric potential
 * @param   rho     Volume charge density
 * @return          Total potential energy
 *  
 * The total potential energy is given by
 * \f[
 *      E_p = \int_{\Omega} \, \phi \,\rho \, \mathrm{d}x,
 * \f]
 * where \f$\Omega\f$ is the simulation domain.
 */
double mesh_potential_energy(df::Function &phi, df::Function &rho);

/**
 * @brief Calculates the total potential energy by interpolating the electric potential 
 * @param[in]   pop     Population
 * @param       phi     Electric potential
 * @return              Total potential energy
 *  @see particle_potential_energy_cg1
 * 
 * The total potential energy is given by
 * \f[
 *      E_p = \sum_{i=0}^{N}\frac{1}{2}q_i\phi(\mathbf{x}_i),
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain, and 
 * \f$\mathbf{x}_i\f$ is the position of particle \f$i\f$.
 */
template <typename PopulationType>
double particle_potential_energy(PopulationType &pop, const df::Function &phi)
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

/**
 * @brief Calculates the total potential energy by interpolating the electric potential in CG1 function space
 * @param[in]   pop     Population
 * @param       phi     Electric potential in CG1
 * @return              Total potential energy
 * @see particle_potential_energy
 * 
 * The total potential energy is given by
 * \f[
 *      E_p = \sum_{i=0}^{N}\frac{1}{2}q_i\phi(\mathbf{x}_i),
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain, and 
 * \f$\mathbf{x}_i\f$ is the position of particle \f$i\f$.
 */
template <typename PopulationType>
double particle_potential_energy_cg1(PopulationType &pop, const df::Function &phi)
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

/**
 * @brief                 Volumetric number density
 * @param[in]   Q         FunctionSpace DG0
 * @param       pop       Population
 * @param      ne, ni     Function - the volumetric number densities
 * @param      dt         timestep
 * @param      tau        relaxation time 
 * @see distribute()
 * 
 * Calculates the volumetric number density in \f$\mathrm{DG}_0\f$ function 
 * space. The number density in each cell \f$T_k\f$, is simply 
 * calculated by adding together the number of particles of each species inside 
 * the cell, and then dividing the total number inside the cell by the volume of 
 * the cell:
 * 
 * \f[
 *       \n_{e,k} = \frac{1}{\mathrm{Vol}(T_k)}\sum_{p} (q_p==-1),
 * \f]
 */
template <typename PopulationType>
void density_dg0(const df::FunctionSpace &Q, PopulationType &pop,
                 df::Function &ne, df::Function &ni, double dt, double tau)
{
    double w = 1.0 - exp(-dt/tau);
    auto ne_vec = ne.vector();
    auto ni_vec = ni.vector();

    std::vector<double> ne0(ne_vec->size());
    std::vector<double> ni0(ni_vec->size());
    ne_vec->get_local(ne0);
    ni_vec->get_local(ni0);

    for (auto &cell : pop.cells)
    {
        auto dof_id = Q.dofmap()->cell_dofs(cell.id);
        double accum_e = 0.0, accum_i = 0.0;
        for (auto &particle : cell.particles)
        {
            if (particle.q>0){
                accum_i += 1;
            }else{
                accum_e += 1;
            }
        }
        ne0[dof_id[0]] = w * (accum_e / cell.volume()) + (1.0 - w) * ne0[dof_id[0]];
        ni0[dof_id[0]] = w * (accum_i / cell.volume()) + (1.0 - w) * ni0[dof_id[0]];
    }
    ne.vector()->set_local(ne0);
    ni.vector()->set_local(ni0);
}

} // namespace punc

#endif // DIAGNOSTICS_H
