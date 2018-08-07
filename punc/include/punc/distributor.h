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
 * @file		distributor.h
 * @brief		Volume charge density distribution
 *
 * Functions for finding the volume charge distribution based on the position of the particles.
 */

#ifndef DISTRIBUTOR_H
#define DISTRIBUTOR_H

#include "population.h"

namespace punc
{

namespace df = dolfin;

/**
 * @brief 
 * @param[in]   V         FunctionSpace
 * @param       voronoi   element type
 * @return                Volume of each element
 * @see weighted_element_volume
 *  
 * Calculates the volume of each patch \f$M_j\f$, where \f$M_j\f$ is the set of 
 * all the cells sharing vertex \f$x_j\f$. If voronoi parameter is true, then an 
 * approximated value for the volume of the Voronoi cell defined at \f$x_j\f$ is
 * given by 
 * \f[
 *      \mathrm{Vol}(R_j) = \frac{1}{D+1}\sum_{i=1}^{k}\mathrm{Vol}(T_k),
 * \f]
 * where \f$T_k\in M_j\f$ has \f$x_j\f$ as one of its vertices.
 */
std::vector<double> element_volume(const df::FunctionSpace &V, bool voronoi = true);

/**
 * @brief 
 * @param[in]   V         FunctionSpace
 * @return                Volume of each element
 * @see element_volume
 *  
 * Calculates the weighted volume of each patch \f$M_j\f$, where \f$M_j\f$ is 
 * the set of all the cells sharing vertex \f$x_j\f$, and returns the reciprocal 
 * values for each element. The volume of each patch is weighted by the 
 * corresponding finite element basis function of continuous Lagrange space of 
 * order 1, CG1. It can be shown that this method is mathematically equivalent 
 * to the approximated Voronoi cell volumes. 
 */
std::vector<double> weighted_element_volume(const df::FunctionSpace &V);

/**
 * @brief                 Volume charge density
 * @param[in]   V         FunctionSpace CG1
 * @param       pop       Population
 * @param       dv_inv    Vector containing the volumes of each element (e.g. Voronoi cell)
 * @return      rho       Function - the volume charge density 
 * @see distribute_cg1, distribute_dg0()
 * 
 * Calculates the volume charge density in \f$\mathrm{CG}_1\f$ function 
 * space. The volume charge density at each mesh vertex \f$\mathbf{x}_j\f$, is 
 * calculated by interpolating the charge of each particle inside all the cells 
 * sharing vertex \f$\mathbf{x}_j\f$, i.e. the patch \f$\mathcal{M}_j\f$. The
 * interpolation is done by evaluating the \f$\mathrm{CG}_1\f$ basis 
 * function \f$\psi_j\f$, at the particle position \f$\mathbf{x}_{p}\f$. The 
 * interpolated charge at each mesh vertex is divided by a proper volume 
 * \f$\mathcal{V}_j\f$ associated with \f$\mathbf{x}_j\f$, to obtain the volume 
 * charge density: 
 * 
 * \f[
 *       \rho_{j} = \frac{1}{\mathcal{V}_j}\sum_{p}q_p\psi_j(\mathbf{x}_{p}).
 * \f]
 */
template <std::size_t len>
df::Function distribute(const df::FunctionSpace &V,
                        Population<len> &pop,
                        const std::vector<double> &dv_inv)
{
    auto mesh = V.mesh();
    auto tdim = mesh->topology().dim();
    df::Function rho(std::make_shared<const df::FunctionSpace>(V));
    auto rho_vec = rho.vector();
    std::size_t len_rho = rho_vec->size();
    std::vector<double> rho0(len_rho);
    rho_vec->get_local(rho0);

    auto element = V.element();
    auto s_dim = element->space_dimension();

    std::vector<double> basis_matrix(s_dim);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();
        auto dof_id = V.dofmap()->cell_dofs(cell_id);

        std::vector<double> basis(1);
        std::vector<double> accum(s_dim, 0.0);

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];

            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x,
                                        vertex_coordinates.data(),
                                        cell_orientation);
                basis_matrix[i] = basis[0];
                accum[i] += particle.q * basis_matrix[i];
            }
        }
        for (std::size_t i = 0; i < s_dim; ++i)
        {
            rho0[dof_id[i]] += accum[i];
        }
    }
    for (std::size_t i = 0; i < len_rho; ++i)
    {
        rho0[i] *= dv_inv[i];
    }
    rho.vector()->set_local(rho0);
    return rho;
}

/**
 * @brief                 Volume charge density in CG1
 * @param[in]   V         FunctionSpace CG1
 * @param       pop       Population
 * @param       dv_inv    Vector containing the volumes of each element (e.g. Voronoi cell)
 * @return      rho       Function (CG1) - the volume charge density 
 * @see distribute, distribute_dg0()
 * 
 * Calculates the volume charge density in \f$\mathrm{CG}_1\f$ function 
 * space. This function work only for \f$\mathrm{CG}_1\f$ function space, and it 
 * is more efficient than the "distribute" function. The volume charge density 
 * at each mesh vertex \f$\mathbf{x}_j\f$, is calculated by interpolating the 
 * charge of each particle inside all the cells sharing vertex \f$\mathbf{x}_j\f$, 
 * i.e. the patch \f$\mathcal{M}_j\f$. The interpolation is done by evaluating 
 * the \f$\mathrm{CG}_1\f$ basis function \f$\psi_j\f$, at the particle position 
 * \f$\mathbf{x}_{p}\f$. The interpolated charge at each mesh vertex is divided 
 * by a proper volume \f$\mathcal{V}_j\f$ associated with \f$\mathbf{x}_j\f$, to 
 * obtain the volume charge density: 
 * 
 * \f[
 *       \rho_{j} = \frac{1}{\mathcal{V}_j}\sum_{p}q_p\psi_j(\mathbf{x}_{p}).
 * \f]
 */
template <std::size_t len>
df::Function distribute_cg1(const df::FunctionSpace &V,
                            Population<len> &pop,
                            const std::vector<double> &dv_inv)
{
    auto mesh = V.mesh();
    df::Function rho(std::make_shared<const df::FunctionSpace>(V));
    auto rho_vec = rho.vector();
    std::size_t len_rho = rho_vec->size();
    std::vector<double> rho0(len_rho);
    rho_vec->get_local(rho0);

    auto element = V.element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);
    auto n_dim = s_dim / v_dim;

    double cell_coords[n_dim];

    for (auto &cell : pop.cells)
    {
        auto dof_id = V.dofmap()->cell_dofs(cell.id);
        std::vector<double> accum(n_dim, 0.0);
        for (auto &particle : cell.particles)
        {
            matrix_vector_product(&cell_coords[0], cell.basis_matrix.data(),
                                  particle.x, n_dim, n_dim);

            for (std::size_t i = 0; i < n_dim; ++i)
            {
                accum[i] += particle.q * cell_coords[i];
            }
        }

        for (std::size_t i = 0; i < s_dim; ++i)
        {
            rho0[dof_id[i]] += accum[i];
        }
    }
    for (std::size_t i = 0; i < len_rho; ++i)
    {
        rho0[i] *= dv_inv[i];
    }
    rho.vector()->set_local(rho0);
    return rho;
}

/**
 * @brief                 Volume charge density
 * @param[in]   Q         FunctionSpace DG0
 * @param       pop       Population
 * @return      rho       Function - the volume charge density 
 * @see distribute()
 * 
 * Calculates the volume charge density in \f$\mathrm{DG}_0\f$ function 
 * space. The volume charge density in each cell \f$T_k\f$, is simply 
 * calculated by adding together the charge of each particle inside the cell, 
 * and then dividing the total charge inside the cell by the volume of the cell:
 * 
 * \f[
 *       \rho_{k} = \frac{1}{\mathrm{Vol}(T_k)}\sum_{p}q_p,
 * \f]
 */
template <std::size_t len>
df::Function distribute_dg0(const df::FunctionSpace &Q, Population<len> &pop)
{
    df::Function rho(std::make_shared<const df::FunctionSpace>(Q));
    auto rho_vec = rho.vector();
    std::size_t len_rho = rho_vec->size();
    std::vector<double> rho0(len_rho);
    rho_vec->get_local(rho0);

    for (auto &cell : pop.cells)
    {
        auto dof_id = Q.dofmap()->cell_dofs(cell.id);
        double accum = 0.0;
        for (auto &particle : cell.particles)
        {
            accum += particle.q;
        }
        rho0[dof_id[0]] = accum / cell.volume();
    }
    rho.vector()->set_local(rho0);
    return rho;
}

// FIXME: This function is redundant. Should be removed.
template <std::size_t len>
df::Function distribute_dg0_old(const df::FunctionSpace &Q, Population<len> &pop)
{
    auto mesh = Q.mesh();
    auto tdim = mesh->topology().dim();
    df::Function rho(std::make_shared<const df::FunctionSpace>(Q));
    auto rho_vec = rho.vector();
    std::size_t len_rho = rho_vec->size();
    std::vector<double> rho0(len_rho);
    rho_vec->get_local(rho0);

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        auto dof_id = Q.dofmap()->cell_dofs(cell_id);
        double accum = 0.0;

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            accum += particle.q;
        }
        rho0[dof_id[0]] = accum / _cell.volume();
    }
    rho.vector()->set_local(rho0);
    return rho;
}

} // namespace punc

#endif // DISTRIBUTOR_H
