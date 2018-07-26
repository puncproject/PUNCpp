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
 *  
 * Calculates the volume of each patch \f$M_j\f$, where \f$M_j\f$ is the set of 
 * all the cells sharing vertex \f$x_j\f$. If voronoi parameter is true, then an 
 * approximated value for the volume of the Voronoi cell defined at \f$x_j\f$ is
 * given by 
 * \f[
 *      \operatorname{Vol}(R_j) = \frac{1}{D+1}\sum_{i=1}^{k}\operatorname{Vol}(T_k),
 * \f]
 * where \f$T_k\in M_j\f$ has \f$x_j\f$ as one of its vertices.
 */
std::vector<double> element_volume(const df::FunctionSpace &V, bool voronoi = false);

/**
 * @brief                 Volume charge density
 * @param[in]   V         FunctionSpace CG1
 * @param       pop       Population
 * @param       dv_inv    Vector containing the volumes of each element (e.g. Voronoi cell)
 * @return      rho       Function - the volume charge density 
 * @see distribute_dg0()
 * 
 * Calculates the volume charge density in \f$\operatorname{CG}_1\f$ function 
 * space. The volume charge density at each mesh vertex \f$\mathbf{x}_j\f$, is 
 * calculated by interpolating the charge of each particle inside all the cells 
 * sharing vertex \f$\mathbf{x}_j\f$, i.e. the patch \f$\mathcal{M}_j\f$. The
 * interpolation is done by evaluating the \f$\operatorname{CG}_1\f$ basis 
 * function \f$\psi_j\f$, at the particle position \f$\mathbf{x}_{p}\f$. The 
 * interpolated charge at each mesh vertex is divided by a proper volume 
 * \f$\mathcal{V}_j\f$ associated with \f$\mathbf{x}_j\f$, to obtain the volume 
 * charge density: 
 * 
 * \f[
 *       \rho_{j} = \frac{1}{\mathcal{V}_j}\sum_{p}q_p\psi_j(\mathbf{x}_{p}).
 * \f]
 */
df::Function distribute(const df::FunctionSpace &V,
                        Population &pop,
                        const std::vector<double> &dv_inv);

/**
 * @brief                 Volume charge density
 * @param[in]   Q         FunctionSpace DG0
 * @param       pop       Population
 * @return      rho       Function - the volume charge density 
 * @see distribute()
 * 
 * Calculates the volume charge density in \f$\operatorname{DG}_0\f$ function 
 * space. The volume charge density in each cell \f$T_k\f$, is simply 
 * calculated by adding together the charge of each particle inside the cell, 
 * and then dividing the total charge inside the cell by the volume of the cell:
 * 
 * \f[
 *       \rho_{k} = \frac{1}{\operatorname{Vol}(T_k)}\sum_{p}q_p,
 * \f]
 */
df::Function distribute_dg0(const df::FunctionSpace &Q, Population &pop);

} // namespace punc

#endif // DISTRIBUTOR_H
