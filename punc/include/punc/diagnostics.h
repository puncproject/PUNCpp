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
double kinetic_energy(Population &pop);

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
 *  
 * The total potential energy is given by
 * \f[
 *      E_p = \sum_{i=0}^{N}\frac{1}{2}q_i\phi(\mathbf{x}_i),
 * \f]
 * where \f$N\f$ is the number of particles in the simulation domain, and 
 * \f$\mathbf{x}_i\f$ is the position of particle \f$i\f$.
 */
double particle_potential_energy(Population &pop, const df::Function &phi);

} // namespace punc

#endif // DIAGNOSTICS_H
