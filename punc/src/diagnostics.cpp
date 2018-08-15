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
 * @file		diagnostics.cpp
 * @brief		Kinetic and potential energy calculations
 *
 * Functions for calculating the kinetic and potential energies.
 */

#include "../include/punc/diagnostics.h"
#include "../ufl/Energy.h"

namespace punc
{

double mesh_potential_energy(df::Function &phi, df::Function &rho)
{
    auto mesh = phi.function_space()->mesh();
    auto dim = mesh->geometry().dim();
    auto phi_ptr = std::make_shared<df::Function>(phi);
    auto rho_ptr = std::make_shared<df::Function>(rho);
    std::shared_ptr<df::Form> energy;
    if (dim == 1)
    {
        energy = std::make_shared<Energy::Form_0>(mesh, phi_ptr, rho_ptr);
    }
    else if (dim == 2)
    {
        energy = std::make_shared<Energy::Form_1>(mesh, phi_ptr, rho_ptr);
    }
    else if (dim == 3)
    {
        energy = std::make_shared<Energy::Form_2>(mesh, phi_ptr, rho_ptr);
    }
    return 0.5 * df::assemble(*energy);
}

} // namespace punc
