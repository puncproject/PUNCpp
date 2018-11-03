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

#include "../include/punc/distributor.h"
#include "../ufl/WeightedVolume.h"

#include <dolfin/fem/assemble.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/fem_utils.h>

namespace punc
{

std::vector<double> element_volume(const df::FunctionSpace &V, bool voronoi)
{
    auto num_dofs = V.dim();
    auto dof_indices = df::vertex_to_dof_map(V);
    std::vector<double> volumes(num_dofs, 0.0);

    auto mesh = V.mesh();
    auto t_dim = mesh->topology().dim();
    auto g_dim = mesh->geometry().dim();
    int j = 0;
    for (df::MeshEntityIterator e(*mesh, 0); !e.end(); ++e)
    {
        auto num_cells = e->num_entities(t_dim);
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            df::Cell cell(*mesh, e->entities(t_dim)[i]);
            volumes[dof_indices[j]] += cell.volume();
        }
        j++;
    }
    if(voronoi)
    {
        for (std::size_t i = 0; i < num_dofs; ++i)
        {
            volumes[i] = (g_dim + 1.0) / volumes[i];
        }
    }else{
        for (std::size_t i = 0; i < num_dofs; ++i)
        {
            volumes[i] = 1.0 / volumes[i];
        }
    }

    return volumes;
}

std::vector<double> weighted_element_volume(const df::FunctionSpace &V)
{
    auto g_dim = V.mesh()->geometry().dim();
    std::shared_ptr<df::Form> volume;
    df::PETScVector volume_vector;
    auto V_ptr = std::make_shared<const df::FunctionSpace>(V);
    if (g_dim == 1)
    {
        volume = std::make_shared<WeightedVolume::Form_form1D>(V_ptr);
    }
    else if (g_dim == 2)
    {
        volume = std::make_shared<WeightedVolume::Form_form2D>(V_ptr);
    }
    else if (g_dim == 3)
    {
        volume = std::make_shared<WeightedVolume::Form_form3D>(V_ptr);
    }
    df::assemble(volume_vector, *volume);
    std::vector<double> volumes;
    volume_vector.get_local(volumes);
    for(std::size_t i = 0; i<volumes.size(); ++i)
    {
        volumes[i] = 1.0/volumes[i];
    }
    return volumes;
}

} // namespace punc
