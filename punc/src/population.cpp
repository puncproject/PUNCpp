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

#include "../include/punc/population.h"
#include <dolfin/geometry/Point.h>
#include <dolfin/geometry/BoundingBoxTree.h>

namespace punc
{

signed long int locate(std::shared_ptr<const df::Mesh> mesh, const double *x)
{
    auto g_dim = mesh->geometry().dim();
    auto tree = mesh->bounding_box_tree();
    df::Point p(g_dim, x);
    unsigned int cell_id = tree->compute_first_entity_collision(p);

    if (cell_id == UINT32_MAX)
    {
        return -1;
    }
    else
    {
        return cell_id;
    }
}

Species::Species(double charge, double mass, double density, double amount,
                 ParticleAmountType type, const Mesh &mesh,
                 std::shared_ptr<Pdf> pdf, std::shared_ptr<Pdf> vdf)
                : pdf(pdf), vdf(vdf) {

    // NB: Deliberate fall-through
    switch(type){
    case ParticleAmountType::per_cell:
        amount *= mesh.mesh->num_cells();
    case ParticleAmountType::in_total:
        amount /= mesh.volume();
    case ParticleAmountType::per_volume:
        amount = density / amount;
    case ParticleAmountType::phys_per_sim:
        break;
    default:
        std::cerr << "Invalid ParticleAmountType";
        exit(1);
    }

    // Amount is now the number of physical particles per simulation particle
    q = charge  * amount;
    m = mass    * amount;
    n = density / amount;

    num = n * mesh.volume();

    std::cout << amount << " " << q << " " << m << " " << n << " " << mesh.volume() << " " << num << std::endl;
}

}
