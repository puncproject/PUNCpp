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

#ifndef DISTRIBUTOR_H
#define DISTRIBUTOR_H

#include "population.h"

namespace punc
{

namespace df = dolfin;

std::vector<double> element_volume(const df::FunctionSpace &V, bool voronoi = false);

df::Function distribute(const df::FunctionSpace &V,
                        Population &pop,
                        const std::vector<double> &dv_inv);

df::Function distribute_new(const df::FunctionSpace &V,
                        Population &pop,
                        const std::vector<double> &dv_inv);

df::Function distribute_dg0_new(const df::FunctionSpace &Q, Population &pop);

df::Function distribute_dg0(const df::FunctionSpace &Q, Population &pop);
}

#endif
