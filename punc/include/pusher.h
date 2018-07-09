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

#ifndef PUSHER_H
#define PUSHER_H

#include <dolfin.h>
#include "population.h"

namespace punc
{

namespace df = dolfin;

double accel(Population &pop, const df::Function &E, const double dt);

double boris(Population &pop, const df::Function &E, 
             const std::vector<double> &B, const double dt);

double boris_nonuniform(Population &pop, const df::Function &E,
                        const df::Function &B, const double dt);

void move_periodic(Population &pop, const double dt, const std::vector<double> &Ld);

void move(Population &pop, const double dt);

std::vector<double> cross(std::vector<double> &v1, std::vector<double> &v2);

}

#endif
