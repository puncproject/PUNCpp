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
 * @file		object.h
 * @brief		Interface for object methods
 *
 * Common interface for objects and circuits for all object methods.
 */

#ifndef OBJECT_H
#define OBJECT_H

#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>

namespace punc {

namespace df = dolfin;

class Source {
public:
    using size_t = std::size_t;

    size_t node_a;
    size_t node_b;
    double value;
};
class Vsource : public Source {};
class Isource : public Source {};

class Object {
public:
    using size_t = std::size_t;

    double charge;
    double current;
    double potential;
    size_t bnd_id;

    Object(size_t bnd_id) : bnd_id(bnd_id) {};
    virtual void apply(df::GenericVector &b){};
    virtual void apply(df::GenericMatrix &A){};
    virtual void update(const df::Function &phi) = 0;
};
using ObjectVector = std::vector<std::shared_ptr<Object>>;

class Circuit {
public:
    // ObjectVector &objects;
    // std::vector<Vsource> vsources;
    // std::vector<Isource> isources;
    // std::vector<std::vector<int>> groups;
    // double dt;
    // double eps0;
    virtual bool check_solver_methods(std::string &method,
                                      std::string &preconditioner) const = 0;

    virtual void apply(df::GenericVector &b){};
    virtual void apply(df::PETScMatrix &A){};
};

} // namespace punc

#endif // OBJECT_H
