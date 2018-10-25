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

    /**
     * @brief The charge on the object.
     * Incremented during Population::update(), and corrected by ::update().
     */
    double charge = 0;

    //! Current collected during last Population::update().
    double current = 0;

    //! Latest known potential on the object.
    double potential = 0; 

    //! Boundary facet function id
    size_t bnd_id;

    /**
     * @brief Constructor
     */
    Object(size_t bnd_id) : bnd_id(bnd_id) {};

    /**
     * @brief Apply any associated boundary conditions (if any) to vector.
     * @param[in, out]  b   Vector
     */
    virtual void apply(df::GenericVector &b){};

    /**
     * @brief Apply any associated boundary conditions (if any) to matrix.
     * @param[in, out]  A   Matrix
     */
    virtual void apply(df::GenericMatrix &A){};

    /**
     * @brief Update the object based on the newest potential.
     * @param           phi     Correct potential
     *
     * Some methods do not know the correct charge and potential of the object
     * before after the potential has been solved for. Given the correct
     * a-posteriori potential the charge and potential of the object is
     * corrected.
     */
    virtual void update(const df::Function &phi) = 0;
};

//! A polymorphic vector of Objects.
using ObjectVector = std::vector<std::shared_ptr<Object>>;

class Circuit {
public:
    // ObjectVector &objects;
    // std::vector<Vsource> vsources;
    // std::vector<Isource> isources;
    // std::vector<std::vector<int>> groups;
    // double dt;
    // double eps0;

    /**
     * @brief Check if selected solvers are valid or assign default solvers.
     * @param[in, out]  method          Linear algebra solver.
     * @param[in, out]  preconditioner  Linear algebra preconditioner.
     * @return                          Whether the choice is valid.
     *
     * Not all combinations of methods and preconditioners may work for all
     * object methods. This method returns true for combinations that are
     * known to work. Combinations which (sometimes) fail, or are untested
     * return false. If method and preconditioner are empty strings, the
     * method fills them with the suggested methods and returns true.
     * The methods used are the ones documented in FEniCS.
     */
    virtual bool check_solver_methods(std::string &method,
                                      std::string &preconditioner) const = 0;

    /**
     * @brief Apply any associated boundary conditions (if any) to vector.
     * @param[in, out]  b   Vector
     */
    virtual void apply(df::GenericVector &b){};

    /**
     * @brief Apply any associated boundary conditions (if any) to matrix.
     * @param[in, out]  A   Matrix
     */
    virtual void apply(df::PETScMatrix &A){};
};

} // namespace punc

#endif // OBJECT_H
