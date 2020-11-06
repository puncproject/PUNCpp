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

#include "mesh.h"

#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/PETScMatrix.h>

namespace punc {

namespace df = dolfin;

/**
 * @brief Voltage/current source
 * 
 * Technically a two-port: a component with a value connected between two nodes.
 * This is used by Circuit as a description of voltage and current sources.
 */
class Source {
public:
    /**
     * @brief Constructor
     * @param   node_a  Ascending object index of object connected to minus
     * @param   node_a  Ascending object index of object connected to plus
     * @param   value   Value of the source
     **/
    Source(int node_a, int node_b, double value) :
        node_a(node_a), node_b(node_b), value(value) {};

    int node_a;     ///< Ascending object index of object connected to minus
    int node_b;     ///< Ascending object index of object connected to plus
    double value;   ///< Value of the source
};

class VSource : public Source {
public:
    VSource(int node_a, int node_b, double value) :
        Source(node_a, node_b, value) {};
};

class ISource : public Source {
public:
    ISource(int node_a, int node_b, double value) :
        Source(node_a, node_b, value) {};
};

/**
 * @brief Object interface
 * 
 * This represents the interface for any implementation of objects. The
 * implemented methods must inherit this interface and implement its methods
 * as appropriate.
 */
class Object {
public:
    using size_t = std::size_t;

    /**
     * @brief Constructor
     * @param bnd_id    The id of the boundary
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
     * @brief Set object potential
     * @param   p   new potential
     */
    virtual void set_potential(double p){potential=p;};

    /**
     * @brief Get object potential
     * @return   object potential
     */
    double get_potential(){return potential;};

    /**
     * @brief The charge on the object.
     * Incremented during Population::update(), and corrected by ::update().
     */
    double charge = 0;

    //! Current collected during last Population::update().
    double current = 0;

    //! Boundary facet function id
    size_t bnd_id;


protected:
    //! Latest known potential on the object.
    double potential = 0; 
};

//! A polymorphic vector of Objects.
using ObjectVector = std::vector<std::shared_ptr<Object>>;

//! A vector of voltage sources
using VSourceVector = std::vector<VSource>;

//! A vector of current sources
using ISourceVector = std::vector<ISource>;

/**
 * @brief Circuit interface
 * 
 * This represents the interface for any implementation of circuits. The
 * implemented methods must inherit this interface and implement its methods
 * as appropriate.
 */
class Circuit {
public:

    /**
     * @brief Constructor
     * @param   object_vector   objects
     * @param   vsources        voltage sources
     * @param   isources        current sources
     */
    Circuit(const ObjectVector &object_vector,
            const VSourceVector &vsources,
            const ISourceVector &isources);

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

    /**
     * @brief pre-solver preparations
     * @see post_solve, correction_required
     *
     * This method is run prior to solving the Poisson equation to perform any
     * preparations necessary, such as resetting the objects.
     */
    virtual void pre_solve() = 0;

    /**
     * @brief post-solver updates
     * @see pre_solve, correction_required
     *
     * This method is run after solving the Poisson equation to update and
     * perform corrections to the objects. Some methods will require a
     * correction to the electric potential after this, as indicated by
     * correction_required. The correction is applied by solving the Poisson
     * equation again.
     */
    virtual void post_solve(const df::Function &phi, Mesh &mesh) = 0;

    //! Number of objects
    std::size_t num_objects;

    //! Whether correcting the electric potential is necessary
    bool correction_required = false;

    //! A reference to the objects connected by the circuitry
    // ObjectVector &objects;
    // Stored in children instead, as derived type

    //! A reference to the voltage sources in the circuit
    const VSourceVector &vsources;

    //! A reference to the current sources in the circuit
    const ISourceVector &isources;

    /*
     * Note: I am not quite sure how good an idea it is to have all these
     * references inside this class. The idea is that e.g. an object can
     * be stored, and changed outside the circuit, for instance when it is
     * updated by Population::update(), and still be part of the circuit.
     * If we notice any surprises we might have to rethink this.
     */

protected:
    //! Groups of object indices of charge-sharing objects.
    std::vector<std::vector<int>> groups;
};

/**
 * @name Printing functions
 *
 * Overloading "put-to" operators for convenient printing.
 */
///@{
std::ostream& operator<<(std::ostream& out, const VSource &s);
std::ostream& operator<<(std::ostream& out, const ISource &s);
///@}

} // namespace punc

#endif // OBJECT_H
