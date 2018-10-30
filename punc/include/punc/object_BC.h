// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++
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
 * @file		object_BC.h
 * @brief		Constant potential boundary condition method
 *
 * Implements objects by altering the linear system of equations to enforce
 * a constant, but unknown potential on all objects. In addition there are
 * voltage constraints (due to voltage sources), and charge constraints.
 * Together close the unknown potentials. Described in PUNC++ paper.
 */

#ifndef OBJECT_BC_H
#define OBJECT_BC_H

#include "object.h"
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/common/types.h>

namespace punc {

namespace df = dolfin;

class ObjectBC: public Object, public df::DirichletBC
{
public:
    ObjectBC(const df::FunctionSpace &V,
             const Mesh &mesh, std::size_t bnd_id, double eps0=1);

    /**
     * @brief Update the object based on the newest potential.
     * @param           phi     Correct potential
     *
     * Some methods do not know the correct charge and potential of the object
     * before after the potential has been solved for. Given the correct
     * a-posteriori potential the charge and potential of the object is
     * corrected.
     */
    void update(const df::Function &phi);

    void apply(df::GenericVector &b);
    void apply(df::GenericMatrix &A);

private:
    friend class CircuitBC;
    df::MeshFunction<std::size_t> bnd;
    std::shared_ptr<df::Form> charge_form;
    std::vector<df::la_index> dofs;
    std::size_t num_dofs;
    df::la_index get_free_row();

};

class CircuitBC : public Circuit
{
public:
    const df::FunctionSpace &V;
    std::vector<std::shared_ptr<ObjectBC>> objects;
    std::vector<std::size_t> bnd_id;

    CircuitBC(const df::FunctionSpace &V,
              const ObjectVector &object_vector,
              const std::vector<Source> &vsources,
              const std::vector<Source> &isources,
              double dt, double eps0 = 1.0);

    void apply(df::GenericVector &b);
    void apply(df::PETScMatrix &A);
    void post_solve(const df::Function &phi, Mesh &mesh);

    bool check_solver_methods(std::string &method,
                              std::string &preconditioner) const;
private:
    void downcast_objects(const ObjectVector &source);
    std::vector<std::size_t> rows_charge;
    std::vector<std::size_t> rows_potential;
    std::shared_ptr<df::Form> charge_constr;
    void apply_vsources_to_vector(df::GenericVector &b);
    void apply_isources_to_object();

    //! The time-step
    double dt;

    //! The vacuum permittivity
    double eps0;
};

} // namespace punc

#endif // OBJECT_CM_H
