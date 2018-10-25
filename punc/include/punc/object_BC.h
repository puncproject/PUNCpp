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
#include "mesh.h"
#include <dolfin.h>

namespace punc {

namespace df = dolfin;

class ObjectBC: public Object, public df::DirichletBC
{
public:
    ObjectBC(const df::FunctionSpace &V,
             const Mesh &mesh, std::size_t bnd_id, double eps0=1);
    void update(const df::Function &phi);
    void apply(df::GenericVector &b);
    void apply(df::GenericMatrix &A);

private:
    friend class CircuitBC;
    df::MeshFunction<std::size_t> bnd;
    std::shared_ptr<df::Form> charge_form;
    void update_charge(const df::Function &phi);
    void update_potential(const df::Function &phi);
    double old_charge = 0.0;
    std::vector<df::la_index> dofs;
    std::size_t num_dofs;
    df::la_index get_free_row();
    double get_boundary_value(const df::Function &phi);
};

class CircuitBC : public Circuit
{
public:
    const df::FunctionSpace &V;
    std::vector<std::shared_ptr<ObjectBC>> objects;
    std::vector<std::vector<int>> isources, vsources;
    std::vector<double> ivalues, vvalues;
    std::vector<std::size_t> bnd_id;
    double dt;
    double eps0;
    std::vector<std::vector<int>> groups;
    std::vector<std::size_t> rows_charge;
    std::vector<std::size_t> rows_potential;
    std::shared_ptr<df::Form> charge_constr;

    CircuitBC(const df::FunctionSpace &V,
              const ObjectVector &object_source,
              std::vector<std::vector<int>> isources,
              std::vector<double> ivalues,
              std::vector<std::vector<int>> vsources,
              std::vector<double> vvalues,
              double dt, double eps0 = 1.0,
              std::string method = "topological");

    void apply(df::GenericVector &b);
    void apply(df::PETScMatrix &A);
    void apply_vsources_to_vector(df::GenericVector &b);
    void apply_isources_to_object();

    bool check_solver_methods(std::string &method,
                              std::string &preconditioner) const;
private:
    void downcast_objects(const ObjectVector &source);
};

} // namespace punc

#endif // OBJECT_CM_H
