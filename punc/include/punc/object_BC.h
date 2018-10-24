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
#include <dolfin.h>

namespace punc {

namespace df = dolfin;

class ConstantBC : public df::DirichletBC
{
public:
    std::vector<df::la_index> dofs;
    std::size_t num_dofs;
    ConstantBC(const df::FunctionSpace &V,
            const df::MeshFunction<std::size_t> &bnd,
            std::size_t bnd_id,
            std::string method = "topological");
    void apply(df::GenericVector &b);
    void apply(df::GenericMatrix &A);
    df::la_index get_free_row();
    double get_boundary_value(df::Function &phi);
};

class ObjectBC: public ConstantBC, public Object
{
public:
    df::MeshFunction<std::size_t> bnd;
    std::shared_ptr<df::Form> charge_form;
    ObjectBC(const df::FunctionSpace &V,
             const df::MeshFunction<std::size_t> &boundaries,
             std::size_t bnd_id,
             double eps0=1,
             std::string method = "topological");
    void update_charge(df::Function &phi);
    void update_potential(df::Function &phi);
    void update_current(double dt);

private:
    double old_charge = 0.0;
};

class CircuitBC : public Circuit
{
public:
    const df::FunctionSpace &V;
    std::vector<ObjectBC> &objects;
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
              std::vector<ObjectBC> &objects,
              std::vector<std::vector<int>> isources,
              std::vector<double> ivalues,
              std::vector<std::vector<int>> vsources,
              std::vector<double> vvalues,
              double dt, double eps0 = 1.0,
              std::string method = "topological");

    void apply(df::GenericVector &b);
    void apply(df::PETScMatrix &A);
    void apply_old(df::PETScMatrix &A, df::PETScMatrix &A_tmp);
    void apply_vsources_to_vector(df::GenericVector &b);
    void apply_isources_to_object();

    bool check_solver_methods(std::string &method,
                              std::string &preconditioner) const;
};

} // namespace punc

#endif // OBJECT_CM_H
