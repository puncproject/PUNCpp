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
 * @file		object_CM.h
 * @brief		Mirror charge object method
 *
 * Solves a particular solution of the equation with zero Dirichlet boundary
 * conditions, and then computes the mirror charge this leads to on the
 * objects, and uses this to compute the correct potential. A second solution
 * with the correct Dirichlet boundary conditions yields the correct solution.
 * Same method as described in PTetra paper.
 */

#ifndef OBJECT_CM_H
#define OBJECT_CM_H

#include "object.h"

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/DirichletBC.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace punc {

namespace df = dolfin;

using boost_matrix = boost::numeric::ublas::matrix<double>;

boost_matrix inv_capacitance(const df::FunctionSpace &V,
                             std::vector<std::shared_ptr<Object>> &objects,
                             Mesh &mesh, double eps0 = 1.0);

void reset_objects(std::vector<std::shared_ptr<Object>> &objects);

class ObjectCM : public Object, public df::DirichletBC
{
public:
    ObjectCM(const df::FunctionSpace &V,
             const Mesh &mesh, std::size_t bnd_id);
    void set_potential(double voltage);
    void apply(df::GenericVector &b);
    void apply(df::GenericMatrix &A);
private:
    friend class CircuitCM;
    double image_charge = 0;
};

class CircuitCM : public Circuit
{
public:
    std::vector<std::shared_ptr<ObjectCM>> objects;

    CircuitCM(const df::FunctionSpace &V,
              ObjectVector &object_vector,
              const std::vector<Source> &vsources,
              const std::vector<Source> &isources,
              Mesh &mesh,
              double dt, double eps0 = 1.0);

    void pre_solve();
    void post_solve(const df::Function &phi, Mesh &mesh);
    void apply(df::Function &phi, Mesh &mesh);
    bool check_solver_methods(std::string &method,
                              std::string &preconditioner) const;
private:
    void downcast_objects(const ObjectVector &source);
    void assemble_matrix();
    void assemble_vector();
    void apply_isources_to_object();

    boost_matrix inv_circuit_mat;
    boost_matrix inv_capacitance_mat;
    std::vector<double> circuit_vector;
    std::vector<std::vector<std::size_t>> fixed_voltage;

    std::shared_ptr<df::Form> image_charge;
    
    //! The time-step
    double dt;

    //! The vacuum permittivity
    double eps0;
};

} // namespace punc

#endif // OBJECT_CM_H
