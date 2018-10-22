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
 * @brief		Objects and circuitry
 *
 * Functions and classes to handle objects and circuitry. 
 */

#ifndef OBJECT_H
#define OBJECT_H

#include <dolfin.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <dolfin/la/PETScMatrix.h>

namespace punc
{

namespace df = dolfin;

typedef boost::numeric::ublas::matrix<double> boost_matrix;
typedef boost::numeric::ublas::vector<double> boost_vector;

bool inv(const boost_matrix &input, boost_matrix &inverse);

class Object : public df::DirichletBC
{
public:
    double potential;
    double charge;
    bool floating;
    std::size_t id;
    df::MeshFunction<std::size_t> bnd;

    double interpolated_charge;
    std::vector<std::size_t> dofs;
    std::size_t size_dofs;

    Object(const df::FunctionSpace &V,
           const df::MeshFunction<std::size_t> &boundaries,
           std::size_t bnd_id,
           double potential = 0.0,
           double charge = 0.0,
           bool floating = true,
           std::string method = "topological");

    void get_dofs();
    void add_charge(const double q);
    void set_potential(const double voltage);
    void compute_interpolated_charge(const df::Function &q_rho);
};

void reset_objects(std::vector<Object> &objcets);

void compute_object_potentials(std::vector<Object> &objects,
                               df::Function &E,
                               const boost_matrix &inv_capacity,
                               std::shared_ptr<const df::Mesh> &mesh);

class CircuitCM
{
public:
    std::vector<Object> &objects;
    const boost_vector &precomputed_charge;
    const boost_matrix &inv_bias;
    double charge;

    CircuitCM(std::vector<Object> &objects,
            const boost_vector &precomputed_charge,
            const boost_matrix &inv_bias,
            double charge = 0.0);

    void circuit_charge();
    void redistribute_charge(const std::vector<double> &tot_charge);
};

void redistribute_circuit_charge(std::vector<CircuitCM> &circuits);

std::vector<std::shared_ptr<df::Function>> solve_laplace(
    std::shared_ptr<df::FunctionSpace> &V,
    std::vector<Object> &objects,
    std::shared_ptr<df::MeshFunction<std::size_t>> &boundaries,
    std::size_t ext_bnd_id);

std::vector<df::Function> solve_laplace(const df::FunctionSpace &V,
                                        std::vector<Object> &objects,
                                        df::MeshFunction<std::size_t> boundaries,
                                        std::size_t ext_bnd_id);

boost_matrix capacitance_matrix(const df::FunctionSpace &V,
                                std::vector<Object> &objects,
                                const df::MeshFunction<std::size_t> &boundaries,
                                std::size_t ext_bnd_id);

boost_matrix bias_matrix(const boost_matrix &inv_capacity,
                         const std::map<int, std::vector<int>> &circuits_info);

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

class ObjectBC: public ConstantBC
{
public:
    double charge = 0.0;
    double current = 0.0;
    double potential = 0.0;
    std::size_t id;
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

class Circuit
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

    Circuit(const df::FunctionSpace &V,
            std::vector<ObjectBC> &objects,
            std::vector<std::vector<int>> isources,
            std::vector<double> ivalues,
            std::vector<std::vector<int>> vsources,
            std::vector<double> vvalues,
            double dt, double eps0 = 1.0,
            std::string method = "topological");

    void apply(df::GenericVector &b);
    void apply(df::PETScMatrix &A, df::PETScMatrix &Bc);
    void apply_vsources_to_vector(df::GenericVector &b);
    void apply_isources_to_object();

    /**
     * @brief  Whether or not this Circuit has charge constraints.
     * @return Whether or not this Circuit has charge constraints.
     */
    bool has_charge_constraints() const;
};


} // namespace punc

#endif // OBJECT_H
