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

#include "../include/punc/object_CM.h"
#include "../include/punc/poisson.h"
#include "../ufl/Charge.h"
#include <dolfin/function/Constant.h>
#include <dolfin/fem/assemble.h>

namespace punc {

/*******************************************************************************
 * GLOBAL DEFINITIONS
 ******************************************************************************/

bool inv(const boost_matrix &mat, boost_matrix &inv_mat);


std::vector<df::Function> laplace_solver(const df::FunctionSpace &V,
                                         std::vector<std::shared_ptr<Object>> &objects,
                                         Mesh &mesh, double eps0);


/*******************************************************************************
 * GLOBAL DEFINITIONS
 ******************************************************************************/

void CircuitCM::pre_solve()
{
    apply_isources_to_object();
    for (auto &o : objects)
    {
        o->set_potential(0.0);
    }
}

boost_matrix inv_capacitance(const df::FunctionSpace &V,
                             std::vector<std::shared_ptr<Object>> &objects,
                             Mesh &mesh,
                             double eps0)
{
    auto dim = mesh.dim;
    auto num_objects = objects.size();

    std::shared_ptr<df::Form> charge;
    auto _eps0 = std::make_shared<df::Constant>(eps0);
    if (dim == 1)
    {
        charge = std::make_shared<Charge::Form_0>(mesh.mesh);
        charge->set_coefficient("w0", _eps0);
    }
    else if (dim == 2)
    {
        charge = std::make_shared<Charge::Form_1>(mesh.mesh);
        charge->set_coefficient("w0", _eps0);
    }
    else if (dim == 3)
    {
        charge = std::make_shared<Charge::Form_2>(mesh.mesh);
        charge->set_coefficient("eps0", _eps0);
    }

    boost_matrix capacitance(num_objects, num_objects);
    boost_matrix inv_capacity(num_objects, num_objects);

    auto phi_vec = laplace_solver(V, objects, mesh, eps0);
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        relabel_mesh_function(mesh.bnd, i + 2, 9999);
        charge->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(mesh.bnd));
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            charge->set_coefficient("w1", std::make_shared<df::Function>(phi_vec[j]));
            capacitance(i, j) = df::assemble(*charge);
        }
        relabel_mesh_function(mesh.bnd, 9999, i + 2);
    }

    inv(capacitance, inv_capacity);
    return inv_capacity;
}

ObjectCM::ObjectCM(const df::FunctionSpace &V,
                   const Mesh &mesh, std::size_t bnd_id)
                   :Object(bnd_id),
                    df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
                    std::make_shared<df::Constant>(0.0),
                    std::make_shared<df::MeshFunction<std::size_t>>(mesh.bnd),
                    bnd_id)
{
    // Do nothing
}

void ObjectCM::set_potential(double voltage)
{
    potential = voltage;
    set_value(std::make_shared<df::Constant>(voltage));
}

void ObjectCM::apply(df::GenericVector &b)
{
    df::DirichletBC::apply(b);
}

void ObjectCM::apply(df::GenericMatrix &A)
{
    df::DirichletBC::apply(A);
}

CircuitCM::CircuitCM(const df::FunctionSpace &V, 
                     ObjectVector &object_vector,
                     const VSourceVector &vsources,
                     const ISourceVector &isources,
                     Mesh &mesh,
                     double dt, double eps0)
                    : Circuit(object_vector, vsources, isources),
                      dt(dt), eps0(eps0)
{
    correction_required = true;

    inv_circuit_mat.resize(2 * object_vector.size(), 2 * object_vector.size(), 0.0);
    circuit_vector.resize(2 * object_vector.size());

    inv_capacitance_mat = inv_capacitance(V, object_vector, mesh);

    downcast_objects(object_vector);

    int node;
    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        if (vsources[i].node_a == -1)
        {
            std::vector<std::size_t> tmp;
            tmp.emplace_back(vsources[i].node_b);
            node = vsources[i].node_b;
            for (std::size_t j = 0; j < vsources.size(); ++j)
            {
                if (vsources[j].node_a == node)
                {
                    tmp.emplace_back(vsources[j].node_b);
                    node = vsources[j].node_b;
                }
            }
            fixed_voltage.emplace_back(tmp);
        }
    }

    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        circuit_vector[vsources[i].node_b] = vsources[i].value;
    }

    assemble_matrix();

    auto _eps0 = std::make_shared<df::Constant>(eps0);
    auto dim = mesh.dim;
    if (dim == 1)
    {
        image_charge = std::make_shared<Charge::Form_0>(mesh.mesh);
        image_charge->set_coefficient("w0", _eps0);
    }
    else if (dim == 2)
    {
        image_charge = std::make_shared<Charge::Form_1>(mesh.mesh);
        image_charge->set_coefficient("w0", _eps0);
    }
    else if (dim == 3)
    {
        image_charge = std::make_shared<Charge::Form_2>(mesh.mesh);
        image_charge->set_coefficient("eps0", _eps0);
    }
}

void CircuitCM::downcast_objects(const ObjectVector &source)
{
    // I am not happy about this down-casting.
    for (auto &o : source)
    {
        objects.push_back(std::dynamic_pointer_cast<ObjectCM>(o));
    }
}

bool CircuitCM::check_solver_methods(std::string &method,
                                     std::string &preconditioner) const
{
    // Defaults
    if (method == "" && preconditioner == "")
    {
        method = "gmres";
        preconditioner = "hypre_amg";
    }

    return (method == "gmres" && preconditioner == "hypre_amg") || (method == "bicgstab" && preconditioner == "ilu");
}

void CircuitCM::assemble_matrix()
{
    boost_matrix circuit_matrix(2*num_objects, 2*num_objects, 0.0);

    // Potential constaints
    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        circuit_matrix(vsources[i].node_b, vsources[i].node_b) = 1.0;
        if (vsources[i].node_a != -1){
            circuit_matrix(vsources[i].node_b, vsources[i].node_a) = -1.0;
        }
    }

    // Charge constraints and image charges
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        auto group = groups[i];
        for (std::size_t j = 0; j < group.size(); ++j)
        {
            circuit_matrix(num_objects + groups[i][0], num_objects + group[j]) = 1.0;
        }

        circuit_matrix(groups[i][0], groups[i][0]) = 1.0;
        for (std::size_t j = 0; j < group.size(); ++j)
        {
            circuit_matrix(groups[i][0], num_objects + group[j]) = -1.0 * inv_capacitance_mat(groups[i][0], group[j]);
        }

        for (std::size_t j = 1; j < group.size(); ++j)
        {
            circuit_matrix(num_objects + group[j], group[j]) = 1.0;
            for (std::size_t k = 0; k < group.size(); ++k)
            {
                circuit_matrix(num_objects + group[j], num_objects + group[k]) = -1.0 * inv_capacitance_mat(group[j], group[k]);
            }
        }
    }

    // Objects with fixed potential
    for (std::size_t i = 0; i < fixed_voltage.size(); ++i)
    {
        for (std::size_t j = 0; j < fixed_voltage[i].size(); ++j)
        {
            circuit_matrix(num_objects + fixed_voltage[i][j], fixed_voltage[i][j]) = 1.0;
            for (std::size_t k = 0; k < fixed_voltage[i].size(); ++k)
            {
                circuit_matrix(num_objects + fixed_voltage[i][j], num_objects + fixed_voltage[i][k]) = -1.0 * inv_capacitance_mat(fixed_voltage[i][j], fixed_voltage[i][k]);
            }
        }
    }

    inv(circuit_matrix, inv_circuit_mat);
}

void CircuitCM::assemble_vector()
{
    double collected_charge;
    // Collected charge
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        auto group = groups[i];
        collected_charge = 0.0;
        for (std::size_t j = 0; j < group.size(); ++j)
        {
            collected_charge += objects[group[j]]->charge;
        }
        circuit_vector[num_objects + groups[i][0]] = collected_charge;
    }

    // Image charge
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        auto group = groups[i];
        collected_charge = 0.0;
        for (std::size_t j = 0; j < group.size(); ++j)
        {
            collected_charge += inv_capacitance_mat(groups[i][0], group[j])*objects[group[j]]->image_charge;
        }
        circuit_vector[groups[i][0]] = collected_charge;
 
        
        for (std::size_t j = 1; j < group.size(); ++j)
        {
            collected_charge = 0.0;
            for (std::size_t k = 0; k < group.size(); ++k)
            {
                collected_charge += inv_capacitance_mat(group[j], group[k]) * objects[group[k]]->image_charge;
            }
            circuit_vector[num_objects + group[j]] = collected_charge;
        }
    }

    // Objects with fixed potential
    for (std::size_t i = 0; i < fixed_voltage.size(); ++i)
    {
        for (std::size_t j = 0; j < fixed_voltage[i].size(); ++j)
        {
            collected_charge = 0.0;
            for (std::size_t k = 0; k < fixed_voltage[i].size(); ++k)
            {
                collected_charge += inv_capacitance_mat(fixed_voltage[i][j], fixed_voltage[i][k]) * objects[fixed_voltage[i][k]]->image_charge;
            }
            circuit_vector[num_objects + fixed_voltage[i][j]] = collected_charge;
        }
    }
}

void CircuitCM::post_solve(const df::Function &phi, Mesh &mesh)
{
    for (std::size_t j = 0; j < num_objects; ++j)
    {
        relabel_mesh_function(mesh.bnd, j + 2, 9999);
        image_charge->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(mesh.bnd));
        image_charge->set_coefficient("w1", std::make_shared<df::Function>(phi));
        objects[j]->image_charge = df::assemble(*image_charge);
        relabel_mesh_function(mesh.bnd, 9999, j + 2);
    }

    assemble_vector();

    double potential, charge;
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        potential = 0.0;
        charge = 0.0;
        for (std::size_t j = 0; j < 2*num_objects; ++j)
        {
            potential += inv_circuit_mat(i,j) * circuit_vector[j];
        }
        for (std::size_t j = 0; j < 2 * num_objects; ++j)
        {
            charge += inv_circuit_mat(num_objects + i, j) * circuit_vector[j];
        }
        objects[i]->set_potential(potential);
        objects[i]->charge = charge;
    }
}

void CircuitCM::apply_isources_to_object()
{
    for (std::size_t i = 0; i < isources.size(); ++i)
    {
        auto obj_a_id = isources[i].node_a;
        auto obj_b_id = isources[i].node_b;
        auto dQ = isources[i].value * dt;

        if (obj_a_id != -1)
        {
            objects[obj_a_id]->charge -= dQ;
        }
        if (obj_b_id != -1)
        {
            objects[obj_b_id]->charge += dQ;
        }
    }
}

/*******************************************************************************
 * LOCAL DEFINITIONS
 ******************************************************************************/

bool inv(const boost_matrix &mat, boost_matrix &inv_mat)
{
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;
    
    boost_matrix A(mat);
    pmatrix pm(A.size1());

    int res = boost::numeric::ublas::lu_factorize(A, pm);
    
    if (res != 0){
        return false;
    }
 
    inv_mat.assign(boost::numeric::ublas::identity_matrix<double>(A.size1()));
    boost::numeric::ublas::lu_substitute(A, pm, inv_mat);

    return true;
}

std::vector<df::Function> laplace_solver(const df::FunctionSpace &V,
                                         std::vector<std::shared_ptr<Object>> &objects,
                                         Mesh &mesh, double eps0)
{
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V),
                       std::make_shared<df::Constant>(0.0),
                       std::make_shared<df::MeshFunction<size_t>>(mesh.bnd),
                       mesh.ext_bnd_id);

    std::vector<df::DirichletBC> ext_bc = {bc};
    PoissonSolver poisson(V, objects, ext_bc, nullptr, eps0);

    auto num_objects = objects.size();

    std::vector<df::Function> phi_vec;
    auto shared_V = std::make_shared<df::FunctionSpace>(V);

    for (std::size_t i = 0; i < num_objects; ++i)
    {
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            if (i == j)
            {
                objects[j]->set_potential(1.0);
            }
            else
            {
                objects[j]->set_potential(0.0);
            }
        }
        df::Function rho(shared_V);
        df::Function phi(shared_V);
        poisson.solve(phi, rho, objects);
        phi_vec.emplace_back(phi);
    }
    return phi_vec;
}

} // namespace punc
