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

void reset_objects(std::vector<std::shared_ptr<Object>> &objects)
{
    for (auto &o : objects)
    {
        o->update(0.0);
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

void ObjectCM::update(double voltage)
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
                     const SourceVector &vsources,
                     const SourceVector &isources,
                     Mesh &mesh,
                     double dt, double eps0)
                    : Circuit(object_vector, vsources, isources),
                      dt(dt), eps0(eps0)
{
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
    bool has_charge_constraints = groups.size() > 0;

    // Defaults
    if (method == "" && preconditioner == "")
    {
        if (has_charge_constraints)
        {
            method = "bicgstab";
            preconditioner = "ilu";
        }
        else
        {
            method = "gmres";
            preconditioner = "hypre_amg";
        }
    }

    if (has_charge_constraints)
    {

        return (method == "bicgstab" && preconditioner == "ilu");
    }
    else
    {

        return (method == "gmres" && preconditioner == "hypre_amg") || (method == "bicgstab" && preconditioner == "ilu");
    }
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
                circuit_matrix(num_objects + group[k], num_objects + group[k]) = -1.0 * inv_capacitance_mat(group[j], group[k]);
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
                circuit_matrix(num_objects + fixed_voltage[i][k], num_objects + fixed_voltage[i][k]) = -1.0 * inv_capacitance_mat(fixed_voltage[i][j], fixed_voltage[i][k]);
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

void CircuitCM::apply(df::Function &phi, Mesh &mesh)
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
        objects[i]->update(potential);
        objects[i]->charge = charge;
    }

    apply_isources_to_object();
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
    PoissonSolver poisson(V, ext_bc, nullptr, eps0);

    auto num_objects = objects.size();

    std::vector<df::Function> phi_vec;
    auto shared_V = std::make_shared<df::FunctionSpace>(V);

    for (std::size_t i = 0; i < num_objects; ++i)
    {
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            if (i == j)
            {
                objects[j]->update(1.0);
            }
            else
            {
                objects[j]->update(0.0);
            }
        }
        df::Function rho(shared_V);
        df::Function phi(shared_V);
        poisson.solve(phi, rho, objects);
        phi_vec.emplace_back(phi);
    }
    return phi_vec;
}

// ObjectCM::ObjectCM(const df::FunctionSpace &V,
//                const df::MeshFunction<std::size_t> &boundaries,
//                std::size_t bnd_id,
//                double potential,
//                double charge,
//                bool floating,
//                std::string method):
//                df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
//                std::make_shared<df::Constant>(potential),
//                std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
//                bnd_id, method), potential(potential),
//                charge(charge), floating(floating), id(bnd_id)
// {
//     auto tags = boundaries.values();
//     auto size = boundaries.size();
//     bnd = boundaries;
//     bnd.set_all(0);
//     for (std::size_t i = 0; i < size; ++i)
//     {
//     	if(tags[i] == id)
//     	{
//     		bnd.set_value(i, 9999);
//     	}
//     }
//     get_dofs();
// }

// void ObjectCM::get_dofs()
// {
//     std::unordered_map<std::size_t, double> dof_map;
//     get_boundary_values(dof_map);

//     for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
//     {
//         dofs.emplace_back(itr->first);
//     }
//     size_dofs = dofs.size();
// }

// void ObjectCM::add_charge(const double q)
// {
//     charge += q;
// }

// void ObjectCM::set_potential(const double voltage)
// {
//     potential = voltage;
//     set_value(std::make_shared<df::Constant>(voltage));
// }

// void ObjectCM::compute_interpolated_charge(const df::Function &q_rho)
// {
//     interpolated_charge = 0.0;
//     for (std::size_t i = 0; i < size_dofs; ++i)
//     {
//         interpolated_charge += q_rho.vector()->getitem(dofs[i]);
//     }
// }

// void reset_objects(std::vector<ObjectCM> &objects)
// {
//     for (auto& obj: objects)
//     {
//         obj.set_potential(0.0);
//     }
// }

// void compute_object_potentials(std::vector<ObjectCM> &objects,
//                                df::Function &E,
//                                const boost_matrix &inv_capacity,
//                                std::shared_ptr<const df::Mesh> &mesh)
// {
//     auto dim = mesh->geometry().dim();
//     auto num_objects = objects.size();
//     std::vector<double> image_charge(num_objects);

//     std::shared_ptr<df::Form> flux;
//     if (dim == 1)
//     {
//         flux = std::make_shared<Flux::Form_0>(mesh);
//     }
//     else if (dim == 2)
//     {
//         flux = std::make_shared<Flux::Form_1>(mesh);
//     }
//     else if (dim == 3)
//     {
//         flux = std::make_shared<Flux::Form_2>(mesh);
//     }
//     for (std::size_t j = 0; j < num_objects; ++j)
//     {
//         flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[j].bnd));
//         flux->set_coefficient("w0", std::make_shared<df::Function>(E));
//         image_charge[j] = df::assemble(*flux);
//     }

//     double potential;
//     for (std::size_t i = 0; i < num_objects; ++i)
//     {
//         potential = 0.0;
//         for (std::size_t j = 0; j < num_objects; ++j)
//         {
//             potential += (objects[j].charge -
//                           image_charge[j]) * inv_capacity(i, j);
//         }
//         objects[i].set_potential(potential);
//     }
// }

// CircuitCM::CircuitCM(std::vector<ObjectCM> &objects,
//                  const boost_vector &precomputed_charge,
//                  const boost_matrix &inv_bias,
//                  double charge):
//                  objects(objects),
//                  precomputed_charge(precomputed_charge),
//                  inv_bias(inv_bias), charge(charge) {}

// void CircuitCM::circuit_charge()
// {
//     double c_charge = 0.0;
//     for (auto obj: objects)
//     {
//         c_charge += obj.charge - obj.interpolated_charge;
//     }
//     this->charge = c_charge;
// }

// void CircuitCM::redistribute_charge(const std::vector<double> &tot_charge)
// {
//     std::size_t num_objects = objects.size();
//     std::size_t num_rows = inv_bias.size1();
//     std::size_t num_cols = inv_bias.size2();
//     std::vector<double> redistr_charge(num_rows);
//     for (std::size_t i = 0; i < num_rows; ++i)
//     {
//         redistr_charge[i] = 0.0;
//         for (std::size_t j = 0; j < num_cols; ++j)
//         {
//             redistr_charge[i] += inv_bias(i, j) * tot_charge[j];
//         }
//     }

//     for (std::size_t i = 0; i < num_objects; ++i)
//     {
//         objects[i].charge = precomputed_charge(i) + redistr_charge[i] +
//                             objects[i].interpolated_charge;
//     }
// }

// void redistribute_circuit_charge(std::vector<CircuitCM> &circuits)
// {
//     std::size_t num_circuits = circuits.size();
//     std::vector<double> tot_charge(num_circuits);
//     for (std::size_t i = 0; i < num_circuits; ++i)
//     {
//         circuits[i].circuit_charge();
//         tot_charge[i] = circuits[i].charge;
//     }
//     for(auto circ: circuits)
//     {
//         circ.redistribute_charge(tot_charge);
//     }
// }

// std::vector<df::Function> solve_laplace(const df::FunctionSpace &V,
//                                         const df::FunctionSpace &W,
//                                         std::vector<ObjectCM> &objects,
//                                         df::MeshFunction<std::size_t> boundaries,
//                                         std::size_t ext_bnd_id)
// {
//     auto phi_bnd = std::make_shared<df::Constant>(0.0);
//     df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V),
//                        phi_bnd,
//                        std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
//                        ext_bnd_id);
//     std::vector<df::DirichletBC> ext_bc = {bc};
//     PoissonSolver poisson(V, ext_bc);
//     ESolver esolver(W);
//     auto num_objects = objects.size();

//     std::vector<df::Function> object_e_field;
//     auto shared_V = std::make_shared<df::FunctionSpace>(V);
//     auto shared_W = std::make_shared<df::FunctionSpace>(W);
//     for (std::size_t i = 0; i < num_objects; ++i)
//     {
//         for (std::size_t j = 0; j < num_objects; ++j)
//         {
//             if (i == j)
//             {
//                 objects[j].set_potential(1.0);
//             }
//             else
//             {
//                 objects[j].set_potential(0.0);
//             }
//         }
//         df::Function rho(shared_V);
//         df::Function phi(shared_V);
//         df::Function E(shared_W);
//         poisson.solve(phi, rho, objects);
//         esolver.solve(E, phi);
//         object_e_field.emplace_back(E);
//     }
//     return object_e_field;
// }

// boost_matrix capacitance_matrix(const df::FunctionSpace &V,
//                                 const df::FunctionSpace &W,
//                                 std::vector<ObjectCM> &objects,
//                                 const df::MeshFunction<std::size_t> &boundaries,
//                                 std::size_t ext_bnd_id)
// {
//     auto mesh = V.mesh();
//     auto dim = mesh->geometry().dim();
//     auto num_objects = objects.size();
//     std::shared_ptr<df::Form> flux;
//     if (dim == 1)
//     {
//         flux = std::make_shared<Flux::Form_0>(mesh);
//     }
//     else if (dim == 2)
//     {
//         flux = std::make_shared<Flux::Form_1>(mesh);
//     }
//     else if (dim == 3)
//     {
//         flux = std::make_shared<Flux::Form_2>(mesh);
//     }
//     boost_matrix capacitance(num_objects, num_objects);
//     boost_matrix inv_capacity(num_objects, num_objects);
//     auto object_e_field = solve_laplace(V, W, objects, boundaries, ext_bnd_id);
//     for (unsigned i = 0; i < num_objects; ++i)
//     {
//         flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[i].bnd));
//         for (unsigned j = 0; j < num_objects; ++j)
//         {
//             flux->set_coefficient("w0", std::make_shared<df::Function>(object_e_field[j]));
//             capacitance(i, j) = df::assemble(*flux);
//         }
//     }
//     inv(capacitance, inv_capacity);
//     return inv_capacity;
// }

// boost_matrix bias_matrix(const boost_matrix &inv_capacity,
//                          const std::map<int, std::vector<int>> &circuits_info)
// {
//     std::size_t num_components = inv_capacity.size1();
//     std::size_t num_circuits = circuits_info.size();
//     boost_matrix bias_matrix(num_components, num_components, 0.0);
//     boost_matrix inv_bias(num_components, num_components);

//     std::vector<int> circuit;
//     int i, s = 0;
//     for (auto const &c_map : circuits_info)
//     {
//         i = c_map.first;
//         circuit = c_map.second;
//         for (unsigned ii = 0; ii < circuit.size(); ++ii)
//         {
//             bias_matrix(num_circuits + i, circuit[ii]) = 1.0;
//         }
//         for (unsigned j = 1; j < circuit.size(); ++j)
//         {
//             for (unsigned k = 0; k < num_components; ++k)
//             {
//                 bias_matrix(j - 1 + s, k) = inv_capacity(circuit[j], k) -
//                                             inv_capacity(circuit[0], k);
//             }
//         }
//         s += circuit.size() - 1;
//     }
//     inv(bias_matrix, inv_bias);
//     return inv_bias;
// }

} // namespace punc
