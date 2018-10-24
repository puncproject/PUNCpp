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
#include "../ufl/Flux.h"

namespace punc
{

/*******************************************************************************
 * GLOBAL DEFINITIONS
 ******************************************************************************/

bool inv(const boost_matrix &input, boost_matrix &inverse)
{
    typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;
    boost::numeric::ublas::matrix<double> A(input);

    pmatrix pm(A.size1());

    int res = boost::numeric::ublas::lu_factorize(A, pm);
    if (res != 0)
        return false;

    inverse.assign(boost::numeric::ublas::identity_matrix<double>(A.size1()));

    boost::numeric::ublas::lu_substitute(A, pm, inverse);

    return true;
}

ObjectCM::ObjectCM(const df::FunctionSpace &V,
               const df::MeshFunction<std::size_t> &boundaries,
               std::size_t bnd_id,
               double potential,
               double charge,
               bool floating,
               std::string method):
               df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
               std::make_shared<df::Constant>(potential),
               std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
               bnd_id, method), potential(potential),
               charge(charge), floating(floating), id(bnd_id)
{
    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
    	if(tags[i] == id)
    	{
    		bnd.set_value(i, 9999);
    	}
    }
    get_dofs();
}

void ObjectCM::get_dofs()
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    size_dofs = dofs.size();
}

void ObjectCM::add_charge(const double q)
{
    charge += q;
}

void ObjectCM::set_potential(const double voltage)
{
    potential = voltage;
    set_value(std::make_shared<df::Constant>(voltage));
}

void ObjectCM::compute_interpolated_charge(const df::Function &q_rho)
{
    interpolated_charge = 0.0;
    for (std::size_t i = 0; i < size_dofs; ++i)
    {
        interpolated_charge += q_rho.vector()->getitem(dofs[i]);
    }
}

void reset_objects(std::vector<ObjectCM> &objects)
{
    for (auto& obj: objects)
    {
        obj.set_potential(0.0);
    }
}

void compute_object_potentials(std::vector<ObjectCM> &objects,
                               df::Function &E,
                               const boost_matrix &inv_capacity,
                               std::shared_ptr<const df::Mesh> &mesh)
{
    auto dim = mesh->geometry().dim();
    auto num_objects = objects.size();
    std::vector<double> image_charge(num_objects);

    std::shared_ptr<df::Form> flux;
    if (dim == 1)
    {
        flux = std::make_shared<Flux::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        flux = std::make_shared<Flux::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        flux = std::make_shared<Flux::Form_2>(mesh);
    }
    for (std::size_t j = 0; j < num_objects; ++j)
    {
        flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[j].bnd));
        flux->set_coefficient("w0", std::make_shared<df::Function>(E));
        image_charge[j] = df::assemble(*flux);
    }

    double potential;
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        potential = 0.0;
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            potential += (objects[j].charge -\
                          image_charge[j]) * inv_capacity(i, j);
        }
        objects[i].set_potential(potential);
    }
}

CircuitCM::CircuitCM(std::vector<ObjectCM> &objects,
                 const boost_vector &precomputed_charge,
                 const boost_matrix &inv_bias,
                 double charge):
                 objects(objects),
                 precomputed_charge(precomputed_charge),
                 inv_bias(inv_bias), charge(charge) {}

void CircuitCM::circuit_charge()
{
    double c_charge = 0.0;
    for (auto obj: objects)
    {
        c_charge += obj.charge - obj.interpolated_charge;
    }
    this->charge = c_charge;
}

void CircuitCM::redistribute_charge(const std::vector<double> &tot_charge)
{
    std::size_t num_objects = objects.size();
    std::size_t num_rows = inv_bias.size1();
    std::size_t num_cols = inv_bias.size2();
    std::vector<double> redistr_charge(num_rows);
    for (std::size_t i = 0; i < num_rows; ++i)
    {
        redistr_charge[i] = 0.0;
        for (std::size_t j = 0; j < num_cols; ++j)
        {
            redistr_charge[i] += inv_bias(i, j) * tot_charge[j];
        }
    }

    for (std::size_t i = 0; i < num_objects; ++i)
    {
        objects[i].charge = precomputed_charge(i) + redistr_charge[i] +\
                            objects[i].interpolated_charge;
    }
}

void redistribute_circuit_charge(std::vector<CircuitCM> &circuits)
{
    std::size_t num_circuits = circuits.size();
    std::vector<double> tot_charge(num_circuits);
    for (std::size_t i = 0; i < num_circuits; ++i)
    {
        circuits[i].circuit_charge();
        tot_charge[i] = circuits[i].charge;
    }
    for(auto circ: circuits)
    {
        circ.redistribute_charge(tot_charge);
    }
}

std::vector<df::Function> solve_laplace(const df::FunctionSpace &V,
                                        std::vector<ObjectCM> &objects,
                                        df::MeshFunction<std::size_t> boundaries,
                                        std::size_t ext_bnd_id)
{
    auto phi_bnd = std::make_shared<df::Constant>(0.0);
    df::DirichletBC bc(std::make_shared<df::FunctionSpace>(V),
                       phi_bnd,
                       std::make_shared<df::MeshFunction<std::size_t>>(boundaries),
                       ext_bnd_id);
    std::vector<df::DirichletBC> ext_bc = {bc};
    PoissonSolver poisson(V, ext_bc);
    ESolver esolver(V);
    auto num_objects = objects.size();

    std::vector<df::Function> object_e_field;
    auto shared_V = std::make_shared<df::FunctionSpace>(V);
    for (std::size_t i = 0; i < num_objects; ++i)
    {
        for (std::size_t j = 0; j < num_objects; ++j)
        {
            if (i == j)
            {
                objects[j].set_potential(1.0);
            }
            else
            {
                objects[j].set_potential(0.0);
            }
        }
        df::Function rho(shared_V);
        auto phi = poisson.solve(rho, objects);
        object_e_field.emplace_back(esolver.solve(phi));
    }
    return object_e_field;
}

boost_matrix capacitance_matrix(const df::FunctionSpace &V,
                                std::vector<ObjectCM> &objects,
                                const df::MeshFunction<std::size_t> &boundaries,
                                std::size_t ext_bnd_id)
{
    auto mesh = V.mesh();
    auto dim = mesh->geometry().dim();
    auto num_objects = objects.size();
    std::shared_ptr<df::Form> flux;
    if (dim == 1)
    {
        flux = std::make_shared<Flux::Form_0>(mesh);
    }
    else if (dim == 2)
    {
        flux = std::make_shared<Flux::Form_1>(mesh);
    }
    else if (dim == 3)
    {
        flux = std::make_shared<Flux::Form_2>(mesh);
    }
    boost_matrix capacitance(num_objects, num_objects);
    boost_matrix inv_capacity(num_objects, num_objects);
    auto object_e_field = solve_laplace(V, objects, boundaries, ext_bnd_id);
    for (unsigned i = 0; i < num_objects; ++i)
    {
        flux->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[i].bnd));
        for (unsigned j = 0; j < num_objects; ++j)
        {
            flux->set_coefficient("w0", std::make_shared<df::Function>(object_e_field[j]));
            capacitance(i, j) = df::assemble(*flux);
        }
    }
    inv(capacitance, inv_capacity);
    return inv_capacity;
}

boost_matrix bias_matrix(const boost_matrix &inv_capacity,
                         const std::map<int, std::vector<int>> &circuits_info)
{
    std::size_t num_components = inv_capacity.size1();
    std::size_t num_circuits = circuits_info.size();
    boost_matrix bias_matrix(num_components, num_components, 0.0);
    boost_matrix inv_bias(num_components, num_components);

    std::vector<int> circuit;
    int i, s = 0;
    for (auto const &c_map : circuits_info)
    {
        i = c_map.first;
        circuit = c_map.second;
        for (unsigned ii = 0; ii < circuit.size(); ++ii)
        {
            bias_matrix(num_circuits + i, circuit[ii]) = 1.0;
        }
        for (unsigned j = 1; j < circuit.size(); ++j)
        {
            for (unsigned k = 0; k < num_components; ++k)
            {
                bias_matrix(j - 1 + s, k) = inv_capacity(circuit[j], k) -
                                            inv_capacity(circuit[0], k);
            }
        }
        s += circuit.size() - 1;
    }
    inv(bias_matrix, inv_bias);
    return inv_bias;
}

} // namespace punc
