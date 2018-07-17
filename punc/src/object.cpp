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

#include "../include/poisson.h"

namespace punc
{

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

Object::Object(const df::FunctionSpace &V,
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

void Object::get_dofs()
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    size_dofs = dofs.size();
}

void Object::add_charge(const double q)
{
    charge += q;
}

void Object::set_potential(const double voltage)
{
    potential = voltage;
    set_value(std::make_shared<df::Constant>(voltage));
}

void Object::compute_interpolated_charge(const df::Function &q_rho)
{
    interpolated_charge = 0.0;
    for (std::size_t i = 0; i < size_dofs; ++i)
    {
        interpolated_charge += q_rho.vector()->getitem(dofs[i]);
    }
}

void reset_objects(std::vector<Object> &objects)
{
    for (auto& obj: objects)
    {
        obj.set_potential(0.0);
    }
}

void compute_object_potentials(std::vector<Object> &objects,
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

CircuitCM::CircuitCM(std::vector<Object> &objects,
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
                                        std::vector<Object> &objects,
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
                                std::vector<Object> &objects,
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

ConstantBC::ConstantBC(const df::FunctionSpace &V,
               const df::MeshFunction<std::size_t> &bnd,
               std::size_t bnd_id, std::string method):
               df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
               std::make_shared<df::Constant>(0.0),
               std::make_shared<df::MeshFunction<std::size_t>>(bnd),
               bnd_id, method)
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    num_dofs = dofs.size();
}

void ConstantBC::apply(df::GenericVector &b)
{
    auto first_ind = dofs[0];
    auto first_element = b[first_ind];
    df::DirichletBC::apply(b);
    b.setitem(first_ind, first_element);
}

void ConstantBC::apply(df::GenericMatrix &A)
{
    std::vector<std::size_t> neighbors;
    std::vector<double> values;
    std::vector<std::size_t> surface_neighbors;
    std::vector<df::la_index> zero_row;
    std::size_t self_index = 0;
    std::vector<df::la_index> ind;
    std::vector<std::vector<std::size_t> > allneighbors;

    for (std::size_t i = 0; i < num_dofs; i++)
        ind.push_back(dofs[i]);

    std::sort (ind.begin(), ind.end(), [](int i, int j) { return (i<j);} );

    for (std::size_t i = 0; i < ind.size(); i++)
    {
      if (ind[i] == dofs[0])
        continue;
      std::size_t row = ind[i];
      A.getrow(row, neighbors, values);
      allneighbors.push_back(neighbors);
    }
    A.zero(dofs.size()-1, dofs.data()+1);

    std::size_t count = 0;
    for (std::size_t i = 0; i < ind.size(); i++)
    {
      if (ind[i] == dofs[0])
        continue;

      std::size_t row = ind[i];
      surface_neighbors.clear();
      values.clear();
      for (std::size_t j = 0; j < allneighbors[count].size(); j++)
      {
         std::size_t n = allneighbors[count][j];
         if (std::binary_search(ind.begin(), ind.end(), n))
         {
           surface_neighbors.push_back(n);
           values.push_back(-1.0);
         }
      }
      for (std::size_t j = 0; j < surface_neighbors.size(); j++)
      {
        if (surface_neighbors[j] == row)
        {
          self_index = j;
          break;
        }
      }
      std::size_t num_of_neighbors = surface_neighbors.size()-1;
      values[self_index] = num_of_neighbors;
      A.setrow(row, surface_neighbors, values);
      count++;
    }
    A.apply("insert");
}

df::la_index ConstantBC::get_free_row()
{
    auto first_bnd_row = dofs[0];
    return first_bnd_row;
}

double ConstantBC::get_boundary_value(df::Function &phi)
{
    std::vector<double> phi_array(num_dofs, 0.0);
    auto phi_vec = phi.vector();
    phi_vec->get_local(phi_array);
    return phi_array[get_free_row()];
}

ObjectBC::ObjectBC(const df::FunctionSpace &V,
                   const df::MeshFunction<std::size_t> &boundaries,
                   std::size_t bnd_id,
                   double eps0,
                   std::string method)
                   :ConstantBC(V, boundaries, bnd_id, method), id(bnd_id)
{
    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries;
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        if (tags[i] == id)
        {
            bnd.set_value(i, 9999);
        }
    }

    auto eps0_ = std::make_shared<df::Constant>(eps0);
    auto mesh = V.mesh();
    auto dim = mesh->geometry().dim();
    if (dim == 1)
    {
        charge_form = std::make_shared<Charge::Form_0>(mesh);
        /* auto V0 = std::make_shared<Charge::Form_0_FunctionSpace_0>(V.mesh()); */
        /* charge_form = std::make_shared<Charge::Form_0>(V0, eps0_); */
    }
    else if (dim == 2)
    {
        charge_form = std::make_shared<Charge::Form_1>(mesh);
        /* auto V0 = std::make_shared<Charge::Form_1_FunctionSpace_1>(V.mesh()); */
        /* charge_form = std::make_shared<Charge::Form_1>(V0, eps0_); */
    }
    else if (dim == 3)
    {
        charge_form = std::make_shared<Charge::Form_2>(mesh);
        /* auto V0 = std::make_shared<Charge::Form_2_FunctionSpace_2>(V.mesh()); */
        /* charge_form = std::make_shared<Charge::Form_2>(V0, eps0_); */
    }
    charge_form->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(bnd));
    charge_form->set_coefficient("eps0", std::make_shared<df::Constant>(eps0));
}

double ObjectBC::update_charge(df::Function &phi)
{
    // FIXME: get rid of this line
    /* charge_form->set_coefficient("eps0", std::make_shared<df::Constant>((double)8.85e-12)); */
    charge_form->set_coefficient("w0", std::make_shared<df::Function>(phi));
    charge = df::assemble(*charge_form);
    return charge;
}

double ObjectBC::update_potential(df::Function &phi)
{
    potential = get_boundary_value(phi);
    return potential;
}

void get_charge_sharing_set(std::vector<std::vector<int>> &vsources, int node, std::vector<int> &group)
{
    group.emplace_back(node);

    std::size_t i = 0;
    while (i < vsources.size())
    {
        std::vector<int> vsource = vsources[i];
        if (vsource[0] == node)
        {
            vsources.erase(vsources.begin() + i);
            get_charge_sharing_set(vsources, vsource[1], group);
        }
        else if (vsource[1] == node)
        {
            vsources.erase(vsources.begin() + i);
            get_charge_sharing_set(vsources, vsource[0], group);
        }
        else
        {
            i += 1;
        }
    }
}

std::vector<std::vector<int>> get_charge_sharing_sets(std::vector<std::vector<int>> vsources, int num_objects)
{
    std::vector<int> nodes(num_objects);
    std::iota(std::begin(nodes), std::end(nodes), 0);

    std::vector<std::vector<int>> groups;

    while (vsources.size() != 0)
    {
        std::vector<int> group;
        get_charge_sharing_set(vsources, vsources[0][0], group);
        groups.emplace_back(group);
    }
 
    for (std::size_t i = 0; i != groups.size(); i++)
    {
        for (std::size_t j = 0; j != groups[i].size(); j++)
        {
            if (groups[i][j] != -1)
            {
                nodes.erase(std::remove(nodes.begin(), nodes.end(), groups[i][j]), nodes.end());
            }
        }
    }

    groups.erase(std::remove_if(groups.begin(), groups.end(), [](std::vector<int> a) { return std::find(a.begin(), a.end(), -1) != a.end(); }), groups.end());

    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        std::vector<int> node{nodes[i]};
        groups.emplace_back(node);
    }
    return groups;
}

void addrow(df::GenericMatrix& A, df::GenericMatrix& Bc,
            const std::vector<std::size_t> &cols,
            const std::vector<double> &vals,
            std::size_t replace_row, const df::FunctionSpace& V)
{
    std::shared_ptr<df::TensorLayout> layout;
    std::vector<const df::GenericDofMap *> dofmaps;
    for (std::size_t i = 0; i < 2; ++i)
    {
        dofmaps.push_back(V.dofmap().get());
    }

    const df::Mesh &mesh = *(V.mesh());
    layout = Bc.factory().create_layout(mesh.mpi_comm(), 2);
    dolfin_assert(layout);

    std::vector<std::shared_ptr<const df::IndexMap>> index_maps;
    for (std::size_t i = 0; i < 2; i++)
    {
        dolfin_assert(dofmaps[i]);
        index_maps.push_back(dofmaps[i]->index_map());
    }
    layout->init(index_maps, df::TensorLayout::Ghosts::UNGHOSTED);

    df::SparsityPattern &new_sparsity_pattern = *layout->sparsity_pattern();
    new_sparsity_pattern.init(index_maps);

    // With the row-by-row algorithm used here there is no need for
    // inserting non_local rows
    const std::size_t primary_dim = new_sparsity_pattern.primary_dim();
    const std::size_t primary_codim = primary_dim == 0 ? 1 : 0;
    const std::pair<std::size_t, std::size_t> primary_range = index_maps[primary_dim]->local_range();
    const std::size_t secondary_range = index_maps[primary_codim]->size(df::IndexMap::MapSize::GLOBAL);
    const std::size_t diagonal_range = std::min(primary_range.second, secondary_range);
    const std::size_t m = diagonal_range - primary_range.first;

    // Declare some variables used to extract matrix information
    std::vector<std::size_t> columns;
    std::vector<double> values;

    // Hold all values of local matrix
    std::vector<double> allvalues;

    // Hold column id for all values of local matrix
    std::vector<df::la_index> allcolumns;

    // Hold accumulated number of cols on local matrix
    std::vector<df::la_index> offset(m + 1);

    offset[0] = 0;
    std::vector<df::ArrayView<const df::la_index>> dofs(2);
    std::vector<std::vector<df::la_index>> global_dofs(2);

    global_dofs[0].push_back(0);
    // Iterate over rows
    for (std::size_t i = 0; i < (diagonal_range - primary_range.first); i++)
    {
        // Get row and locate nonzeros. Store non-zero values and columns
        // for later
        const std::size_t global_row = i + primary_range.first;
        std::size_t count = 0;
        global_dofs[1].clear();
        columns.clear();
        values.clear();
        if (global_row == replace_row)
        {
            if (df::MPI::rank(mesh.mpi_comm()) == 0)
            {
                columns = cols;
                values = vals;
            }
        }
        else
        {
            A.getrow(global_row, columns, values);
        }
        for (std::size_t j = 0; j < columns.size(); j++)
        {
            // Store if non-zero or diagonal entry.
            if (std::abs(values[j]) > DOLFIN_EPS || columns[j] == global_row)
            {
                global_dofs[1].push_back(columns[j]);
                allvalues.push_back(values[j]);
                allcolumns.push_back(columns[j]);
                count++;
            }
        }
        global_dofs[0][0] = global_row;
        offset[i + 1] = offset[i] + count;
        dofs[0].set(global_dofs[0]);
        dofs[1].set(global_dofs[1]);
        new_sparsity_pattern.insert_global(dofs);
    }

    // Finalize sparsity pattern
    new_sparsity_pattern.apply();

    // Create matrix with the new layout
    Bc.init(*layout);

    // Put the values back into new matrix
    for (std::size_t i = 0; i < m; i++)
    {
        const df::la_index global_row = i + primary_range.first;
        Bc.set(&allvalues[offset[i]], 1, &global_row,
               offset[i + 1] - offset[i], &allcolumns[offset[i]]);
    }
    Bc.apply("insert");
}

Circuit::Circuit(const df::FunctionSpace &V,
                std::vector<ObjectBC> &objects,
                std::vector<std::vector<int>> isources,
                std::vector<double> ivalues,
                std::vector<std::vector<int>> vsources,
                std::vector<double> vvalues,
                double dt, double eps0, std::string method)
                : V(V), objects(objects), isources(isources),
                vsources(vsources), ivalues(ivalues),
                vvalues(vvalues), dt(dt), eps0(eps0)
{
    auto num_objects = objects.size();
    groups = get_charge_sharing_sets(vsources, num_objects);

    std::vector<std::size_t> rows_p(num_objects), rows_c;

    for(std::size_t i = 0; i<groups.size(); ++i)
    {
        rows_c.emplace_back(groups[i][0]);
        rows_charge.emplace_back(objects[rows_c[i]].get_free_row());
    }

    std::iota(std::begin(rows_p), std::end(rows_p), 0);
    auto pred = [&rows_c](std::size_t elem) -> bool {
        return std::find(rows_c.begin(), rows_c.end(), elem) != rows_c.end();
    };
    rows_p.erase(std::remove_if(rows_p.begin(), rows_p.end(), pred), rows_p.end());

    for (std::size_t i = 0; i < rows_p.size(); ++i)
    {
        rows_potential.emplace_back(objects[rows_p[i]].get_free_row());
    }
}

void Circuit::apply(df::GenericVector &b)
{
    apply_isources_to_object();
    apply_vsources_to_vector(b);
}

void Circuit::apply(df::PETScMatrix &A, df::PETScMatrix &Bc)
{
    /* charge_constr->set_coefficient("eps0", ); */
    auto eps0_ = std::make_shared<df::Constant>(eps0);
    auto dim = V.mesh()->geometry().dim();
    if (dim == 1)
    {
        auto V0 = std::make_shared<Constraint::Form_0_FunctionSpace_0>(V.mesh());
        auto V1 = std::make_shared<Constraint::Form_0_FunctionSpace_1>(V.mesh());
        charge_constr = std::make_shared<Constraint::Form_0>(V1, V0, eps0_);
    }
    else if (dim == 2)
    {
        auto V0 = std::make_shared<Constraint::Form_1_FunctionSpace_0>(V.mesh());
        auto V1 = std::make_shared<Constraint::Form_1_FunctionSpace_1>(V.mesh());
        charge_constr = std::make_shared<Constraint::Form_1>(V1, V0, eps0_);
    }
    else if (dim == 3)
    {
        auto V0 = std::make_shared<Constraint::Form_2_FunctionSpace_0>(V.mesh());
        auto V1 = std::make_shared<Constraint::Form_2_FunctionSpace_1>(V.mesh());
        charge_constr = std::make_shared<Constraint::Form_2>(V1,V0, eps0_);
    }

    // Charge constraints
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        for (std::size_t j = 0; j < groups[i].size(); ++j)
        {
            charge_constr->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[groups[i][j]].bnd));
        }
        df::PETScMatrix A0;
        df::assemble(A0, *charge_constr);

        std::vector<std::size_t> cols;
        std::vector<double> vals;
        A0.getrow(0, cols, vals);

        addrow(A, Bc, cols, vals, rows_charge[i], V);
        auto Amat = A.mat();
        PetscErrorCode ierr = MatDuplicate(Bc.mat(), MAT_COPY_VALUES, &Amat);
        if (ierr != 0)
        {
            std::cout << "Error in PETSc MatDuplicate." << '\n';
        }
    }

    // Potential constraints
    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        auto obj_a_id = vsources[i][0];
        auto obj_b_id = vsources[i][1];
        std::vector<std::size_t> cols;
        std::vector<double> vals;

        if (obj_a_id != -1)
        {
            auto dof_a = objects[obj_a_id].get_free_row();
            cols.emplace_back(dof_a);
            vals.emplace_back(-1.0);
        }

        if (obj_b_id != -1)
        {
            auto dof_b = objects[obj_b_id].get_free_row();
            cols.emplace_back(dof_b);
            vals.emplace_back(1.0);
        }
        addrow(A, Bc, cols, vals, rows_potential[i], V);
        auto Amat = A.mat();
        PetscErrorCode ierr = MatDuplicate(Bc.mat(), MAT_COPY_VALUES, &Amat);
        if (ierr != 0)
        {
            std::cout << "Error in PETSc MatDuplicate." << '\n';
        }
    }
}

void Circuit::apply_vsources_to_vector(df::GenericVector &b)
{
    // Charge constaints
    double object_charge;
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        auto group = groups[i];
        object_charge = 0.0;
        for (std::size_t j = 0; j < group.size(); ++j)
        {
            object_charge += objects[group[j]].charge;
        }
        b.setitem(rows_charge[i], object_charge);
    }

    // Potential constaints
    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        b.setitem(rows_potential[i], vvalues[i]);
    }
}

void Circuit::apply_isources_to_object()
{
    for (std::size_t i = 0; i < isources.size(); ++i)
    {
        auto obj_a_id = isources[i][0];
        auto obj_b_id = isources[i][1];
        auto dQ = ivalues[i]*dt;

        if (obj_a_id != -1)
        {
            objects[obj_a_id].charge -= dQ;
        }
        if (obj_b_id != -1)
        {
            objects[obj_b_id].charge += dQ;
        }
    }
}

}
