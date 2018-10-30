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

#include "../include/punc/object_BC.h"
#include "../ufl/Charge.h"
#include "../ufl/Constraint.h"
#include <dolfin/function/Constant.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/common/MPI.h>

namespace punc {

/*******************************************************************************
 * LOCAL DECLARATIONS
 ******************************************************************************/

/**
 * @brief Replace a row in a matrix
 * @param       A       Input matrix
 * @param[out]  Bc      Output matrix
 * @param       row     Which row to place values in
 * @param       cols    Which columns to place values in
 * @param       vals    Values, one for each column in cols
 * @param       V       Function Space
 *
 * Since the sparsity pattern in a PETSc matrix cannot be changed after it is
 * made, one has to make an entirely new matrix when violating the original
 * sparsity pattern. cols is the new sparsity pattern for the row to be
 * replaced. Columns not in cols will be zero.
 */
static void addrow(const df::GenericMatrix& A, df::GenericMatrix& Bc,
                   std::size_t row,
                   const std::vector<std::size_t> &cols,
                   const std::vector<double> &vals,
                   const df::FunctionSpace& V);

/*******************************************************************************
 * GLOBAL DEFINITIONS
 ******************************************************************************/

void ObjectBC::apply(df::GenericVector &b)
{
    auto first_ind = dofs[0];
    auto first_element = b[first_ind];
    df::DirichletBC::apply(b);
    b.setitem(first_ind, first_element);
}

void ObjectBC::apply(df::GenericMatrix &A)
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

df::la_index ObjectBC::get_free_row()
{
    auto first_bnd_row = dofs[0];
    return first_bnd_row;
}

ObjectBC::ObjectBC(const df::FunctionSpace &V,
                   const Mesh &mesh, std::size_t bnd_id, double eps0)
                   :Object(bnd_id),
                   df::DirichletBC(std::make_shared<df::FunctionSpace>(V),
                   std::make_shared<df::Constant>(0.0),
                   std::make_shared<df::MeshFunction<std::size_t>>(mesh.bnd),
                   bnd_id)
{

    const df::MeshFunction<std::size_t> &boundaries = mesh.bnd;

    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);

    for (auto itr = dof_map.begin(); itr != dof_map.end(); ++itr)
    {
        dofs.emplace_back(itr->first);
    }
    num_dofs = dofs.size();

    auto tags = boundaries.values();
    auto size = boundaries.size();
    bnd = boundaries; // Important: This is a copy.
    bnd.set_all(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        if (tags[i] == bnd_id)
        {
            bnd.set_value(i, 9999);
        }
    }

    auto eps0_ = std::make_shared<df::Constant>(eps0);

    if (mesh.dim == 1)
    {
        charge_form = std::make_shared<Charge::Form_0>(mesh.mesh);
        charge_form->set_coefficient("w0", eps0_);
    }
    else if (mesh.dim == 2)
    {
        charge_form = std::make_shared<Charge::Form_1>(mesh.mesh);
        charge_form->set_coefficient("w0", eps0_);
    }
    else if (mesh.dim == 3)
    {
        charge_form = std::make_shared<Charge::Form_2>(mesh.mesh);
        charge_form->set_coefficient("eps0", eps0_);
    }
    charge_form->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(bnd));
}

void ObjectBC::update(const df::Function &phi)
{
    // update charge
    charge_form->set_coefficient("w1", std::make_shared<df::Function>(phi));
    charge = df::assemble(*charge_form);

    // update potential
    std::vector<double> phi_array(num_dofs, 0.0);
    auto phi_vec = phi.vector();
    phi_vec->get_local(phi_array);
    potential = phi_array[get_free_row()];
}

CircuitBC::CircuitBC(const df::FunctionSpace &V,
                     const ObjectVector &object_vector,
                     const SourceVector &vsources,
                     const SourceVector &isources,
                     double dt, double eps0)
                     : Circuit(object_vector, vsources, isources),
                       V(V), dt(dt), eps0(eps0)
{
    downcast_objects(object_vector);

    std::vector<std::size_t> rows_p(num_objects), rows_c;

    for(std::size_t i = 0; i<groups.size(); ++i)
    {
        rows_c.emplace_back(groups[i][0]);
        rows_charge.emplace_back(objects[rows_c[i]]->get_free_row());
    }

    std::iota(std::begin(rows_p), std::end(rows_p), 0);
    auto pred = [&rows_c](std::size_t elem) -> bool {
        return std::find(rows_c.begin(), rows_c.end(), elem) != rows_c.end();
    };
    rows_p.erase(std::remove_if(rows_p.begin(), rows_p.end(), pred), rows_p.end());

    for (std::size_t i = 0; i < rows_p.size(); ++i)
    {
        rows_potential.emplace_back(objects[rows_p[i]]->get_free_row());
    }
}

void CircuitBC::downcast_objects(const ObjectVector &source){
    // I am not happy about this down-casting.
    for(auto &o : source){
        objects.push_back(std::dynamic_pointer_cast<ObjectBC>(o));
    }
}

void CircuitBC::post_solve(const df::Function &phi, Mesh &mesh){
    for(auto &o : objects){
        o->update(phi);
    }
}

bool CircuitBC::check_solver_methods(std::string &method,
                                     std::string &preconditioner) const
{
    bool has_charge_constraints = groups.size()>0;

    // Defaults
    if(method=="" && preconditioner==""){
        if(has_charge_constraints){
            method = "bicgstab";
            preconditioner = "ilu";
        } else {
            method = "gmres";
            preconditioner = "hypre_amg";
        }
    }

    if(has_charge_constraints){

        return (method=="bicgstab" && preconditioner=="ilu");

    } else {

        return (method=="gmres" && preconditioner=="hypre_amg")
            || (method=="bicgstab" && preconditioner=="ilu");
    }
}

void CircuitBC::pre_solve()
{
    apply_isources_to_object();
}

void CircuitBC::apply(df::GenericVector &b)
{
    apply_vsources_to_vector(b);
}

void CircuitBC::apply(df::PETScMatrix &A)
{
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
        charge_constr = std::make_shared<Constraint::Form_2>(V1, V0, eps0_);
    }

    df::PETScMatrix A_tmp;

    // Charge constraints
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        for (std::size_t j = 0; j < groups[i].size(); ++j)
        {
            charge_constr->set_exterior_facet_domains(std::make_shared<df::MeshFunction<std::size_t>>(objects[groups[i][j]]->bnd));
        }
        df::PETScMatrix A_constraint; // Only one row
        df::assemble(A_constraint, *charge_constr);

        std::vector<std::size_t> cols;
        std::vector<double> vals;
        A_constraint.getrow(0, cols, vals);

        addrow(A, A_tmp, rows_charge[i], cols, vals, V);
        auto Amat = A.mat();
        PetscErrorCode ierr = MatDuplicate(A_tmp.mat(), MAT_COPY_VALUES, &Amat);
        if (ierr != 0)
        {
            std::cout << "Error in PETSc MatDuplicate." << '\n';
        }
    }

    // Potential constraints
    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        auto obj_a_id = vsources[i].node_a;
        auto obj_b_id = vsources[i].node_b;
        std::vector<std::size_t> cols;
        std::vector<double> vals;

        if (obj_a_id != -1)
        {
            auto dof_a = objects[obj_a_id]->get_free_row();
            cols.emplace_back(dof_a);
            vals.emplace_back(-1.0);
        }

        if (obj_b_id != -1)
        {
            auto dof_b = objects[obj_b_id]->get_free_row();
            cols.emplace_back(dof_b);
            vals.emplace_back(1.0);
        }
        addrow(A, A_tmp, rows_potential[i], cols, vals, V);
        auto Amat = A.mat();
        PetscErrorCode ierr = MatDuplicate(A_tmp.mat(), MAT_COPY_VALUES, &Amat);
        if (ierr != 0)
        {
            std::cout << "Error in PETSc MatDuplicate." << '\n';
        }
    }

    PetscErrorCode ierr = MatCopy(A_tmp.mat(), A.mat(), DIFFERENT_NONZERO_PATTERN);
    if (ierr != 0)
    {
        std::cout << "Error in PETSc MatCopy." << '\n';
    }
}

void CircuitBC::apply_vsources_to_vector(df::GenericVector &b)
{
    // Charge constaints
    double object_charge;
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        auto group = groups[i];
        object_charge = 0.0;
        for (std::size_t j = 0; j < group.size(); ++j)
        {
            object_charge += objects[group[j]]->charge;
        }
        b.setitem(rows_charge[i], object_charge);
    }

    // Potential constaints
    for (std::size_t i = 0; i < vsources.size(); ++i)
    {
        b.setitem(rows_potential[i], vsources[i].value);
    }
}

// Should be moved to Circuit base class, but requires some structural changes
void CircuitBC::apply_isources_to_object()
{
    for (std::size_t i = 0; i < isources.size(); ++i)
    {
        auto obj_a_id = isources[i].node_a;
        auto obj_b_id = isources[i].node_b;
        auto dQ = isources[i].value*dt;

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

static void addrow(const df::GenericMatrix& A, df::GenericMatrix& Bc,
                   std::size_t row,
                   const std::vector<std::size_t> &cols,
                   const std::vector<double> &vals,
                   const df::FunctionSpace& V)
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
        if (global_row == row)
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

} // namespace punc
