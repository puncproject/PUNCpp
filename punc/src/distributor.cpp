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

#include "../include/punc/distributor.h"
#include "../ufl/WeightedVolume.h"

namespace punc
{

std::vector<double> element_volume(const df::FunctionSpace &V, bool voronoi)
{
    auto num_dofs = V.dim();
    auto dof_indices = df::vertex_to_dof_map(V);
    std::vector<double> volumes(num_dofs, 0.0);

    auto mesh = V.mesh();
    auto tdim = mesh->topology().dim();
    auto gdim = mesh->geometry().dim();
    int j = 0;
    mesh->init(0, tdim);
    for (df::MeshEntityIterator e(*mesh, 0); !e.end(); ++e)
    {
        auto num_cells = e->num_entities(tdim);
        for (std::size_t i = 0; i < num_cells; ++i)
        {
            df::Cell cell(*mesh, e->entities(tdim)[i]);
            volumes[dof_indices[j]] += cell.volume();
        }
        j++;
    }
    if(voronoi)
    {
        for (std::size_t i = 0; i < num_dofs; ++i)
        {
            volumes[i] = (gdim + 1.0) / volumes[i];
        }
    }else{
        for (std::size_t i = 0; i < num_dofs; ++i)
        {
            volumes[i] = 1.0 / volumes[i];
        }
    }

    return volumes;
}

std::vector<double> weighted_element_volume(const df::FunctionSpace &V)
{
    auto g_dim = V.mesh()->geometry().dim();
    std::shared_ptr<df::Form> volume;
    df::PETScVector volume_vector;
    auto V_ptr = std::make_shared<const df::FunctionSpace>(V);
    if (g_dim == 1)
    {
        volume = std::make_shared<WeightedVolume::Form_form1D>(V_ptr);
    }
    else if (g_dim == 2)
    {
        volume = std::make_shared<WeightedVolume::Form_form2D>(V_ptr);
    }
    else if (g_dim == 3)
    {
        volume = std::make_shared<WeightedVolume::Form_form3D>(V_ptr);
    }
    df::assemble(volume_vector, *volume);
    std::vector<double> volumes;
    volume_vector.get_local(volumes);
    for(std::size_t i = 0; i<volumes.size(); ++i)
    {
        volumes[i] = 1.0/volumes[i];
    }
    return volumes;
}


// df::Function distribute(const df::FunctionSpace &V,
//                         Population &pop,
//                         const std::vector<double> &dv_inv)
// {
//     auto mesh = V.mesh();
//     auto tdim = mesh->topology().dim();
//     df::Function rho(std::make_shared<const df::FunctionSpace>(V));
//     auto rho_vec = rho.vector();
//     std::size_t len_rho = rho_vec->size();
//     std::vector<double> rho0(len_rho);
//     rho_vec->get_local(rho0);

//     auto element = V.element();
//     auto s_dim = element->space_dimension();

//     std::vector<double> basis_matrix(s_dim);
//     std::vector<double> vertex_coordinates;

//     for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
//     {
//         auto cell_id = e->index();
//         df::Cell _cell(*mesh, cell_id);
//         _cell.get_vertex_coordinates(vertex_coordinates);
//         auto cell_orientation = _cell.orientation();
//         auto dof_id = V.dofmap()->cell_dofs(cell_id);

//         std::vector<double> basis(1);
//         std::vector<double> accum(s_dim, 0.0);

//         std::size_t num_particles = pop.cells[cell_id].particles.size();
//         for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
//         {
//             auto particle = pop.cells[cell_id].particles[p_id];

//             for (std::size_t i = 0; i < s_dim; ++i)
//             {
//                 element->evaluate_basis(i, basis.data(),
//                                         particle.x,
//                                         vertex_coordinates.data(),
//                                         cell_orientation);
//                 basis_matrix[i] = basis[0];
//                 accum[i] += particle.q * basis_matrix[i];
//             }

//         }
//         for (std::size_t i = 0; i < s_dim; ++i)
//         {
//             rho0[dof_id[i]] += accum[i];
//         }
//     }
//     for (std::size_t i = 0; i < len_rho; ++i)
//     {
//         rho0[i] *= dv_inv[i];
//     }
//     rho.vector()->set_local(rho0);
//     return rho;
// }

// df::Function distribute_new(const df::FunctionSpace &V,
//                         Population &pop,
//                         const std::vector<double> &dv_inv)
// {
//     auto mesh = V.mesh();
//     df::Function rho(std::make_shared<const df::FunctionSpace>(V));
//     auto rho_vec = rho.vector();
//     std::size_t len_rho = rho_vec->size();
//     std::vector<double> rho0(len_rho);
//     rho_vec->get_local(rho0);

//     auto element = V.element();
//     auto s_dim = element->space_dimension();
//     auto v_dim = element->value_dimension(0);
//     auto n_dim = s_dim / v_dim;

//     double cell_coords[n_dim];

//     for(auto &cell: pop.cells)
//     {
//         auto dof_id = V.dofmap()->cell_dofs(cell.id);
//         std::vector<double> accum(n_dim, 0.0);
//         for (auto &particle : cell.particles)
//         {
//             matrix_vector_product(&cell_coords[0], cell.basis_matrix.data(),
//                                   particle.x, n_dim, n_dim);

//             for (std::size_t i = 0; i < n_dim; ++i)
//             {
//                 accum[i] += particle.q * cell_coords[i];
//             }
//         }

//         for (std::size_t i = 0; i < s_dim; ++i)
//         {
//             rho0[dof_id[i]] += accum[i];
//         }
//     }
//     for (std::size_t i = 0; i < len_rho; ++i)
//     {
//         rho0[i] *= dv_inv[i];
//     }
//     rho.vector()->set_local(rho0);
//     return rho;
// }

// df::Function distribute_dg0_new(const df::FunctionSpace &Q, Population &pop)
// {
//     df::Function rho(std::make_shared<const df::FunctionSpace>(Q));
//     auto rho_vec = rho.vector();
//     std::size_t len_rho = rho_vec->size();
//     std::vector<double> rho0(len_rho);
//     rho_vec->get_local(rho0);

//     for(auto &cell: pop.cells)
//     {
//         auto dof_id = Q.dofmap()->cell_dofs(cell.id);
//         double accum = 0.0;
//         for (auto &particle : cell.particles)
//         {
//             accum += particle.q;
//         }
//         rho0[dof_id[0]] = accum / cell.volume();
//     }
//     rho.vector()->set_local(rho0);
//     return rho;
// }

// df::Function distribute_dg0(const df::FunctionSpace &Q, Population &pop)
// {
//     auto mesh = Q.mesh();
//     auto tdim = mesh->topology().dim();
//     df::Function rho(std::make_shared<const df::FunctionSpace>(Q));
//     auto rho_vec = rho.vector();
//     std::size_t len_rho = rho_vec->size();
//     std::vector<double> rho0(len_rho);
//     rho_vec->get_local(rho0);

//     for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
//     {
//         auto cell_id = e->index();
//         df::Cell _cell(*mesh, cell_id);
//         auto dof_id = Q.dofmap()->cell_dofs(cell_id);
//         double accum = 0.0;

//         std::size_t num_particles = pop.cells[cell_id].particles.size();
//         for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
//         {
//             auto particle = pop.cells[cell_id].particles[p_id];
//             accum += particle.q;
//         }
//         rho0[dof_id[0]] = accum/_cell.volume();
//     }
//     rho.vector()->set_local(rho0);
//     return rho;
// }

// static inline void matrix_vector_product(double *y, const double *A,
//                                          const double *x, std::size_t n,
//                                          std::size_t m)
// {
//     for (std::size_t i = 0; i < n; ++i)
//     {
//         y[i] = A[i * m];
//     }
//     for (std::size_t i = 0; i < n; ++i)
//     {
//         for (std::size_t j = 0; j < m - 1; ++j)
//         {
//             y[i] += A[i * m + j + 1] * x[j];
//         }
//     }
// }

}
