#include "../include/diagnostics.h"

namespace punc
{
    
double kinetic_energy(Population &pop)
{
    double KE = 0.0;
    for (df::MeshEntityIterator e(*pop.mesh, pop.tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            auto particle = pop.cells[cell_id].particles[p_id];
            auto m = particle.m;
            auto v = particle.v;
            for (std::size_t i = 0; i < pop.gdim; ++i)
            {
                KE += 0.5 * m * v[i] * v[i];
            }
        }
    }
}

double mesh_potential_energy(df::Function &phi, df::Function &rho)
{
    auto mesh = phi.function_space()->mesh();
    auto dim = mesh->geometry().dim();
    auto phi_ptr = std::make_shared<df::Function>(phi);
    auto rho_ptr = std::make_shared<df::Function>(rho);
    std::shared_ptr<df::Form> energy;
    if (dim == 1)
    {
        energy = std::make_shared<Energy::Form_0>(mesh, phi_ptr, rho_ptr);
    }
    else if (dim == 2)
    {
        energy = std::make_shared<Energy::Form_1>(mesh, phi_ptr, rho_ptr);
    }
    else if (dim == 3)
    {
        energy = std::make_shared<Energy::Form_2>(mesh, phi_ptr, rho_ptr);
    }
    return 0.5 * df::assemble(*energy);
}

double particle_potential_energy(Population &pop, const df::Function &phi)
{
    auto V = phi.function_space();
    auto mesh = V->mesh();
    auto tdim = mesh->topology().dim();
    auto element = V->element();
    auto s_dim = element->space_dimension();
    auto v_dim = element->value_dimension(0);

    double PE = 0.0;

    std::vector<std::vector<double>> basis_matrix;
    std::vector<double> coefficients(s_dim, 0.0);
    std::vector<double> vertex_coordinates;

    for (df::MeshEntityIterator e(*mesh, tdim); !e.end(); ++e)
    {
        auto cell_id = e->index();
        df::Cell _cell(*mesh, cell_id);
        _cell.get_vertex_coordinates(vertex_coordinates);
        auto cell_orientation = _cell.orientation();

        ufc::cell ufc_cell;
        _cell.get_cell_data(ufc_cell);

        phi.restrict(&coefficients[0], *element, _cell,
                     vertex_coordinates.data(), ufc_cell);

        std::vector<double> basis(v_dim);
        basis_matrix.resize(v_dim);
        for (std::size_t i = 0; i < v_dim; ++i)
        {
            basis_matrix[i].resize(s_dim);
        }

        std::size_t num_particles = pop.cells[cell_id].particles.size();
        for (std::size_t p_id = 0; p_id < num_particles; ++p_id)
        {
            std::vector<double> phii(v_dim, 0.0);
            auto particle = pop.cells[cell_id].particles[p_id];
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                element->evaluate_basis(i, basis.data(),
                                        particle.x.data(),
                                        vertex_coordinates.data(),
                                        cell_orientation);

                for (std::size_t j = 0; j < v_dim; ++j)
                {
                    basis_matrix[j][i] = basis[j];
                }
            }
            for (std::size_t i = 0; i < s_dim; ++i)
            {
                for (std::size_t j = 0; j < v_dim; j++)
                {
                    phii[j] += coefficients[i] * basis_matrix[j][i];
                }
            }
            auto q = particle.q;
            for (std::size_t j = 0; j < v_dim; j++)
            {
                PE += 0.5 * q * phii[j];
            }
        }
    }
    return PE;
}

}
