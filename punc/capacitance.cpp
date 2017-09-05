#include "capacitance.h"

namespace punc
{

std::shared_ptr<df::FacetFunction<std::size_t>> markers(
                            std::shared_ptr<const df::Mesh> &mesh,
                            const std::vector<std::shared_ptr<Object>> &objects)
{
    auto num_objects = objects.size();
    auto facet_func = std::make_shared<df::FacetFunction<std::size_t>>(mesh);
    facet_func->set_all(num_objects);
    for (int i = 0; i < num_objects; ++i)
    {
        objects[i]->mark_facets(facet_func, i);
    }
    return facet_func;
}

std::vector<std::shared_ptr<df::Function>> solve_laplace(
    const std::shared_ptr<Potential::FunctionSpace> &V,
    const std::shared_ptr<PoissonSolver> &poisson,
    const std::shared_ptr<NonPeriodicBoundary> &non_periodic_bnd,
    const std::vector<std::shared_ptr<Object>> &objects)
{
    auto poisson_bc = poisson->bc;
    auto phi_bnd = std::make_shared<df::Constant>(0.0);
    auto bnd = std::make_shared<df::DirichletBC>(V, phi_bnd, non_periodic_bnd);
    poisson->bc = {bnd}; 
    auto num_objects = objects.size();

    std::vector<std::shared_ptr<df::Function>> object_e_field(num_objects);
    for (int i = 0; i < num_objects; ++i)
    {
        for (int j = 0; j < num_objects; ++j)
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
        auto rho = std::make_shared<df::Function>(V);
        auto phi = std::make_shared<df::Function>(V);
        poisson->solve(phi, rho, objects);
        object_e_field[i] = electric_field(phi);
    }
    poisson->bc = poisson_bc;
    return object_e_field;
}

boost_matrix capacitance_matrix(
                const std::shared_ptr<Potential::FunctionSpace> &V,
                const std::shared_ptr<PoissonSolver> &poisson,
                const std::shared_ptr<NonPeriodicBoundary> &non_periodic_bnd,
                const std::vector<std::shared_ptr<Object>> &objects)
{
    auto mesh = V->mesh();
    auto facet_func = markers(mesh, objects);
    auto num_objects = objects.size();

    boost_matrix capacitance(num_objects, num_objects);
    boost_matrix inv_capacity(num_objects, num_objects);
    auto object_e_field = solve_laplace(V, poisson, non_periodic_bnd, objects);

    std::vector<std::shared_ptr<df::SubsetIterator>> subset_itr(num_objects);
    for (auto i = 0; i < num_objects; ++i)
    {
        auto f = std::make_shared<df::SubsetIterator>(*facet_func, i);
        subset_itr[i] = f;
    }

    for (unsigned i = 0; i < num_objects; ++i)
    {
        facet_func->set_all(num_objects);
        auto f = *subset_itr[i];
        for (; !f.end(); ++f)
        {
            facet_func->set_value(f->index(), 0);
        }
        Flux::Form_flux flux(mesh, object_e_field[i]);
        flux.ds = facet_func;
        for (unsigned j = 0; j < num_objects; ++j)
        {
            flux.e = object_e_field[j];
            capacitance(i, j) = df::assemble(flux);
        }
    }
    inv(capacitance, inv_capacity);
    std::cout << "C: " << capacitance << ", inv(C): " << inv_capacity << '\n';
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
    std::cout << "B: " << bias_matrix << ", inv(B): " << inv_bias << '\n';
    return inv_bias;
}

}