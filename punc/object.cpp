#include "object.h"

namespace punc
{

ObjectBoundary::ObjectBoundary(std::function<bool(const df::Array<double>&)> func):
                               SubDomain(), func(func){}

bool ObjectBoundary::inside(const df::Array<double>& x, bool on_boundary) const
{
    return on_boundary && func(x);
}

Object::Object(const std::shared_ptr<Potential::FunctionSpace> &V,
               const std::shared_ptr<ObjectBoundary> &sub_domain,
               const std::shared_ptr<df::Constant> &potential0,
               std::string method):
               df::DirichletBC(V, potential0, sub_domain, method), 
               V(V),
               sub_domain(sub_domain), 
               init_potential(potential0), 
               method(method)
{
    std::unordered_map<std::size_t, double> dof_map;
    get_boundary_values(dof_map);
    std::vector<int> dofs;

    for (auto it = dof_map.begin(); it != dof_map.end(); ++it)
    {
        dofs.push_back(it->first);
    }
    this->dofs = dofs;
}

void Object::add_charge(const double &q)
{
    charge += q;
}

void Object::set_potential(const double &pot)
{
    potential_ = pot;
    auto potential = std::make_shared<df::Constant>(potential_);
    set_value(potential);
}

void Object::compute_interpolated_charge(const std::shared_ptr<df::Function> &q_rho)
{
    interpolated_charge = 0.0;
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
        interpolated_charge += q_rho->vector()->getitem(dofs[i]);
    }
}

std::vector<double> Object::vertices()
{
    auto g_dim = V->mesh()->geometry().dim();
    auto coordinates = V->mesh()->coordinates();

    std::vector<std::size_t> d2v;
    d2v = df::dof_to_vertex_map(*V);
    std::vector<std::size_t> vertex_indices(dofs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
        vertex_indices[i] = d2v[dofs[i]];
    }
    std::vector<double> coords(g_dim * dofs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
        for (std::size_t j = 0; j < g_dim; ++j)
        {
            coords[i * g_dim + j] = coordinates[vertex_indices[i] * g_dim + j];
        }
    }
    return coords;
}

std::vector<std::size_t> Object::cells(const std::shared_ptr<df::FacetFunction<std::size_t> > &facet_func, 
                                       int &id)
{
    auto t_dim = V->mesh()->topology().dim();
    V->mesh()->init(t_dim - 1, t_dim);

    std::vector<std::size_t> cell_f;
    df::SubsetIterator f(*facet_func, id);
    for (; !f.end(); ++f)
    {
        cell_f.push_back(*f->entities(t_dim));
    }
    return cell_f;
}

void Object::mark_facets(std::shared_ptr<df::FacetFunction<std::size_t>> &facet_func, 
                         int id)
{
    sub_domain->mark(*facet_func, id);
}

void Object::mark_cells(std::shared_ptr<df::CellFunction<std::size_t>> &cell_func,
                        std::shared_ptr<df::FacetFunction<std::size_t>> &facet_func,
                        int id)
{
    std::vector<std::size_t> cells_vec;
    cells_vec = cells(facet_func, id);

    for (std::size_t i = 0; i < cells_vec.size(); ++i)
    {
        (*cell_func)[cells_vec[i]] = id;
    }
}

void compute_object_potentials(const std::shared_ptr<df::Function> &q,
                               const std::vector<std::shared_ptr<Object>> &objects,
                               const boost_matrix &inv_capacity)
{
    for (auto &obj : objects)
    {
        obj->compute_interpolated_charge(q);
    }
    int i, j;
    double potential;
    for (i = 0; i < objects.size(); ++i)
    {
        potential = 0.0;
        for (j = 0; j < objects.size(); ++j)
        {
            potential += (objects[j]->charge -\
                          objects[j]->interpolated_charge) * inv_capacity(i, j);
        }
        objects[i]->set_potential(potential);
    }
}


Circuit::Circuit(const std::vector<std::shared_ptr<Object>> &objects,
                 const boost_vector &precomputed_charge,
                 const boost_matrix &inv_bias,
                 double charge):
                 objects(objects),
                 precomputed_charge(precomputed_charge),
                 inv_bias(inv_bias), charge(charge) {}

void Circuit::circuit_charge()
{
    double c_charge = 0.0;
    for (std::size_t i = 0; i < objects.size(); ++i)
    {
        c_charge += (objects[i]->charge - objects[i]->interpolated_charge);
    }
    this->charge = c_charge;
}

void Circuit::redistribute_charge(const std::vector<double> &tot_charge)
{
    std::vector<double> redistr_charge(inv_bias.size1());
    for (std::size_t i = 0; i < inv_bias.size1(); ++i)
    {
        redistr_charge[i] = 0.0;
        for (std::size_t j = 0; j < inv_bias.size2(); ++j)
        {
            redistr_charge[i] += inv_bias(i, j) * tot_charge[j];
        }
    }

    for (std::size_t i = 0; i < objects.size(); ++i)
    {
        objects[i]->charge = precomputed_charge(i) + redistr_charge[i] +\
                             objects[i]->interpolated_charge;
    }
}

void redistribute_circuit_charge(const std::vector<std::shared_ptr<Circuit> > &circuits)
{
    std::size_t num_circuits = circuits.size();
    std::vector<double> tot_charge(num_circuits);
    for (std::size_t i = 0; i < num_circuits; ++i)
    {
        circuits[i]->circuit_charge();
        tot_charge[i] = circuits[i]->charge;
    }
    for (std::size_t i = 0; i < num_circuits; ++i)
    {
        circuits[i]->redistribute_charge(tot_charge);
    }
}
}